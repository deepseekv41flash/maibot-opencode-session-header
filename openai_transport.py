"""OpenAI 兼容 HTTP 传输层。

只做一件事：把已经拼好的请求体发到 ``{base_url}/chat/completions``，
并按 Provider 配置补上鉴权头。所有异常都转成带可读原因的
:class:`OpenCodeTransportError`，便于宿主日志直接定位问题。

连接池与超时按「分相」配置（见 :class:`TransportSettings`）：``pool`` 排队超时远小于
``read``，这样连接池被占满时会在几秒内报出明确原因，而不是挂 120 秒后伪装成网关超时。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

#: 出错时附带在异常信息里的响应体长度上限。
_ERROR_BODY_LIMIT = 600

DEFAULT_CHAT_COMPLETIONS_PATH = "/chat/completions"

#: Responses API 端点路径。
RESPONSES_PATH = "/responses"

#: 视为「该端点不存在」的状态码，用于在错误信息里给出可读提示。
UNSUPPORTED_ENDPOINT_STATUS = frozenset({404, 405, 501})


class OpenCodeTransportError(RuntimeError):
    """HTTP 传输或网关返回异常。

    Attributes:
        status_code: 网关返回的 HTTP 状态码；网络层失败时为 ``None``。
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        """初始化异常。

        Args:
            message: 可读的错误描述。
            status_code: 网关返回的 HTTP 状态码。
        """

        super().__init__(message)
        self.status_code = status_code


def is_endpoint_unsupported(error: BaseException) -> bool:
    """判断异常是否表示「该端点不被支持」。

    Args:
        error: 待判断的异常。

    Returns:
        bool: 状态码属于 :data:`UNSUPPORTED_ENDPOINT_STATUS` 时返回 ``True``。
    """

    return isinstance(error, OpenCodeTransportError) and error.status_code in UNSUPPORTED_ENDPOINT_STATUS


@dataclass(slots=True)
class JsonTransportResult:
    """一次 JSON 端点调用的结果。"""

    status_code: int
    payload: dict[str, Any]
    latency_ms: int
    request_headers: dict[str, str]
    endpoint: str


def build_endpoint(base_url: str, path: str = DEFAULT_CHAT_COMPLETIONS_PATH) -> str:
    """拼接完整请求地址。

    Args:
        base_url: Provider 的 Base URL，允许带或不带尾部斜杠。
        path: 端点路径。

    Returns:
        str: 完整 URL。
    """

    normalized_base = str(base_url or "").strip().rstrip("/")
    normalized_path = "/" + str(path or DEFAULT_CHAT_COMPLETIONS_PATH).strip().lstrip("/")
    return f"{normalized_base}{normalized_path}"


def build_auth_headers(api_provider: Mapping[str, Any]) -> dict[str, str]:
    """按 Provider 配置构造鉴权请求头。

    Args:
        api_provider: 宿主序列化后的 API Provider 配置。

    Returns:
        dict[str, str]: 需要附加的鉴权头；``auth_type`` 为 ``none``/``query`` 时为空。
    """

    auth_type = str(api_provider.get("auth_type") or "bearer").strip().lower()
    api_key = str(api_provider.get("api_key") or "")

    if auth_type == "bearer":
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if auth_type == "header":
        header_name = str(api_provider.get("auth_header_name") or "Authorization").strip()
        prefix = str(api_provider.get("auth_header_prefix") or "").strip()
        if not header_name:
            return {}
        value = f"{prefix} {api_key}".strip() if prefix else api_key
        return {header_name: value} if value else {}
    return {}


def build_auth_query(api_provider: Mapping[str, Any]) -> dict[str, str]:
    """按 Provider 配置构造鉴权查询参数。

    Args:
        api_provider: 宿主序列化后的 API Provider 配置。

    Returns:
        dict[str, str]: 需要附加的查询参数；非 ``query`` 模式时为空。
    """

    if str(api_provider.get("auth_type") or "").strip().lower() != "query":
        return {}
    query_name = str(api_provider.get("auth_query_name") or "api_key").strip()
    api_key = str(api_provider.get("api_key") or "")
    if not query_name or not api_key:
        return {}
    return {query_name: api_key}


def _stringify_mapping(value: Any) -> dict[str, str]:
    """把任意映射规范化为字符串字典。

    Args:
        value: 待规范化的值。

    Returns:
        dict[str, str]: 字符串字典。
    """

    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if item is None:
            continue
        name = str(key).strip()
        if not name:
            continue
        normalized[name] = str(item)
    return normalized


def _strip_header_conflicts(headers: Mapping[str, str]) -> dict[str, str]:
    """删除会破坏 httpx 自身行为的请求头。

    Args:
        headers: 待过滤的请求头。

    Returns:
        dict[str, str]: 过滤后的请求头。
    """

    blocked = {"content-length", "host", "transfer-encoding"}
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


@dataclass(frozen=True, slots=True)
class TransportSettings:
    """HTTP 传输层参数。

    连接的**分相超时**是这里的关键：把 ``pool``（连接池排队）与 ``read``（等响应）
    拆开，池满时能在几秒内快速失败并给出明确原因，而不是伪装成「网关超时 120s」。
    """

    max_connections: int = 64
    """连接池上限。非流式思考模型单请求可占用连接数十秒，池太小会被占满。"""

    max_keepalive_connections: int = 32
    """保活连接数。"""

    connect_timeout: float = 10.0
    """建连超时（秒）。"""

    write_timeout: float = 30.0
    """发送请求体超时（秒）。"""

    pool_timeout: float = 10.0
    """连接池排队超时（秒）。应当远小于 read 超时，否则池满时会长时间挂起。"""

    read_timeout_ratio: float = 0.9
    """read 超时 = Provider ``timeout`` × 本比例。

    留出的余量用于在宿主 RPC 到点前返回可读错误（宿主 RPC 超时等于 Provider timeout）。
    """

    trust_env: bool = True
    """是否读取系统 ``HTTP(S)_PROXY`` 环境变量。

    关闭可避免代理层 TLS 握手卡死时长期占用连接池；直连场景建议关闭。
    """


def resolve_read_timeout(settings: TransportSettings, timeout_seconds: float) -> float:
    """计算本次请求的 read 超时。

    Args:
        settings: 传输层参数。
        timeout_seconds: Provider 配置的请求超时。

    Returns:
        float: read 阶段超时（秒）。
    """

    ratio = max(0.1, min(1.0, float(settings.read_timeout_ratio)))
    return max(1.0, float(timeout_seconds) * ratio)


def build_request_timeout(settings: TransportSettings, timeout_seconds: float) -> httpx.Timeout:
    """按分相超时构造本次请求的 :class:`httpx.Timeout`。

    Args:
        settings: 传输层参数。
        timeout_seconds: Provider 配置的请求超时。

    Returns:
        httpx.Timeout: 分相超时对象。
    """

    return httpx.Timeout(
        connect=max(1.0, float(settings.connect_timeout)),
        read=resolve_read_timeout(settings, timeout_seconds),
        write=max(1.0, float(settings.write_timeout)),
        pool=max(1.0, float(settings.pool_timeout)),
    )


class JsonTransport:
    """复用同一个 ``httpx.AsyncClient`` 的 JSON 端点传输器。"""
    def __init__(self, settings: TransportSettings | None = None) -> None:
        """初始化传输器。

        Args:
            settings: 传输层参数；省略时使用默认值。
        """

        self._settings = settings or TransportSettings()
        self._client: httpx.AsyncClient | None = None
        #: 配置热更新后退休的旧客户端，等卸载时统一关闭（避免中断在途请求）。
        self._retired: list[httpx.AsyncClient] = []

    @property
    def settings(self) -> TransportSettings:
        """返回当前传输层参数。

        Returns:
            TransportSettings: 当前参数。
        """

        return self._settings

    def configure(self, settings: TransportSettings) -> None:
        """更新传输层参数；参数变化时重建客户端。

        旧客户端不立即关闭（可能仍有在途请求），而是移入退休列表，由
        :meth:`aclose` 统一释放。配置热更新本身很少发生，这样做最安全。

        Args:
            settings: 新的传输层参数。
        """

        if settings == self._settings:
            return
        self._settings = settings
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            self._retired.append(client)

    def _get_client(self) -> httpx.AsyncClient:
        """惰性创建并复用 HTTP 客户端。

        Returns:
            httpx.AsyncClient: 共享客户端实例。
        """

        if self._client is None or self._client.is_closed:
            settings = self._settings
            # 客户端默认超时仅作兜底：每次请求都会通过 build_request_timeout 覆盖。
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=max(1.0, float(settings.connect_timeout)),
                    read=60.0,
                    write=max(1.0, float(settings.write_timeout)),
                    pool=max(1.0, float(settings.pool_timeout)),
                ),
                follow_redirects=True,
                trust_env=bool(settings.trust_env),
                limits=httpx.Limits(
                    max_connections=max(1, int(settings.max_connections)),
                    max_keepalive_connections=max(0, int(settings.max_keepalive_connections)),
                ),
            )
        return self._client

    async def aclose(self) -> None:
        """关闭底层的当前客户端与所有退休客户端。"""

        clients = [client for client in (self._client, *self._retired) if client is not None]
        self._client = None
        self._retired.clear()
        for client in clients:
            if client.is_closed:
                continue
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - 关闭失败不应影响插件卸载
                pass

    async def post_json(
        self,
        *,
        base_url: str,
        body: Mapping[str, Any],
        api_provider: Mapping[str, Any],
        injected_headers: Mapping[str, str],
        timeout_seconds: float,
        path: str = DEFAULT_CHAT_COMPLETIONS_PATH,
    ) -> JsonTransportResult:
        """向指定 JSON 端点发起一次请求。

        Args:
            base_url: Provider 的 Base URL。
            body: 请求体。
            api_provider: 宿主序列化后的 API Provider 配置。
            injected_headers: 本插件额外注入的请求头（仅 opencode.ai 命中时非空）。
            timeout_seconds: 请求超时时间（秒）。
            path: 端点路径。

        Returns:
            JsonTransportResult: 调用结果。

        Raises:
            OpenCodeTransportError: 网络异常、超时或网关返回非 2xx 时抛出。
        """

        headers: dict[str, str] = {}
        headers.update(_stringify_mapping(api_provider.get("default_headers")))
        headers.update(build_auth_headers(api_provider))
        headers.update(_stringify_mapping(injected_headers))
        headers = _strip_header_conflicts(headers)
        headers.setdefault("Content-Type", "application/json")
        headers.setdefault("Accept", "application/json")

        params: dict[str, str] = {}
        params.update(_stringify_mapping(api_provider.get("default_query")))
        params.update(build_auth_query(api_provider))

        endpoint = build_endpoint(base_url, path)
        client = self._get_client()
        settings = self._settings
        read_timeout = resolve_read_timeout(settings, timeout_seconds)
        started = time.monotonic()
        try:
            response = await client.post(
                endpoint,
                json=dict(body),
                headers=headers,
                params=params or None,
                timeout=build_request_timeout(settings, timeout_seconds),
            )
        except httpx.PoolTimeout as exc:
            # 注意：PoolTimeout 是 TimeoutException 的子类，必须排在前面捕获。
            raise OpenCodeTransportError(
                f"本地连接池排队超时（{settings.pool_timeout:.0f}s，上限 {settings.max_connections} 条）："
                "请求还没发出去就被放弃，通常是上游响应太慢把连接池占满了。"
                "可调大 transport.max_connections，或把 transport.trust_env 设为 false 排除代理卡死",
            ) from exc
        except httpx.TimeoutException as exc:
            raise OpenCodeTransportError(
                f"请求 {endpoint} 超时"
                f"（read={read_timeout:.0f}s / connect={settings.connect_timeout:.0f}s / "
                f"pool={settings.pool_timeout:.0f}s）: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OpenCodeTransportError(f"请求 {endpoint} 失败: {exc}") from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        text = response.text or ""

        if response.status_code >= 400:
            excerpt = text[:_ERROR_BODY_LIMIT]
            hint = ""
            if "MissingSessionID" in text:
                hint = "（网关要求会话头；请确认插件已启用且本模型的 client_type 指向本插件）"
            elif response.status_code in UNSUPPORTED_ENDPOINT_STATUS:
                hint = (
                    f"（端点 {path} 可能不被该网关支持；"
                    "可把该 API Provider 的 client_type 换成 opencode-session 或 "
                    "opencode-session-responses 再试）"
                )
            raise OpenCodeTransportError(
                f"网关返回 HTTP {response.status_code}{hint}，响应片段: {excerpt or '<空>'}",
                status_code=response.status_code,
            )

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise OpenCodeTransportError(
                f"网关返回内容不是合法 JSON: {text[:_ERROR_BODY_LIMIT] or '<空>'}",
                status_code=response.status_code,
            ) from exc
        if not isinstance(payload, dict):
            raise OpenCodeTransportError(
                "网关返回的 JSON 顶层不是对象",
                status_code=response.status_code,
            )

        return JsonTransportResult(
            status_code=response.status_code,
            payload=payload,
            latency_ms=latency_ms,
            request_headers=headers,
            endpoint=endpoint,
        )
