"""OpenAI 兼容 HTTP 传输层。

只做一件事：把已经拼好的请求体发到 ``{base_url}/chat/completions``，
并按 Provider 配置补上鉴权头。所有异常都转成带可读原因的
:class:`OpenCodeTransportError`，便于宿主日志直接定位问题。
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

#: 视为「该端点不存在」的状态码，用于 ``wire_protocol = auto`` 的协议回退。
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


class JsonTransport:
    """复用同一个 ``httpx.AsyncClient`` 的 JSON 端点传输器。"""

    def __init__(self, *, connect_timeout: float = 15.0) -> None:
        """初始化传输器。

        Args:
            connect_timeout: 建立连接的超时时间（秒）。
        """

        self._connect_timeout = max(1.0, float(connect_timeout))
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """惰性创建并复用 HTTP 客户端。

        Returns:
            httpx.AsyncClient: 共享客户端实例。
        """

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._connect_timeout, connect=min(10.0, self._connect_timeout)),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
        return self._client

    async def aclose(self) -> None:
        """关闭底层客户端，释放连接池。"""

        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
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
        started = time.monotonic()
        try:
            response = await client.post(
                endpoint,
                json=dict(body),
                headers=headers,
                params=params or None,
                timeout=max(1.0, float(timeout_seconds)),
            )
        except httpx.TimeoutException as exc:
            raise OpenCodeTransportError(f"请求 {endpoint} 超时（{timeout_seconds:.0f}s）: {exc}") from exc
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
                hint = f"（端点 {path} 可能不被该网关支持；可尝试切换 wire_protocol）"
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
