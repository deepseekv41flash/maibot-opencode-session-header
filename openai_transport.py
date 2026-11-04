"""OpenAI 兼容 HTTP 传输层。

主路径（Chat Completions + Bearer 鉴权）改由官方 ``openai`` SDK 的
:class:`AsyncOpenAI` 发送：连接池（1000/100）、分层超时、直连网关与重试策略
都由 SDK 负责，插件不再自己维护小连接池。

``/responses`` 端点与 header/query/none 鉴权方式仍走自带的 httpx 实现，
保证插件原有行为不变。所有异常都转成带可读原因的
:class:`OpenCodeTransportError`，便于宿主日志直接定位问题。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

try:  # openai 是宿主自带依赖；缺失时全部退回纯 httpx 实现
    import openai
    from openai import AsyncOpenAI
except Exception:  # noqa: BLE001 - 老环境可能没有 openai SDK
    openai = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment]

#: 出错时附带在异常信息里的响应体长度上限。
_ERROR_BODY_LIMIT = 600

DEFAULT_CHAT_COMPLETIONS_PATH = "/chat/completions"

#: Responses API 端点路径。
RESPONSES_PATH = "/responses"

#: 视为「该端点不存在」的状态码，用于在错误信息里给出可读提示。
UNSUPPORTED_ENDPOINT_STATUS = frozenset({404, 405, 501})

#: 由 SDK/httpx 自己管理的请求头，透传会破坏传输层行为。
_SDK_MANAGED_HEADERS = frozenset(
    {"authorization", "content-type", "content-length", "host", "transfer-encoding", "accept"}
)


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


def _sdk_http_module() -> Any:
    """返回 openai SDK 实际使用的 HTTP 库模块。

    Returns:
        Any: SDK 依赖的 httpx 兼容模块（新版为 ``httpx2``，旧版为 ``httpx``）。
    """

    if openai is None:
        return httpx
    try:
        from openai import _constants as sdk_constants
    except Exception:  # noqa: BLE001 - 私有模块不可用时退回 httpx
        return httpx
    return getattr(sdk_constants, "httpx2", None) or getattr(sdk_constants, "httpx", None) or httpx


class JsonTransport:
    """按端点选择 SDK 或 httpx 发送 JSON 请求的传输器。"""

    def __init__(self, *, connect_timeout: float = 15.0) -> None:
        """初始化传输器。

        Args:
            connect_timeout: 建立连接的超时时间（秒）。
        """

        self._connect_timeout = max(1.0, min(10.0, float(connect_timeout)))
        # 读超时固定 90s：必须短于宿主任务 hard_timeout（120s），否则插件与宿主
        # 会在同一时刻取消/超时，连接回收容易被打断，进而长期占住池槽位。
        self._request_timeout = 90.0
        # 池排队超时 10s：池满时快速失败并自愈，不再让请求干等 120s。
        self._pool_timeout = 10.0
        self._write_timeout = 30.0
        # 与 SDK 默认值对齐：1000 条连接 / 100 条保活，避免并发峰值被小池卡住。
        self._max_connections = 1000
        self._max_keepalive_connections = 100
        self._sdk_http = _sdk_http_module()
        self._sdk_client: Any = None
        self._sdk_key: tuple[Any, ...] | None = None
        self._sdk_http_client: Any = None
        self._client: httpx.AsyncClient | None = None

    # ── SDK 路径（Chat Completions + Bearer） ───────────────────

    def _sdk_timeout(self, read_timeout: float) -> Any:
        """构造 SDK 侧的分层超时。

        Args:
            read_timeout: 读取超时（秒）。

        Returns:
            Any: SDK 依赖的 ``Timeout`` 实例。
        """

        return self._sdk_http.Timeout(
            read=read_timeout,
            connect=self._connect_timeout,
            write=self._write_timeout,
            pool=self._pool_timeout,
        )

    def _get_sdk_http_client(self) -> Any:
        """惰性创建并复用 SDK 的 HTTP 连接池。

        Returns:
            Any: 与 SDK 同栈的 ``AsyncClient``。
        """

        client = self._sdk_http_client
        if client is None or getattr(client, "is_closed", False):
            client = self._sdk_http.AsyncClient(
                timeout=self._sdk_timeout(self._request_timeout),
                limits=self._sdk_http.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive_connections,
                ),
                follow_redirects=True,
                # 直连网关：系统代理（如 Clash）对 opencode.ai 的 TLS 握手偶发卡死，
                # 且 httpx 只认 HTTP(S)_PROXY/NO_PROXY，不读 Windows 系统代理例外表。
                trust_env=False,
            )
            self._sdk_http_client = client
        return client

    async def _close_sdk_client(self) -> None:
        """丢弃并关闭 SDK 客户端（连带关闭共享连接池）。"""

        client = self._sdk_client
        self._sdk_client = None
        self._sdk_key = None
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - 关闭失败不应影响后续调用
                pass
        # AsyncOpenAI 会一并关闭传入的 http_client，这里直接丢弃引用。
        self._sdk_http_client = None

    async def _get_sdk_client(
        self,
        *,
        base_url: str,
        api_key: str,
        default_headers: Mapping[str, str],
        default_query: Mapping[str, str],
    ) -> Any:
        """按 Provider 身份获取（必要时重建）SDK 客户端。

        Args:
            base_url: Provider 的 Base URL。
            api_key: Bearer 模式的 API Key。
            default_headers: 所有请求都要带的静态请求头。
            default_query: 所有请求都要带的静态查询参数。

        Returns:
            Any: ``AsyncOpenAI`` 实例。
        """

        key = (
            base_url,
            api_key,
            tuple(sorted(default_headers.items())),
            tuple(sorted(default_query.items())),
        )
        if self._sdk_client is not None and self._sdk_key == key:
            return self._sdk_client
        await self._close_sdk_client()
        sdk_client = AsyncOpenAI(
            api_key=api_key or "missing",
            base_url=base_url,
            max_retries=0,  # 重试由宿主负责，插件不重复发送长请求
            timeout=self._sdk_timeout(self._request_timeout),
            http_client=self._get_sdk_http_client(),
            default_headers=dict(default_headers) or None,
            default_query=dict(default_query) or None,
        )
        self._sdk_client = sdk_client
        self._sdk_key = key
        return sdk_client

    @staticmethod
    def _can_use_sdk(path: str, api_provider: Mapping[str, Any]) -> bool:
        """判断本次请求能否走 SDK 主路径。

        Args:
            path: 端点路径。
            api_provider: 宿主序列化后的 API Provider 配置。

        Returns:
            bool: Chat Completions 且为 Bearer 鉴权时返回 ``True``。
        """

        if AsyncOpenAI is None:
            return False
        normalized_path = "/" + str(path or "").strip().lstrip("/")
        if normalized_path != DEFAULT_CHAT_COMPLETIONS_PATH:
            return False
        auth_type = str(api_provider.get("auth_type") or "bearer").strip().lower()
        return auth_type == "bearer" and bool(str(api_provider.get("api_key") or ""))

    async def _handle_sdk_error(
        self,
        exc: BaseException,
        *,
        endpoint: str,
        read_timeout: float,
    ) -> None:
        """把 SDK 异常翻译成 :class:`OpenCodeTransportError`。

        Args:
            exc: SDK 抛出的异常。
            endpoint: 完整请求地址。
            read_timeout: 本次请求的读取超时（秒）。

        Raises:
            asyncio.CancelledError: 调用方取消时原样抛出。
            OpenCodeTransportError: 其余所有传输失败。
        """

        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise exc

        pool_timeout_type = getattr(self._sdk_http, "PoolTimeout", None)
        cause: BaseException | None = exc
        seen: set[int] = set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if pool_timeout_type is not None and isinstance(cause, pool_timeout_type):
                # 池满基本等于「连接没归还」：丢掉旧池重建，避免插件卡死到人工重载。
                await self._close_sdk_client()
                raise OpenCodeTransportError(
                    f"请求 {endpoint} 连接池排队超时（{self._pool_timeout:.0f}s，已重置连接池）: {exc}"
                ) from exc
            cause = cause.__cause__ or cause.__context__

        status_error_type = getattr(openai, "APIStatusError", None) if openai is not None else None
        if status_error_type is not None and isinstance(exc, status_error_type):
            status_code = int(getattr(exc, "status_code", 0) or 0)
            response = getattr(exc, "response", None)
            text = str(getattr(response, "text", "") or "")
            raise OpenCodeTransportError(
                self._status_error_message(status_code, text),
                status_code=status_code or None,
            ) from exc

        timeout_error_type = getattr(openai, "APITimeoutError", None) if openai is not None else None
        if timeout_error_type is not None and isinstance(exc, timeout_error_type):
            raise OpenCodeTransportError(f"请求 {endpoint} 超时（{read_timeout:.0f}s）: {exc}") from exc

        raise OpenCodeTransportError(f"请求 {endpoint} 失败: {exc}") from exc

    async def _post_with_sdk(
        self,
        *,
        base_url: str,
        endpoint: str,
        path: str,
        body: Mapping[str, Any],
        api_provider: Mapping[str, Any],
        headers: Mapping[str, str],
        params: Mapping[str, str],
        injected_headers: Mapping[str, str],
        read_timeout: float,
    ) -> JsonTransportResult:
        """用官方 SDK 发送一次 Chat Completions 请求。

        Args:
            base_url: Provider 的 Base URL。
            endpoint: 完整请求地址（仅用于错误信息）。
            path: 端点路径。
            body: 请求体。
            api_provider: 宿主序列化后的 API Provider 配置。
            headers: 已合并好的请求头（含鉴权）。
            params: 已合并好的查询参数。
            injected_headers: 本插件额外注入的请求头。
            read_timeout: 本次请求的读取超时（秒）。

        Returns:
            JsonTransportResult: 调用结果。

        Raises:
            OpenCodeTransportError: 网络异常、超时或网关返回非 2xx 时抛出。
        """

        static_headers = {
            key: value
            for key, value in _stringify_mapping(api_provider.get("default_headers")).items()
            if key.lower() not in _SDK_MANAGED_HEADERS
        }
        per_call_headers = {
            key: value
            for key, value in _stringify_mapping(injected_headers).items()
            if key.lower() not in _SDK_MANAGED_HEADERS
        }
        sdk_client = await self._get_sdk_client(
            base_url=base_url,
            api_key=str(api_provider.get("api_key") or ""),
            default_headers=static_headers,
            default_query=params,
        )

        model = str(body.get("model") or "")
        messages = list(body.get("messages") or [])
        # 除 model/messages 外全部走 extra_body 原样透传，保持与手写 HTTP 完全一致的线格式。
        extra_body = {
            key: value
            for key, value in body.items()
            if key not in ("model", "messages", "stream", "stream_options")
        }

        started = time.monotonic()
        try:
            raw = await sdk_client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                extra_body=extra_body or None,
                extra_headers=per_call_headers or None,
                timeout=self._sdk_timeout(read_timeout),
            )
        except BaseException as exc:  # noqa: BLE001 - 统一翻译成插件异常
            await self._handle_sdk_error(exc, endpoint=endpoint, read_timeout=read_timeout)
            raise

        latency_ms = int((time.monotonic() - started) * 1000)
        response = raw.http_response
        return self._finalize(
            status_code=int(getattr(response, "status_code", 200) or 200),
            text=str(getattr(response, "text", "") or ""),
            latency_ms=latency_ms,
            request_headers=headers,
            endpoint=endpoint,
            path=path,
        )

    # ── httpx 回退路径（/responses 与非 Bearer 鉴权） ────────────

    def _get_client(self) -> httpx.AsyncClient:
        """惰性创建并复用回退路径的 HTTP 客户端。

        Returns:
            httpx.AsyncClient: 共享客户端实例。
        """

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._request_timeout,
                    connect=self._connect_timeout,
                    write=self._write_timeout,
                    pool=self._pool_timeout,
                ),
                follow_redirects=True,
                trust_env=False,
                limits=httpx.Limits(
                    max_connections=self._max_connections,
                    max_keepalive_connections=self._max_keepalive_connections,
                ),
            )
        return self._client

    async def _reset_client(self) -> None:
        """丢弃并关闭回退路径客户端，下次调用时重建连接池。

        用于 ``PoolTimeout`` 之后自愈：池内残留的占用连接会被一并丢弃，
        避免插件必须靠人工重载才能恢复。
        """

        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - 重置失败不应影响后续调用
                pass

    async def _post_with_httpx(
        self,
        *,
        endpoint: str,
        path: str,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
        params: Mapping[str, str],
        read_timeout: float,
    ) -> JsonTransportResult:
        """用自带 httpx 客户端发送一次请求。

        Args:
            endpoint: 完整请求地址。
            path: 端点路径。
            body: 请求体。
            headers: 已合并好的请求头。
            params: 已合并好的查询参数。
            read_timeout: 本次请求的读取超时（秒）。

        Returns:
            JsonTransportResult: 调用结果。

        Raises:
            OpenCodeTransportError: 网络异常、超时或网关返回非 2xx 时抛出。
        """

        client = self._get_client()
        started = time.monotonic()
        try:
            response = await client.post(
                endpoint,
                json=dict(body),
                headers=dict(headers),
                params=dict(params) or None,
                timeout=httpx.Timeout(
                    read=read_timeout,
                    connect=self._connect_timeout,
                    write=self._write_timeout,
                    pool=self._pool_timeout,
                ),
            )
        except httpx.PoolTimeout as exc:
            await self._reset_client()
            raise OpenCodeTransportError(
                f"请求 {endpoint} 连接池排队超时（{self._pool_timeout:.0f}s，已重置连接池）: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise OpenCodeTransportError(f"请求 {endpoint} 超时（{read_timeout:.0f}s）: {exc}") from exc
        except httpx.HTTPError as exc:
            raise OpenCodeTransportError(f"请求 {endpoint} 失败: {exc}") from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        return self._finalize(
            status_code=response.status_code,
            text=response.text or "",
            latency_ms=latency_ms,
            request_headers=headers,
            endpoint=endpoint,
            path=path,
        )

    # ── 公共入口与结果处理 ─────────────────────────────────────

    async def aclose(self) -> None:
        """关闭底层客户端，释放连接池。"""

        await self._close_sdk_client()
        await self._reset_client()

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
        try:
            requested_timeout = float(timeout_seconds)
        except (TypeError, ValueError):
            requested_timeout = self._request_timeout
        read_timeout = max(30.0, min(requested_timeout, self._request_timeout))

        if self._can_use_sdk(path, api_provider):
            return await self._post_with_sdk(
                base_url=base_url,
                endpoint=endpoint,
                path=path,
                body=body,
                api_provider=api_provider,
                headers=headers,
                params=params,
                injected_headers=injected_headers,
                read_timeout=read_timeout,
            )
        return await self._post_with_httpx(
            endpoint=endpoint,
            path=path,
            body=body,
            headers=headers,
            params=params,
            read_timeout=read_timeout,
        )

    def _status_error_message(self, status_code: int, text: str) -> str:
        """构造「网关返回非 2xx」的可读错误信息。

        Args:
            status_code: 网关返回的状态码。
            text: 响应体文本。

        Returns:
            str: 错误信息。
        """

        excerpt = text[:_ERROR_BODY_LIMIT]
        hint = ""
        if "MissingSessionID" in text:
            hint = "（网关要求会话头；请确认插件已启用且本模型的 client_type 指向本插件）"
        elif status_code in UNSUPPORTED_ENDPOINT_STATUS:
            hint = (
                "（端点可能不被该网关支持；可把该 API Provider 的 client_type 换成 "
                "opencode-session 或 opencode-session-responses 再试）"
            )
        return f"网关返回 HTTP {status_code}{hint}，响应片段: {excerpt or '<空>'}"

    def _finalize(
        self,
        *,
        status_code: int,
        text: str,
        latency_ms: int,
        request_headers: Mapping[str, str],
        endpoint: str,
        path: str,
    ) -> JsonTransportResult:
        """校验状态码并解析响应体。

        Args:
            status_code: 网关返回的状态码。
            text: 响应体文本。
            latency_ms: 本次调用耗时（毫秒）。
            request_headers: 本次请求使用的请求头。
            endpoint: 完整请求地址。
            path: 端点路径。

        Returns:
            JsonTransportResult: 调用结果。

        Raises:
            OpenCodeTransportError: 非 2xx 或响应体不是合法 JSON 对象时抛出。
        """

        if status_code >= 400:
            raise OpenCodeTransportError(
                self._status_error_message(status_code, text),
                status_code=status_code,
            )

        try:
            payload = json.loads(text) if text else {}
        except (json.JSONDecodeError, ValueError) as exc:
            raise OpenCodeTransportError(
                f"网关返回内容不是合法 JSON: {text[:_ERROR_BODY_LIMIT] or '<空>'}",
                status_code=status_code,
            ) from exc
        if not isinstance(payload, dict):
            raise OpenCodeTransportError(
                "网关返回的 JSON 顶层不是对象",
                status_code=status_code,
            )

        return JsonTransportResult(
            status_code=status_code,
            payload=payload,
            latency_ms=latency_ms,
            request_headers=dict(request_headers),
            endpoint=endpoint,
        )
