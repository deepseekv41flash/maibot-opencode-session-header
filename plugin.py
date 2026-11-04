"""OpenCode 会话头注入插件。

MaiBot 的 OpenAI 兼容模型请求默认不带任何会话头，而 OpenCode 网关
（``opencode.ai``）在缺少会话头时会直接返回 ``400 MissingSessionID``。
本插件以「插件 LLM Provider」的形式接管被显式指向它的模型请求：

* 命中 ``opencode.ai``（含子域，可配置）时，注入稳定的 ``X-OpenCode-Session``；
* 令牌由常见会话头（``Session-Id`` 等）经 SHA-256 截断哈希得到，
  原始会话 ID 永不出网；没有可用会话头时使用持久化随机种子兜底；
* 未命中时原样透传，不添加任何头，也不改变请求体 —— 其他 Provider 不受影响。

注意：主程序不允许插件覆盖内置 ``client_type``，因此需要在模型配置里把
OpenCode 的 API Provider 的 ``client_type`` 设为下面二者之一：

* ``opencode-session`` —— 走 ``POST /chat/completions``；
* ``opencode-session-responses`` —— 走 ``POST /responses``。

两者可以同时存在，因此不同模型可以分别使用两种协议。
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from maibot_sdk import (
    Command,
    Field,
    LLMProvider,
    LLMProviderBase,
    MaiBotPlugin,
    PluginConfigBase,
    Tool,
)
from maibot_sdk.types import ToolParameterInfo, ToolParamType

from .openai_transport import (
    DEFAULT_CHAT_COMPLETIONS_PATH,
    RESPONSES_PATH,
    JsonTransport,
    OpenCodeTransportError,
    TransportSettings,
)
from .openai_wire import (
    build_chat_messages,
    build_chat_tools,
    build_responses_input,
    build_responses_text_config,
    build_responses_tools,
    build_response_format,
    build_tool_calls,
    build_usage,
    extract_response_content,
    parse_responses_output,
)
from .session_identity import (
    DEFAULT_CANDIDATE_HEADERS,
    DEFAULT_HOST_SUFFIXES,
    OPENCODE_SESSION_HEADER,
    SessionSeed,
    SessionSeedStore,
    clamp_token_length,
    derive_session_token,
    extract_host,
    extract_session_seed,
    is_opencode_endpoint,
    normalize_base_url,
    preview_seed,
    resolve_alphabet,
    strip_session_prefix,
)

#: 本插件对外声明的 LLM Provider 客户端类型，需与 ``_manifest.json`` 保持一致。
#: Chat Completions 协议（历史名称，保持向后兼容）。
PROVIDER_CLIENT_TYPE_CHAT = "opencode-session"
#: Responses 协议。
PROVIDER_CLIENT_TYPE_RESPONSES = "opencode-session-responses"

#: 请求体中由本插件显式承载、不应再从 ``extra_params`` 平铺进来的字段。
_RESERVED_BODY_KEYS = frozenset(
    {"max_tokens", "messages", "model", "response_format", "stream", "temperature", "tools"}
)

#: Responses 协议下同样由本插件显式承载的字段。
_RESPONSES_RESERVED_BODY_KEYS = _RESERVED_BODY_KEYS | {
    "input",
    "instructions",
    "max_output_tokens",
    "store",
    "text",
}

#: 两种线上协议；由模型配置里选用的 client_type 决定，不再有全局开关。
PROTOCOL_CHAT_COMPLETIONS = "chat_completions"
PROTOCOL_RESPONSES = "responses"

#: client_type → 线上协议。
PROTOCOL_BY_CLIENT_TYPE: dict[str, str] = {
    PROVIDER_CLIENT_TYPE_CHAT: PROTOCOL_CHAT_COMPLETIONS,
    PROVIDER_CLIENT_TYPE_RESPONSES: PROTOCOL_RESPONSES,
}

#: 协议 → 端点路径。
PATH_BY_PROTOCOL: dict[str, str] = {
    PROTOCOL_CHAT_COMPLETIONS: DEFAULT_CHAT_COMPLETIONS_PATH,
    PROTOCOL_RESPONSES: RESPONSES_PATH,
}

_COMMAND_PATTERN = r"^/(?:oc_session|opencode_session)\s*$"
_COMMAND_RESET_PATTERN = r"^/(?:oc_session_reset|opencode_session_reset)\s*$"


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(
        default=True,
        description="是否注入会话头。关闭后请求照常转发，但不再添加任何头。",
    )
    config_version: str = Field(default="1.4.0", description="配置版本")


class EndpointSectionConfig(PluginConfigBase):
    """端点匹配配置。"""

    __ui_label__ = "端点匹配"
    __ui_icon__ = "globe"
    __ui_order__ = 1

    host_suffixes: list[str] = Field(
        default_factory=lambda: list(DEFAULT_HOST_SUFFIXES),
        description="视为 OpenCode 的域名后缀，自动覆盖其子域。默认 opencode.ai。",
    )
    extra_base_url_prefixes: list[str] = Field(
        default_factory=list,
        description="额外的 Base URL 前缀，用于自建 OpenCode 网关（例如 https://gw.example.com/opencode）。",
    )


class SessionHeaderSectionConfig(PluginConfigBase):
    """会话头注入配置。"""

    __ui_label__ = "会话头注入"
    __ui_icon__ = "key-round"
    __ui_order__ = 2

    header_name: str = Field(
        default=OPENCODE_SESSION_HEADER,
        description="注入的会话头名称，OpenCode 网关读取 x-opencode-session。",
        json_schema_extra={"placeholder": OPENCODE_SESSION_HEADER},
    )
    extra_header_names: list[str] = Field(
        default_factory=list,
        description="可选：额外注入同名令牌的头，例如 x-session-affinity、x-client-request-id。",
    )
    token_length: int = Field(
        default=8,
        ge=4,
        le=32,
        description="截断后的令牌长度（4-32）。",
    )
    token_alphabet: str = Field(
        default="alphanumeric",
        description="令牌字符集：alphanumeric（纯字母数字）或 urlsafe（额外含 - 和 _）。",
        json_schema_extra={"placeholder": "alphanumeric"},
    )
    salt: str = Field(
        default="",
        description="可选盐值，用于隔离不同部署的令牌空间；修改后所有令牌都会变化。",
    )
    candidate_headers: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CANDIDATE_HEADERS),
        description="会话种子候选头，按优先级排列；先命中者参与哈希。",
    )
    session_id_env: str = Field(
        default="",
        description="可选：找不到会话头时读取的环境变量名作为兜底种子。",
        json_schema_extra={"placeholder": "MAIBOT_SESSION_ID"},
    )
    verbose: bool = Field(
        default=False,
        description="开启后每次注入都会在日志中记录端点、种子来源和令牌（令牌本身不是敏感信息）。",
    )


class TransportSectionConfig(PluginConfigBase):
    """HTTP 传输层配置。

    核心是**分相超时**：``pool_timeout`` 远小于 read 超时，连接池被占满时能在几秒内
    报出「本地连接池排队超时」，而不是挂 120 秒后伪装成网关超时（见 issue #1）。
    """

    __ui_label__ = "传输层"
    __ui_icon__ = "network"
    __ui_order__ = 3

    max_connections: int = Field(
        default=64,
        ge=8,
        le=512,
        description="连接池上限。非流式思考模型单请求可占用连接数十秒，池太小会被占满。",
    )
    max_keepalive_connections: int = Field(
        default=32,
        ge=0,
        le=256,
        description="保活连接数。",
    )
    connect_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="建连超时（秒）。",
    )
    write_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="发送请求体超时（秒）。",
    )
    pool_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="连接池排队超时（秒）。应远小于 read 超时，池满时快速失败。",
    )
    read_timeout_ratio: float = Field(
        default=0.9,
        ge=0.3,
        le=1.0,
        description="read 超时 = Provider timeout × 本比例；留出余量以便在宿主 RPC 超时前返回可读错误。",
    )
    trust_env: bool = Field(
        default=True,
        description="是否读取系统 HTTP(S)_PROXY 环境变量。若怀疑代理导致连接卡死，设 false 直连。",
    )


class OpenCodeSessionHeaderConfig(PluginConfigBase):
    """插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    endpoint: EndpointSectionConfig = Field(default_factory=EndpointSectionConfig)
    session_header: SessionHeaderSectionConfig = Field(default_factory=SessionHeaderSectionConfig)
    transport: TransportSectionConfig = Field(default_factory=TransportSectionConfig)


class OpenCodeSessionHeaderPlugin(MaiBotPlugin, LLMProviderBase):
    """OpenCode 会话头注入插件。"""

    config_model = OpenCodeSessionHeaderConfig

    def __init__(self) -> None:
        """初始化插件内部状态。"""

        super().__init__()
        self._transport = JsonTransport()
        self._transport_configured = False
        self._seed_store: SessionSeedStore | None = None

    # ── 生命周期 ────────────────────────────────────────────────

    async def on_load(self) -> None:
        """插件加载时执行。"""

        self._apply_transport_settings()
        suffix_text = ", ".join(self.config.endpoint.host_suffixes) or "<空>"
        self.ctx.logger.info(
            f"OpenCode 会话头注入插件已加载：匹配域名 [{suffix_text}]，"
            f"注入头 {self.config.session_header.header_name}，"
            f"令牌长度 {clamp_token_length(self.config.session_header.token_length)}，"
            f"客户端类型 {PROVIDER_CLIENT_TYPE_CHAT}={PROTOCOL_CHAT_COMPLETIONS}、"
            f"{PROVIDER_CLIENT_TYPE_RESPONSES}={PROTOCOL_RESPONSES}"
        )

    async def on_unload(self) -> None:
        """插件卸载时执行，释放连接池。"""

        await self._transport.aclose()
        self.ctx.logger.info("OpenCode 会话头注入插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        """配置热重载时执行。

        Args:
            scope: 重载范围。
            config_data: 新的配置数据。
            version: 配置版本。
        """

        del scope, config_data, version
        self._apply_transport_settings()
        self.ctx.logger.info("OpenCode 会话头注入插件配置已热重载")

    # ── LLM Provider ───────────────────────────────────────────

    @LLMProvider(
        PROVIDER_CLIENT_TYPE_CHAT,
        handler_name="dispatch_chat",
        name="OpenCode 会话头注入（Chat Completions）",
        description="转发 POST /chat/completions，命中 opencode.ai 时注入稳定的 X-OpenCode-Session。",
        version="1.4.0",
    )
    async def dispatch_chat(
        self,
        operation: str = "",
        request: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Chat Completions 协议的 Provider 入口。

        Args:
            operation: 操作类型，目前仅支持 ``response``。
            request: 宿主序列化后的 LLM 请求。
            **kwargs: 预留的额外参数。

        Returns:
            dict[str, Any]: 可由宿主恢复为 APIResponse 的响应字典。

        Raises:
            ValueError: 操作类型不受支持时抛出。
        """

        del kwargs
        return await self._serve(PROTOCOL_CHAT_COMPLETIONS, operation, request)

    @LLMProvider(
        PROVIDER_CLIENT_TYPE_RESPONSES,
        handler_name="dispatch_responses",
        name="OpenCode 会话头注入（Responses）",
        description="转发 POST /responses，命中 opencode.ai 时注入稳定的 X-OpenCode-Session。",
        version="1.4.0",
    )
    async def dispatch_responses(
        self,
        operation: str = "",
        request: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Responses 协议的 Provider 入口。

        Args:
            operation: 操作类型，目前仅支持 ``response``。
            request: 宿主序列化后的 LLM 请求。
            **kwargs: 预留的额外参数。

        Returns:
            dict[str, Any]: 可由宿主恢复为 APIResponse 的响应字典。

        Raises:
            ValueError: 操作类型不受支持时抛出。
        """

        del kwargs
        return await self._serve(PROTOCOL_RESPONSES, operation, request)

    async def _serve(
        self,
        protocol: str,
        operation: str,
        request: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """校验操作类型后转交 :meth:`get_response`。

        Args:
            protocol: 本次使用的线上协议。
            operation: 操作类型。
            request: 宿主序列化后的 LLM 请求。

        Returns:
            dict[str, Any]: 响应字典。

        Raises:
            ValueError: 操作类型不受支持时抛出。
        """

        if operation != "response":
            raise ValueError(f"OpenCode 会话头注入 Provider 暂不支持操作类型: {operation!r}")
        return await self.get_response(dict(request or {}), protocol=protocol)

    async def get_response(
        self,
        request: dict[str, Any],
        *,
        protocol: str = PROTOCOL_CHAT_COMPLETIONS,
    ) -> dict[str, Any]:
        """按指定协议转发一次请求，并按需注入会话头。

        Args:
            request: 宿主序列化后的响应请求。
            protocol: 线上协议，由调用方选用的 client_type 决定。

        Returns:
            dict[str, Any]: 响应字典，包含 ``content`` / ``reasoning_content`` / ``tool_calls`` / ``usage``。

        Raises:
            RuntimeError: 传输失败或网关返回异常时抛出，宿主会据此重试或切换模型。
        """

        config = self.config
        if not self._transport_configured:
            # on_load 未跑（例如异常路径直接进 Provider）时兜底应用一次。
            self._apply_transport_settings()
        api_provider = dict(request.get("api_provider") or {})
        model_info = dict(request.get("model_info") or {})
        base_url = str(api_provider.get("base_url") or "")

        matched = self._endpoint_matched(base_url)

        seed: SessionSeed | None = None
        injected: dict[str, str] = {}
        if config.plugin.enabled and matched:
            seed = self._resolve_seed(api_provider, model_info, request, base_url)
            token = derive_session_token(
                seed.value,
                salt=config.session_header.salt,
                length=config.session_header.token_length,
                alphabet=config.session_header.token_alphabet,
            )
            for header_name in self._header_names():
                injected[header_name] = token

        timeout_seconds = self._resolve_timeout(api_provider)
        path, body = self._build_protocol_request(protocol, request, model_info)

        try:
            transport_result = await self._transport.post_json(
                base_url=base_url,
                body=body,
                api_provider=api_provider,
                injected_headers=injected,
                timeout_seconds=timeout_seconds,
                path=path,
            )
        except OpenCodeTransportError as exc:
            raise RuntimeError(f"OpenCode 会话头注入 Provider 请求失败: {exc}") from exc

        if config.session_header.verbose:
            self.ctx.logger.info(
                "OpenCode 会话头注入: "
                f"endpoint={transport_result.endpoint} protocol={protocol} matched={matched} "
                f"injected={sorted(injected)} seed_source={seed.source if seed else '<未解析>'} "
                f"status={transport_result.status_code} latency={transport_result.latency_ms}ms"
            )

        if protocol == PROTOCOL_RESPONSES:
            return self._build_responses_result(transport_result.payload, model_info, api_provider)
        return self._build_chat_result(transport_result.payload, model_info, api_provider)

    # ── 命令 ───────────────────────────────────────────────────

    @Command("opencode_session", description="查看 OpenCode 会话头注入状态", pattern=_COMMAND_PATTERN)
    async def opencode_session_status(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, int]:
        """回复当前插件状态与各端点的令牌预览。

        Args:
            stream_id: 目标会话 ID。
            **kwargs: 宿主传入的额外参数。

        Returns:
            tuple[bool, str, int]: ``(是否成功, 回复文本, 拦截等级)``。
        """

        del kwargs
        text = self._build_status_text()
        await self._safe_send(text, stream_id)
        return True, text, 2

    @Command("opencode_session_reset", description="重置 OpenCode 会话兜底种子", pattern=_COMMAND_RESET_PATTERN)
    async def opencode_session_reset(self, stream_id: str = "", **kwargs: Any) -> tuple[bool, str, int]:
        """轮换兜底随机种子。

        Args:
            stream_id: 目标会话 ID。
            **kwargs: 宿主传入的额外参数。

        Returns:
            tuple[bool, str, int]: ``(是否成功, 回复文本, 拦截等级)``。
        """

        del kwargs
        store = self._seed_store_instance()
        store.rotate(None)
        text = "已清空 OpenCode 会话兜底种子；下次请求会重新生成，网关侧会看到新的会话令牌。"
        await self._safe_send(text, stream_id)
        return True, text, 2

    # ── 工具 ───────────────────────────────────────────────────

    @Tool(
        "opencode_session_token",
        description="计算给定 Base URL 会被注入的 OpenCode 会话头，用于排查 400 MissingSessionID。",
        parameters=[
            ToolParameterInfo(
                name="base_url",
                param_type=ToolParamType.STRING,
                description="要检查的 API Base URL，例如 https://opencode.ai/zen/v1",
                required=True,
            ),
            ToolParameterInfo(
                name="session_hint",
                param_type=ToolParamType.STRING,
                description="可选：模拟的会话值（如 Session-Id），留空则使用兜底种子",
                required=False,
            ),
            ToolParameterInfo(
                name="client_type",
                param_type=ToolParamType.STRING,
                description="可选：要预览的客户端类型，opencode-session（chat）或 opencode-session-responses",
                required=False,
            ),
        ],
    )
    async def opencode_session_token(
        self,
        base_url: str = "",
        session_hint: str = "",
        client_type: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """计算指定 Base URL 的注入结果。

        Args:
            base_url: 待检查的 Base URL。
            session_hint: 可选的模拟会话值。
            client_type: 可选，要预览的客户端类型；留空按 chat 预览。
            **kwargs: 宿主传入的额外参数。

        Returns:
            dict[str, Any]: 供模型阅读的结果字典。
        """

        del kwargs
        config = self.config
        matched = self._endpoint_matched(base_url)
        preview_note = ""

        if session_hint.strip():
            seed = SessionSeed(value=strip_session_prefix(session_hint), source="session_hint")
        else:
            # 排障路径只读：不创建、不落盘，避免模型逐域名试探把种子文件撑大。
            store = self._seed_store_instance()
            host = extract_host(base_url) or "unknown"
            key = f"tool|{host}"
            existing = store.peek(key)
            provider_entry = store.peek_host(host) if existing is None else None
            if existing is not None:
                seed = SessionSeed(value=existing, source=f"fallback:{key}")
            elif provider_entry is not None:
                provider_key, provider_value = provider_entry
                seed = SessionSeed(value=provider_value, source=f"fallback:{provider_key}")
            else:
                seed = SessionSeed(value=preview_seed(host), source=f"preview:{host}")
                preview_note = "（该主机尚无兜底种子，此处为不落盘的只读预览值）"

        token = derive_session_token(
            seed.value,
            salt=config.session_header.salt,
            length=config.session_header.token_length,
            alphabet=config.session_header.token_alphabet,
        )
        headers = {name: token for name in self._header_names()} if matched else {}

        if matched:
            summary = f"{normalize_base_url(base_url) or '<空>'} 命中 OpenCode 端点，将注入 {headers}。{preview_note}"
        else:
            summary = f"{normalize_base_url(base_url) or '<空>'} 未命中 OpenCode 端点，不会注入任何头。"

        selected = str(client_type or "").strip()
        protocol = PROTOCOL_BY_CLIENT_TYPE.get(selected, PROTOCOL_CHAT_COMPLETIONS)
        path = PATH_BY_PROTOCOL[protocol]
        return {
            "content": (
                f"{summary} 当前预览协议 {protocol}，端点 {path}。"
                f"两个 client_type 分别是：{PROVIDER_CLIENT_TYPE_CHAT} → /chat/completions，"
                f"{PROVIDER_CLIENT_TYPE_RESPONSES} → /responses。"
            ),
            "matched": matched,
            "headers": headers,
            "client_types": dict(PROTOCOL_BY_CLIENT_TYPE),
            "preview_client_type": selected or PROVIDER_CLIENT_TYPE_CHAT,
            "wire_protocol": protocol,
            "endpoint_path": path,
            "token_length": len(token),
        }

    # ── 内部实现 ────────────────────────────────────────────────

    def _apply_transport_settings(self) -> None:
        """把 ``[transport]`` 配置同步到 HTTP 传输层。"""

        config = self.config.transport
        self._transport.configure(
            TransportSettings(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_keepalive_connections,
                connect_timeout=config.connect_timeout,
                write_timeout=config.write_timeout,
                pool_timeout=config.pool_timeout,
                read_timeout_ratio=config.read_timeout_ratio,
                trust_env=config.trust_env,
            )
        )
        self._transport_configured = True

    def _seed_store_instance(self) -> SessionSeedStore:
        """取得（必要时创建）会话种子仓库。

        Returns:
            SessionSeedStore: 指向插件数据目录的种子仓库。
        """

        if self._seed_store is None:
            self._seed_store = SessionSeedStore(self.ctx.paths.data_dir / "session_seeds.json")
        return self._seed_store

    def _header_names(self) -> list[str]:
        """返回本次要注入的全部头名（已去重、保序）。

        Returns:
            list[str]: 小写头名列表。
        """

        config = self.config.session_header
        names: list[str] = []
        for raw_name in [config.header_name, *config.extra_header_names]:
            name = str(raw_name or "").strip()
            if name and name.lower() not in {existing.lower() for existing in names}:
                names.append(name.lower())
        return names

    def _endpoint_matched(self, base_url: str) -> bool:
        """判断给定 Base URL 是否需要注入会话头。

        Args:
            base_url: 待判断的 Base URL。

        Returns:
            bool: 命中返回 ``True``。
        """

        return is_opencode_endpoint(
            base_url,
            host_suffixes=self.config.endpoint.host_suffixes,
            extra_base_url_prefixes=self.config.endpoint.extra_base_url_prefixes,
        )

    def _resolve_seed(
        self,
        api_provider: Mapping[str, Any],
        model_info: Mapping[str, Any],
        request: Mapping[str, Any],
        base_url: str,
    ) -> SessionSeed:
        """按「显式头 → 环境变量 → 持久化兜底」的顺序解析会话种子。

        Args:
            api_provider: API Provider 配置快照。
            model_info: 模型信息快照。
            request: 完整请求快照。
            base_url: 当前 Base URL。

        Returns:
            SessionSeed: 会话种子及其来源。
        """

        sources = [
            api_provider.get("default_headers"),
            self._extract_headers(model_info.get("extra_params")),
            self._extract_headers(request.get("extra_params")),
        ]
        found = extract_session_seed(sources, self.config.session_header.candidate_headers)
        if found is not None:
            return found

        env_name = str(self.config.session_header.session_id_env or "").strip()
        if env_name:
            env_value = os.environ.get(env_name, "").strip()
            if env_value:
                return SessionSeed(value=strip_session_prefix(env_value), source=f"env:{env_name}")

        key = f"{api_provider.get('name') or 'provider'}|{extract_host(base_url) or 'unknown'}"
        return SessionSeed(value=self._seed_store_instance().get(key), source=f"fallback:{key}")

    @staticmethod
    def _extract_headers(extra_params: Any) -> Mapping[str, Any]:
        """从 ``extra_params`` 中取出 ``headers`` 子映射。

        Args:
            extra_params: 任意附加参数映射。

        Returns:
            Mapping[str, Any]: 请求头映射；不存在时返回空字典。
        """

        if not isinstance(extra_params, Mapping):
            return {}
        headers = extra_params.get("headers")
        return headers if isinstance(headers, Mapping) else {}

    def _resolve_timeout(self, api_provider: Mapping[str, Any]) -> float:
        """解析请求超时时间。

        Args:
            api_provider: API Provider 配置快照。

        Returns:
            float: 超时秒数。
        """

        try:
            return float(api_provider.get("timeout") or 120)
        except (TypeError, ValueError):
            return 120.0

    def _build_protocol_request(
        self,
        protocol: str,
        request: Mapping[str, Any],
        model_info: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """按协议构造端点路径与请求体。

        Args:
            protocol: 目标协议。
            request: 完整请求快照。
            model_info: 模型信息快照。

        Returns:
            tuple[str, dict[str, Any]]: ``(端点路径, 请求体)``。
        """

        if protocol == PROTOCOL_RESPONSES:
            return RESPONSES_PATH, self._build_responses_body(request, model_info)
        return DEFAULT_CHAT_COMPLETIONS_PATH, self._build_request_body(request, model_info)

    # ── 请求体构造 ──────────────────────────────────────────────

    def _build_request_body(
        self,
        request: Mapping[str, Any],
        model_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把宿主请求快照转换为 Chat Completions 请求体。

        Args:
            request: 完整请求快照。
            model_info: 模型信息快照。

        Returns:
            dict[str, Any]: 请求体。
        """

        tool_options = request.get("tool_options") or []
        tools = build_chat_tools(tool_options)
        wire_stats: dict[str, Any] = {}
        messages = build_chat_messages(
            request.get("context_items") or [], has_tools=bool(tools), stats=wire_stats
        )
        self._log_dropped_items(wire_stats, PROTOCOL_CHAT_COMPLETIONS)

        body: dict[str, Any] = {
            "model": str(model_info.get("model_identifier") or ""),
            "messages": messages,
        }
        if tools:
            body["tools"] = tools

        temperature = request.get("temperature")
        if temperature is not None:
            try:
                body["temperature"] = float(temperature)
            except (TypeError, ValueError):
                pass

        max_tokens = request.get("max_tokens")
        if max_tokens:
            try:
                body["max_tokens"] = int(max_tokens)
            except (TypeError, ValueError):
                pass

        response_format = build_response_format(request.get("response_format"))
        if response_format is not None:
            body["response_format"] = response_format

        body.update(self._collect_extra_body(model_info.get("extra_params"), _RESERVED_BODY_KEYS))
        body.update(self._collect_extra_body(request.get("extra_params"), _RESERVED_BODY_KEYS))
        return body

    def _log_dropped_items(self, stats: Mapping[str, Any], protocol: str) -> None:
        """Context Item 被跳过时记录日志。

        宿主已知且按设计没有可移植投影的类型（reasoning、Provider 原生活动、不透明项）
        只记 debug，避免刷屏；只有真正未知的类型才告警。

        Args:
            stats: 线格式投影写回的统计字典。
            protocol: 当前线上协议名。
        """

        unprojectable = int(stats.get("unprojectable_items") or 0)
        if unprojectable > 0:
            unprojectable_types = ", ".join(
                str(item) for item in (stats.get("unprojectable_types") or [])
            )
            self.ctx.logger.debug(
                f"OpenCode 会话头注入: 线格式 {protocol} 跳过了 {unprojectable} 个无投影的 Context Item"
                f"（类型：{unprojectable_types or '<空类型>'}），这是预期行为"
            )

        dropped = int(stats.get("dropped_items") or 0)
        if dropped <= 0:
            return
        dropped_types = ", ".join(str(item) for item in (stats.get("dropped_types") or []))
        self.ctx.logger.warning(
            f"OpenCode 会话头注入: 线格式 {protocol} 丢弃了 {dropped} 个无法识别的 Context Item"
            f"（类型：{dropped_types or '<空类型>'}）；通常意味着宿主新增了插件尚不认识的 Item 类型"
        )

    def _build_responses_body(
        self,
        request: Mapping[str, Any],
        model_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把宿主请求快照转换为 Responses API 请求体。

        Args:
            request: 完整请求快照。
            model_info: 模型信息快照。

        Returns:
            dict[str, Any]: 请求体。
        """

        tool_options = request.get("tool_options") or []
        tools = build_responses_tools(tool_options)
        wire_stats: dict[str, Any] = {}
        input_items = build_responses_input(
            request.get("context_items") or [], has_tools=bool(tools), stats=wire_stats
        )
        self._log_dropped_items(wire_stats, PROTOCOL_RESPONSES)

        # 先铺扩展字段，再补规范字段，避免扩展字段覆盖关键结构。
        body: dict[str, Any] = {}
        body.update(self._collect_extra_body(model_info.get("extra_params"), _RESPONSES_RESERVED_BODY_KEYS))
        body.update(self._collect_extra_body(request.get("extra_params"), _RESPONSES_RESERVED_BODY_KEYS))

        body.setdefault("model", str(model_info.get("model_identifier") or ""))
        body.setdefault("input", input_items)
        # 固定 store=false：OpenCode 网关不是原生 Responses 目标，无法持久化响应。
        # 开启 store 会让请求直接 503（Endpoint is unavailable），且插件并不发送
        # previous_response_id，存了也不会取回，只会带来 30 天留存成本。
        body["store"] = False
        if tools:
            body.setdefault("tools", tools)

        temperature = request.get("temperature")
        if temperature is not None:
            try:
                body.setdefault("temperature", float(temperature))
            except (TypeError, ValueError):
                pass

        max_tokens = request.get("max_tokens")
        if max_tokens:
            try:
                body.setdefault("max_output_tokens", int(max_tokens))
            except (TypeError, ValueError):
                pass

        text_config = build_responses_text_config(request.get("response_format"))
        if text_config is not None:
            body.setdefault("text", text_config)
        return body

    @staticmethod
    def _collect_extra_body(extra_params: Any, reserved_keys: frozenset[str]) -> dict[str, Any]:
        """按宿主规则把 ``extra_params`` 平铺为请求体扩展字段。

        Args:
            extra_params: 模型级或请求级附加参数。
            reserved_keys: 由插件显式承载、不再平铺的键集合。

        Returns:
            dict[str, Any]: 需要合并进请求体的字段。
        """

        if not isinstance(extra_params, Mapping):
            return {}
        raw_params = dict(extra_params)
        body: dict[str, Any] = {}
        nested_body = raw_params.pop("body", None)
        if isinstance(nested_body, Mapping):
            body.update(dict(nested_body))
        raw_params.pop("headers", None)
        raw_params.pop("query", None)
        for key, value in raw_params.items():
            if key in reserved_keys:
                continue
            body[key] = value
        return body

    # ── 响应构造 ────────────────────────────────────────────────

    def _build_chat_result(
        self,
        payload: Mapping[str, Any],
        model_info: Mapping[str, Any],
        api_provider: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把 Chat Completions 响应转换为宿主可消费的响应字典。

        Args:
            payload: 网关返回的 JSON。
            model_info: 模型信息快照。
            api_provider: API Provider 配置快照。

        Returns:
            dict[str, Any]: 响应字典。

        Raises:
            RuntimeError: 响应中缺少 ``choices`` 时抛出。
        """

        choices = payload.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
            raise RuntimeError(f"OpenCode 网关响应缺少 choices 字段: {str(payload)[:300]}")

        first_choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = first_choice.get("message")
        message_payload = message if isinstance(message, Mapping) else {}

        content, reasoning = extract_response_content(message_payload)
        return self._assemble_result(
            content=content,
            reasoning=reasoning,
            tool_calls=build_tool_calls(message_payload),
            payload=payload,
            model_info=model_info,
            api_provider=api_provider,
            response_id=str(payload.get("id") or ""),
            status="completed",
            finish_reason=str(first_choice.get("finish_reason") or ""),
            model=str(payload.get("model") or model_info.get("model_identifier") or ""),
        )

    def _build_responses_result(
        self,
        payload: Mapping[str, Any],
        model_info: Mapping[str, Any],
        api_provider: Mapping[str, Any],
    ) -> dict[str, Any]:
        """把 Responses 响应转换为宿主可消费的响应字典。

        Args:
            payload: 网关返回的 JSON。
            model_info: 模型信息快照。
            api_provider: API Provider 配置快照。

        Returns:
            dict[str, Any]: 响应字典。

        Raises:
            RuntimeError: 响应状态为 failed / incomplete / cancelled 时抛出。
        """

        parsed = parse_responses_output(payload)
        if parsed.incomplete_reason:
            raise RuntimeError(f"Responses 请求未完整完成（{parsed.status}）: {parsed.incomplete_reason}")

        return self._assemble_result(
            content=parsed.content,
            reasoning=parsed.reasoning,
            tool_calls=parsed.tool_calls,
            payload=payload,
            model_info=model_info,
            api_provider=api_provider,
            response_id=parsed.response_id,
            status=parsed.status or "completed",
            finish_reason=parsed.status or "completed",
            model=str(payload.get("model") or model_info.get("model_identifier") or ""),
        )

    @staticmethod
    def _assemble_result(
        *,
        content: str,
        reasoning: str,
        tool_calls: list[dict[str, Any]],
        payload: Mapping[str, Any],
        model_info: Mapping[str, Any],
        api_provider: Mapping[str, Any],
        response_id: str,
        status: str,
        finish_reason: str,
        model: str,
    ) -> dict[str, Any]:
        """装配两种协议共用的响应字典。

        Args:
            content: 正文。
            reasoning: 推理内容。
            tool_calls: 工具调用列表。
            payload: 原始响应体。
            model_info: 模型信息快照。
            api_provider: API Provider 配置快照。
            response_id: 网关响应 ID。
            status: 响应状态。
            finish_reason: 结束原因。
            model: 实际模型名。

        Returns:
            dict[str, Any]: 响应字典。
        """

        model_name = str(model_info.get("name") or model_info.get("model_identifier") or "")
        provider_name = str(api_provider.get("name") or "")
        usage = build_usage(payload.get("usage"), model_name=model_name, provider_name=provider_name)

        result: dict[str, Any] = {
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "response_id": response_id,
            "status": status,
            "finish_reason": finish_reason,
            "model": model,
            "raw_data": dict(payload),
        }
        if usage is not None:
            result["usage"] = usage
        return result

    def _build_status_text(self) -> str:
        """构造状态命令的回复文本。

        Returns:
            str: 可直接发送给用户的文本。
        """

        config = self.config
        store = self._seed_store_instance()
        snapshot = store.snapshot()

        lines = [
            "【OpenCode 会话头注入】",
            f"注入开关：{'开启' if config.plugin.enabled else '关闭（仅透传）'}",
            f"匹配域名：{', '.join(config.endpoint.host_suffixes) or '<空>'}",
            f"额外前缀：{', '.join(config.endpoint.extra_base_url_prefixes) or '<无>'}",
            f"注入头名：{', '.join(self._header_names()) or '<空>'}",
            f"令牌长度：{clamp_token_length(config.session_header.token_length)}"
            f"（字符集：{config.session_header.token_alphabet or 'alphanumeric'}，"
            f"实际字符数 {len(resolve_alphabet(config.session_header.token_alphabet))}）",
            "客户端类型（在模型配置的 API Provider 里二选一）：",
            f"  · {PROVIDER_CLIENT_TYPE_CHAT} → POST /chat/completions",
            f"  · {PROVIDER_CLIENT_TYPE_RESPONSES} → POST /responses",
        ]

        if snapshot:
            lines.append("兜底种子令牌预览：")
            for key, value in sorted(snapshot.items()):
                token = derive_session_token(
                    value,
                    salt=config.session_header.salt,
                    length=config.session_header.token_length,
                    alphabet=config.session_header.token_alphabet,
                )
                lines.append(f"  · {key} → {token}")
        else:
            lines.append("兜底种子：尚未生成（首次请求 opencode.ai 时创建）")

        lines.append("提示：/opencode_session_reset 可轮换兜底种子。")
        return "\n".join(lines)

    async def _safe_send(self, text: str, stream_id: str) -> None:
        """发送文本消息，失败只记录日志。

        Args:
            text: 消息文本。
            stream_id: 目标会话 ID。
        """

        if not stream_id:
            self.ctx.logger.warning("OpenCode 会话头注入：命令缺少 stream_id，无法发送回复")
            return
        try:
            await self.ctx.send.text(text, stream_id)
        except Exception as exc:  # noqa: BLE001 - 发送失败不应让命令报错
            self.ctx.logger.error(f"OpenCode 会话头注入：发送状态消息失败: {exc}")


def create_plugin() -> OpenCodeSessionHeaderPlugin:
    """创建插件实例，宿主固定按这个名字找工厂函数。

    Returns:
        OpenCodeSessionHeaderPlugin: 插件实例。
    """

    return OpenCodeSessionHeaderPlugin()
