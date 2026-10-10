"""OpenCode 会话头注入插件。

MaiBot 的 OpenAI 兼容模型请求默认不带任何会话头，而 OpenCode 网关
（``opencode.ai``）在缺少会话头时会直接返回 ``400 MissingSessionID``。
本插件以「插件 LLM Provider」的形式接管被显式指向它的模型请求：

* 命中 ``opencode.ai``（含子域，可配置）时，注入稳定的 ``X-OpenCode-Session``；
* 令牌由常见会话头（``Session-Id`` 等）经 SHA-256 截断哈希得到，
  原始会话 ID 永不出网；没有可用会话头时使用持久化随机种子兜底；
* 未命中时原样透传，不添加任何头，也不改变请求体 —— 其他 Provider 不受影响。

注意：主程序不允许插件覆盖内置 ``client_type``，因此需要在模型配置里把
OpenCode 的 API Provider 的 ``client_type`` 设为 ``opencode-session``。
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
    is_endpoint_unsupported,
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
    resolve_alphabet,
    strip_session_prefix,
)

#: 本插件对外声明的 LLM Provider 客户端类型，需与 ``_manifest.json`` 保持一致。
PROVIDER_CLIENT_TYPE = "opencode-session"

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

#: 支持的线上协议。
PROTOCOL_CHAT_COMPLETIONS = "chat_completions"
PROTOCOL_RESPONSES = "responses"
PROTOCOL_AUTO = "auto"
_VALID_PROTOCOLS = frozenset({PROTOCOL_CHAT_COMPLETIONS, PROTOCOL_RESPONSES, PROTOCOL_AUTO})

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
    config_version: str = Field(default="1.1.0", description="配置版本")


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


class WireSectionConfig(PluginConfigBase):
    """请求协议配置。"""

    __ui_label__ = "请求协议"
    __ui_icon__ = "plug"
    __ui_order__ = 3

    wire_protocol: str = Field(
        default=PROTOCOL_CHAT_COMPLETIONS,
        description=(
            "chat_completions：POST /chat/completions；"
            "responses：POST /responses；"
            "auto：先用 chat_completions，端点不存在（404/405/501）时自动改用 responses。"
        ),
        json_schema_extra={"placeholder": "chat_completions"},
    )
    responses_store: bool = Field(
        default=False,
        description="仅 responses 协议使用：是否让网关保存响应（store）。默认 false。",
    )


class OpenCodeSessionHeaderConfig(PluginConfigBase):
    """插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    endpoint: EndpointSectionConfig = Field(default_factory=EndpointSectionConfig)
    session_header: SessionHeaderSectionConfig = Field(default_factory=SessionHeaderSectionConfig)
    wire: WireSectionConfig = Field(default_factory=WireSectionConfig)


class OpenCodeSessionHeaderPlugin(MaiBotPlugin, LLMProviderBase):
    """OpenCode 会话头注入插件。"""

    config_model = OpenCodeSessionHeaderConfig

    def __init__(self) -> None:
        """初始化插件内部状态。"""

        super().__init__()
        self._transport = JsonTransport()
        self._seed_store: SessionSeedStore | None = None
        #: ``wire_protocol = auto`` 时记住「该端点只支持 responses」的结论。
        self._protocol_overrides: dict[str, str] = {}

    # ── 生命周期 ────────────────────────────────────────────────

    async def on_load(self) -> None:
        """插件加载时执行。"""

        suffix_text = ", ".join(self.config.endpoint.host_suffixes) or "<空>"
        self.ctx.logger.info(
            f"OpenCode 会话头注入插件已加载：匹配域名 [{suffix_text}]，"
            f"注入头 {self.config.session_header.header_name}，"
            f"令牌长度 {clamp_token_length(self.config.session_header.token_length)}，"
            f"线上协议 {self._configured_protocol()}"
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
        self.ctx.logger.info("OpenCode 会话头注入插件配置已热重载")

    # ── LLM Provider ───────────────────────────────────────────

    @LLMProvider(
        PROVIDER_CLIENT_TYPE,
        handler_name="dispatch",
        name="OpenCode 会话头注入",
        description="转发 OpenAI 兼容请求，并在命中 opencode.ai 端点时注入稳定的 X-OpenCode-Session。",
        version="1.1.2",
    )
    async def dispatch(
        self,
        operation: str = "",
        request: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """按操作类型分发宿主请求。

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
        if operation != "response":
            raise ValueError(f"OpenCode 会话头注入 Provider 暂不支持操作类型: {operation!r}")
        return await self.get_response(dict(request or {}))

    async def get_response(self, request: dict[str, Any]) -> dict[str, Any]:
        """按配置的线上协议转发一次请求，并按需注入会话头。

        Args:
            request: 宿主序列化后的响应请求。

        Returns:
            dict[str, Any]: 响应字典，包含 ``content`` / ``reasoning_content`` / ``tool_calls`` / ``usage``。

        Raises:
            RuntimeError: 传输失败或网关返回异常时抛出，宿主会据此重试或切换模型。
        """

        config = self.config
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
        protocol = self._resolve_protocol(base_url)
        path, body = self._build_protocol_request(protocol, request, model_info)

        fallback_note = ""
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
            fallback_protocol = self._fallback_protocol(protocol, exc, base_url)
            if fallback_protocol is None:
                raise RuntimeError(f"OpenCode 会话头注入 Provider 请求失败: {exc}") from exc
            self.ctx.logger.warning(
                f"OpenCode 会话头注入: 端点 {path} 不可用（{exc.status_code}），"
                f"自动改用 {fallback_protocol} 重试"
            )
            fallback_note = f" fallback={protocol}->{fallback_protocol}"
            protocol = fallback_protocol
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
            except OpenCodeTransportError as retry_exc:
                raise RuntimeError(f"OpenCode 会话头注入 Provider 请求失败: {retry_exc}") from retry_exc

        if config.session_header.verbose:
            self.ctx.logger.info(
                "OpenCode 会话头注入: "
                f"endpoint={transport_result.endpoint} protocol={protocol}{fallback_note} matched={matched} "
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
        ],
    )
    async def opencode_session_token(
        self,
        base_url: str = "",
        session_hint: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """计算指定 Base URL 的注入结果。

        Args:
            base_url: 待检查的 Base URL。
            session_hint: 可选的模拟会话值。
            **kwargs: 宿主传入的额外参数。

        Returns:
            dict[str, Any]: 供模型阅读的结果字典。
        """

        del kwargs
        config = self.config
        matched = self._endpoint_matched(base_url)

        if session_hint.strip():
            seed = SessionSeed(value=strip_session_prefix(session_hint), source="session_hint")
        else:
            store = self._seed_store_instance()
            key = f"tool|{extract_host(base_url) or 'unknown'}"
            seed = SessionSeed(value=store.get(key), source=f"fallback:{key}")

        token = derive_session_token(
            seed.value,
            salt=config.session_header.salt,
            length=config.session_header.token_length,
            alphabet=config.session_header.token_alphabet,
        )
        headers = {name: token for name in self._header_names()} if matched else {}

        if matched:
            summary = f"{normalize_base_url(base_url) or '<空>'} 命中 OpenCode 端点，将注入 {headers}。"
        else:
            summary = f"{normalize_base_url(base_url) or '<空>'} 未命中 OpenCode 端点，不会注入任何头。"

        protocol = self._resolve_protocol(base_url)
        path = RESPONSES_PATH if protocol == PROTOCOL_RESPONSES else DEFAULT_CHAT_COMPLETIONS_PATH
        return {
            "content": f"{summary} 线上协议 {protocol}，端点 {path}。",
            "matched": matched,
            "headers": headers,
            "wire_protocol": protocol,
            "endpoint_path": path,
            "token_length": len(token),
        }

    # ── 内部实现 ────────────────────────────────────────────────

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

    # ── 协议选择 ────────────────────────────────────────────────

    def _configured_protocol(self) -> str:
        """返回配置中的线上协议。

        Returns:
            str: ``chat_completions`` / ``responses`` / ``auto`` 之一；非法值回退为 ``chat_completions``。
        """

        configured = str(self.config.wire.wire_protocol or PROTOCOL_CHAT_COMPLETIONS).strip().lower()
        return configured if configured in _VALID_PROTOCOLS else PROTOCOL_CHAT_COMPLETIONS

    def _endpoint_key(self, base_url: str) -> str:
        """返回用于记忆协议回退结论的端点键。

        Args:
            base_url: 当前 Base URL。

        Returns:
            str: 端点键。
        """

        return f"{extract_host(base_url) or 'unknown'}|{normalize_base_url(base_url)}"

    def _resolve_protocol(self, base_url: str) -> str:
        """解析本次请求应使用的线上协议。

        ``auto`` 模式默认走 chat_completions，只有在本端点被确认不支持该端点后
        才改用 responses（结论按端点记忆，避免每次都多打一次 404）。

        Args:
            base_url: 当前 Base URL。

        Returns:
            str: 实际使用的协议。
        """

        configured = self._configured_protocol()
        if configured != PROTOCOL_AUTO:
            return configured
        return self._protocol_overrides.get(self._endpoint_key(base_url), PROTOCOL_CHAT_COMPLETIONS)

    def _fallback_protocol(self, protocol: str, error: BaseException, base_url: str) -> str | None:
        """判断是否需要回退到另一种协议。

        Args:
            protocol: 刚失败的协议。
            error: 传输异常。
            base_url: 当前 Base URL。

        Returns:
            str | None: 需要回退时返回目标协议，否则返回 ``None``。
        """

        if self._configured_protocol() != PROTOCOL_AUTO:
            return None
        if protocol != PROTOCOL_CHAT_COMPLETIONS:
            return None
        if not is_endpoint_unsupported(error):
            return None
        self._protocol_overrides[self._endpoint_key(base_url)] = PROTOCOL_RESPONSES
        return PROTOCOL_RESPONSES

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
        messages = build_chat_messages(request.get("context_items") or [], has_tools=bool(tools))

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
        input_items = build_responses_input(request.get("context_items") or [], has_tools=bool(tools))

        # 先铺扩展字段，再补规范字段，避免扩展字段覆盖关键结构。
        body: dict[str, Any] = {}
        body.update(self._collect_extra_body(model_info.get("extra_params"), _RESPONSES_RESERVED_BODY_KEYS))
        body.update(self._collect_extra_body(request.get("extra_params"), _RESPONSES_RESERVED_BODY_KEYS))

        body.setdefault("model", str(model_info.get("model_identifier") or ""))
        body.setdefault("input", input_items)
        body.setdefault("store", bool(self.config.wire.responses_store))
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
            f"线上协议：{self._configured_protocol()}"
            f"（chat_completions 或 responses；auto 会在端点不支持时自动改用 responses）",
            f"Provider 类型：{PROVIDER_CLIENT_TYPE}（请在模型配置的 API Provider 中选用）",
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
