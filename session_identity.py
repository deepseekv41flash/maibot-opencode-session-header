"""OpenCode 会话身份推导。

本模块只做三件事，且全部是纯函数或纯本地状态：

1. 判断一个 Base URL 是否属于 opencode.ai 端点（含子域）。
2. 从"常见会话头"里挑出稳定的会话种子（``Session-Id`` 等）。
3. 把会话种子单向散列成固定长度的匿名令牌 —— 原始会话 ID 永不出网。

设计要点：OpenCode 网关用请求头携带会话 ID，缺头会直接返回
``400 MissingSessionID``。会话 ID 本身是内部标识，直接透传给第三方
网关既没必要也有隐私风险，因此这里始终发送 SHA-256 截断哈希。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

#: OpenCode 网关读取的会话头名称（小写）。
OPENCODE_SESSION_HEADER = "x-opencode-session"

#: 默认视为 OpenCode 的域名后缀；``opencode.ai`` 同时覆盖其所有子域。
DEFAULT_HOST_SUFFIXES: tuple[str, ...] = ("opencode.ai",)

#: 常见会话头候选，按优先级排列；先命中者作为会话种子。
DEFAULT_CANDIDATE_HEADERS: tuple[str, ...] = (
    "x-opencode-session",
    "session-id",
    "x-session-id",
    "session_id",
    "x-session-affinity",
    "x-client-request-id",
    "x-request-id",
    "conversation-id",
    "x-conversation-id",
    "thread-id",
    "x-thread-id",
    "chat-id",
    "x-chat-id",
)

_ALPHABET_ALPHANUMERIC = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_ALPHABET_URLSAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

#: 令牌字符集：``alphanumeric`` 为纯字母数字，``urlsafe`` 额外包含 ``-`` 与 ``_``。
ALPHABETS: dict[str, str] = {
    "alphanumeric": _ALPHABET_ALPHANUMERIC,
    "urlsafe": _ALPHABET_URLSAFE,
}

#: 形如 ``session-`` / ``sess_`` 的前缀不参与哈希，避免同一会话换前缀后令牌漂移。
_SESSION_PREFIX_PATTERN = re.compile(r"^(?:session|sess|sid)[-_]", re.IGNORECASE)

MIN_TOKEN_LENGTH = 4
MAX_TOKEN_LENGTH = 32


@dataclass(frozen=True, slots=True)
class SessionSeed:
    """一次会话种子解析的结果。"""

    value: str
    """真正参与哈希的原始值（已剥离 ``session-`` 前缀）。"""

    source: str
    """命中的来源描述，例如 ``header:session-id`` 或 ``fallback:store``。"""


def resolve_alphabet(name: str) -> str:
    """把配置中的字符集名称解析为实际字符表。

    Args:
        name: 字符集名称，未知名称回退到 ``alphanumeric``。

    Returns:
        str: 可用于编码的字符表。
    """

    key = str(name or "").strip().lower()
    return ALPHABETS.get(key, _ALPHABET_ALPHANUMERIC)


def clamp_token_length(length: Any) -> int:
    """把令牌长度限制在安全区间内。

    Args:
        length: 原始长度值。

    Returns:
        int: 位于 ``[4, 32]`` 的长度。
    """

    try:
        parsed = int(length)
    except (TypeError, ValueError):
        parsed = 8
    return max(MIN_TOKEN_LENGTH, min(MAX_TOKEN_LENGTH, parsed))


def extract_host(base_url: str) -> str:
    """从 Base URL 中提取小写主机名。

    Args:
        base_url: 形如 ``https://opencode.ai/zen/v1`` 的地址；缺少协议时也能解析。

    Returns:
        str: 主机名；无法解析时返回空字符串。
    """

    value = str(base_url or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = "//" + value.lstrip("/")
    try:
        return (urlsplit(value).hostname or "").lower()
    except ValueError:
        return ""


def normalize_base_url(base_url: str) -> str:
    """去掉尾部斜杠，便于做前缀比较。

    Args:
        base_url: 原始地址。

    Returns:
        str: 规范化后的地址。
    """

    return str(base_url or "").strip().rstrip("/")


def is_opencode_endpoint(
    base_url: str,
    *,
    host_suffixes: Sequence[str] = DEFAULT_HOST_SUFFIXES,
    extra_base_url_prefixes: Sequence[str] = (),
) -> bool:
    """判断 Base URL 是否指向 OpenCode 端点。

    命中规则（满足其一即可）：

    1. 主机名等于某个后缀，或为其子域（``opencode.ai`` 命中 ``zen.opencode.ai``）；
    2. 规范化后的 URL 以 ``extra_base_url_prefixes`` 中的某一项开头（自定义网关场景）。

    Args:
        base_url: 待判断的 Base URL。
        host_suffixes: 域名后缀列表。
        extra_base_url_prefixes: 额外的 URL 前缀列表。

    Returns:
        bool: 命中返回 ``True``。
    """

    host = extract_host(base_url)
    if host:
        for raw_suffix in host_suffixes or ():
            suffix = str(raw_suffix or "").strip().strip(".").lower()
            if suffix and (host == suffix or host.endswith("." + suffix)):
                return True

    normalized = normalize_base_url(base_url).lower()
    if normalized:
        for raw_prefix in extra_base_url_prefixes or ():
            prefix = normalize_base_url(raw_prefix).lower()
            if prefix and normalized.startswith(prefix):
                return True
    return False


def normalize_headers(headers: Any) -> dict[str, str]:
    """把请求头映射规范化为「小写键 → 去空白字符串值」。

    Args:
        headers: 任意请求头映射。

    Returns:
        dict[str, str]: 规范化后的请求头；非法输入返回空字典。
    """

    normalized: dict[str, str] = {}
    if not isinstance(headers, Mapping):
        return normalized
    for key, value in headers.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        normalized[str(key).strip().lower()] = text
    return normalized


def strip_session_prefix(value: str) -> str:
    """剥离 ``session-`` 一类前缀，保留稳定标识本体。

    Args:
        value: 原始会话值，例如 ``session-6f1c...``。

    Returns:
        str: 剥离前缀后的值；无前缀时原样返回。
    """

    text = str(value or "").strip()
    if not text:
        return ""
    return _SESSION_PREFIX_PATTERN.sub("", text, count=1) or text


def extract_session_seed(
    sources: Iterable[Any],
    candidate_headers: Sequence[str] = DEFAULT_CANDIDATE_HEADERS,
) -> SessionSeed | None:
    """按候选头优先级，从多个请求头来源中提取会话种子。

    外层按候选头排序、内层按来源排序，因此「同一优先级下更靠前的来源」优先，
    但更高优先级的头名总能胜出（例如 ``x-opencode-session`` 优先于 ``session-id``）。

    Args:
        sources: 请求头映射序列，靠前者优先级更高。
        candidate_headers: 候选头名称列表。

    Returns:
        SessionSeed | None: 命中时返回种子及其来源；全部未命中返回 ``None``。
    """

    normalized_sources = [normalize_headers(source) for source in sources]
    for raw_name in candidate_headers or ():
        name = str(raw_name or "").strip().lower()
        if not name:
            continue
        for index, headers in enumerate(normalized_sources):
            raw_value = headers.get(name)
            if not raw_value:
                continue
            stripped = strip_session_prefix(raw_value)
            if not stripped:
                continue
            return SessionSeed(value=stripped, source=f"header:{name}#{index}")
    return None


def derive_session_token(
    seed: str,
    *,
    salt: str = "",
    length: int = 8,
    alphabet: str = "alphanumeric",
) -> str:
    """把会话种子单向散列为固定长度的稳定令牌。

    先对 ``salt\\x00seed`` 做 SHA-256，再把摘要按字符表做进制展开并截断。
    同一 ``(salt, seed, length, alphabet)`` 永远得到同一令牌，
    而令牌无法反推出原始会话 ID。

    Args:
        seed: 会话种子。
        salt: 可选盐值，用于隔离不同部署的令牌空间。
        length: 令牌长度，会被限制在 ``[4, 32]``。
        alphabet: 字符集名称，见 :data:`ALPHABETS`。

    Returns:
        str: 固定长度的令牌。

    Raises:
        ValueError: 当会话种子为空时抛出。
    """

    normalized_seed = str(seed or "").strip()
    if not normalized_seed:
        raise ValueError("会话种子不能为空")

    characters = resolve_alphabet(alphabet)
    size = len(characters)
    digest = hashlib.sha256(f"{salt}\x00{normalized_seed}".encode("utf-8")).digest()
    number = int.from_bytes(digest, "big")

    produced: list[str] = []
    for _ in range(clamp_token_length(length)):
        number, remainder = divmod(number, size)
        produced.append(characters[remainder])
    return "".join(produced)


class SessionSeedStore:
    """为没有显式会话头的端点维护稳定的回退种子。

    OpenCode 只要求「同一个会话稳定」，并不要求令牌可逆；
    当用户环境中不存在任何常见会话头时，用一份持久化随机种子兜底，
    既能满足网关要求，也不会把任何真实身份信息带出去。
    """

    def __init__(self, path: Path) -> None:
        """初始化种子仓库。

        Args:
            path: JSON 状态文件路径。
        """

        self._path = Path(path)
        self._seeds: dict[str, str] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        """返回状态文件路径。"""

        return self._path

    def _load(self) -> None:
        """惰性读取状态文件；文件损坏时按空状态处理，不影响主流程。"""

        if self._loaded:
            return
        self._loaded = True
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        seeds = payload.get("seeds")
        if isinstance(seeds, dict):
            self._seeds = {str(key): str(value) for key, value in seeds.items() if str(value).strip()}

    def _persist(self) -> None:
        """以「临时文件 + 原子替换」方式写回状态文件。"""

        payload = json.dumps({"version": 1, "seeds": self._seeds}, ensure_ascii=False, indent=2, sort_keys=True)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_suffix(self._path.suffix + ".tmp")
            temporary.write_text(payload + "\n", encoding="utf-8")
            os.replace(temporary, self._path)
        except OSError:
            # 状态文件只是优化项：写失败时本次运行仍使用内存中的种子。
            pass

    def get(self, key: str) -> str:
        """取得（必要时创建）指定键的稳定种子。

        Args:
            key: 端点标识，通常为 ``provider|host``。

        Returns:
            str: 稳定的随机种子。
        """

        self._load()
        normalized_key = str(key or "").strip() or "default"
        existing = self._seeds.get(normalized_key)
        if existing:
            return existing
        created = uuid.uuid4().hex
        self._seeds[normalized_key] = created
        self._persist()
        return created

    def rotate(self, key: str | None = None) -> str:
        """轮换指定键（或全部）的种子。

        Args:
            key: 目标键；为 ``None`` 时清空全部种子。

        Returns:
            str: 轮换后的种子；清空全部时返回空字符串。
        """

        self._load()
        if key is None:
            self._seeds.clear()
            self._persist()
            return ""
        normalized_key = str(key).strip() or "default"
        created = uuid.uuid4().hex
        self._seeds[normalized_key] = created
        self._persist()
        return created

    def snapshot(self) -> dict[str, str]:
        """返回当前种子的只读副本。

        Returns:
            dict[str, str]: 键到种子的映射副本。
        """

        self._load()
        return dict(self._seeds)
