"""Context Item 与 OpenAI Chat Completions 线格式之间的纯转换。

MaiBot 1.3.5 把模型请求统一表达为 ``Context Item`` 快照，而 OpenAI 兼容
网关只认 ``messages``/``tools``。本模块负责两个方向的映射，逻辑对齐主程序
``src/llm_models/model_client/openai_client.py`` 的投影规则，但保持纯函数、
无副作用，便于单独测试。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: 会折叠进同一条 assistant 消息的 Item 类型。
_ASSISTANT_GROUP_TYPES = frozenset(
    {
        "ReasoningItem",
        "AssistantMessageItem",
        "FunctionCallItem",
        "ProviderActivityItem",
        "ProviderOpaqueItem",
    }
)

#: 无工具请求时需要清洗掉的 Item 类型（部分兼容接口拒收悬空的工具链）。
_TOOL_CHAIN_TYPES = frozenset({"FunctionCallItem", "FunctionCallOutputItem"})

_JSON_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """把任意值安全地视为映射。

    Args:
        value: 待检查的值。

    Returns:
        Mapping[str, Any]: 原值（若为映射）或空字典。
    """

    return value if isinstance(value, Mapping) else {}


def _part_text(part: Mapping[str, Any]) -> str | None:
    """取出片段的文本投影。

    Args:
        part: 单个内容片段。

    Returns:
        str | None: 文本或拒答内容；非文本片段返回 ``None``。
    """

    part_type = str(part.get("type") or "").strip().lower()
    if part_type == "text":
        text = part.get("text")
        return text if isinstance(text, str) else None
    if part_type == "refusal":
        refusal = part.get("refusal")
        return refusal if isinstance(refusal, str) else None
    return None


def _normalize_image_part(part: Mapping[str, Any]) -> tuple[str, str] | None:
    """把图片片段规范化为 ``(格式, base64)``。

    Args:
        part: 单个内容片段。

    Returns:
        tuple[str, str] | None: 规范化结果；缺少必要字段时返回 ``None``。
    """

    image_format = str(part.get("image_format") or "").strip().lower()
    image_base64 = part.get("image_base64")
    if not image_format or not isinstance(image_base64, str) or not image_base64:
        return None
    if image_format in {"jpg", "jpeg"}:
        image_format = "jpeg"
    return image_format, image_base64


def _text_only_content(parts: Sequence[Mapping[str, Any]]) -> str | list[dict[str, Any]]:
    """把只允许文本的消息内容转换为 OpenAI 结构。

    Args:
        parts: 内容片段序列。

    Returns:
        str | list[dict[str, Any]]: 单文本片段返回字符串，否则返回文本片段数组。
    """

    if not parts:
        return ""
    if len(parts) == 1:
        single = _part_text(parts[0])
        if single is not None:
            return single

    content: list[dict[str, Any]] = []
    for part in parts:
        text = _part_text(part)
        if text is None or not text.strip():
            continue
        content.append({"type": "text", "text": text})
    return content or ""


def _user_content(parts: Sequence[Mapping[str, Any]]) -> str | list[dict[str, Any]]:
    """把用户消息内容转换为 OpenAI 结构（支持内联图片）。

    Args:
        parts: 内容片段序列。

    Returns:
        str | list[dict[str, Any]]: 单文本片段返回字符串，否则返回混合片段数组。
    """

    if len(parts) == 1 and str(parts[0].get("type") or "").strip().lower() == "text":
        text = parts[0].get("text")
        if isinstance(text, str):
            return text

    content: list[dict[str, Any]] = []
    for part in parts:
        text = _part_text(part)
        if text is not None:
            if text.strip():
                content.append({"type": "text", "text": text})
            continue

        image = _normalize_image_part(part)
        if image is None:
            content.append({"type": "text", "text": "[图片内容不可用]"})
            continue
        image_format, image_base64 = image
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_format};base64,{image_base64}"},
            }
        )
    return content


def _assistant_group_message(items: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """把相邻的模型输出 Item 折叠为一条 assistant 消息。

    Args:
        items: 同一组的 Item 序列。

    Returns:
        dict[str, Any] | None: assistant 消息；无正文也无工具调用时返回 ``None``。
    """

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in items:
        item_type = str(item.get("item_type") or "").strip()
        if item_type == "AssistantMessageItem":
            for part in item.get("parts") or []:
                if not isinstance(part, Mapping):
                    continue
                text = _part_text(part)
                if text:
                    text_parts.append(text)
        elif item_type == "FunctionCallItem":
            normalized = _normalize_tool_call(item.get("tool_call"))
            if normalized is not None:
                tool_calls.append(normalized)

    content = "".join(text_parts)
    if not content and not tool_calls:
        return None
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _normalize_tool_call(raw_tool_call: Any) -> dict[str, Any] | None:
    """把内部工具调用结构转为 OpenAI 结构。

    Args:
        raw_tool_call: ``tool_call`` 字典。

    Returns:
        dict[str, Any] | None: OpenAI 工具调用；字段缺失时返回 ``None``。
    """

    payload = _as_mapping(raw_tool_call)
    call_id = str(payload.get("call_id") or "").strip()
    func_name = str(payload.get("func_name") or "").strip()
    if not call_id or not func_name:
        return None

    arguments = payload.get("args")
    arguments_text = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, Mapping) else "{}"
    normalized: dict[str, Any] = {
        "id": call_id,
        "type": "function",
        "function": {"name": func_name, "arguments": arguments_text},
    }
    extra_content = payload.get("extra_content")
    if isinstance(extra_content, Mapping) and extra_content:
        normalized["extra_content"] = dict(extra_content)
    return normalized


def build_chat_messages(
    context_items: Iterable[Any],
    *,
    has_tools: bool,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """把 Context Item 快照转换为 OpenAI ``messages``。

    Args:
        context_items: 快照中的 ``context_items`` 列表。
        has_tools: 本次请求是否携带工具定义；为 ``False`` 时清洗工具链。
        stats: 可选统计字典；传入时写回 ``dropped_items`` / ``dropped_types``。

    Returns:
        list[dict[str, Any]]: OpenAI 兼容消息列表。
    """

    items = [item for item in context_items if isinstance(item, Mapping)]
    if not has_tools:
        items = [item for item in items if str(item.get("item_type") or "") not in _TOOL_CHAIN_TYPES]

    messages: list[dict[str, Any]] = []
    dropped = 0
    dropped_types: set[str] = set()
    index = 0
    while index < len(items):
        item = items[index]
        item_type = str(item.get("item_type") or "").strip()

        if item_type == "SystemMessageItem":
            parts = [part for part in (item.get("parts") or []) if isinstance(part, Mapping)]
            messages.append({"role": "system", "content": _text_only_content(parts)})
            index += 1
            continue

        if item_type == "UserMessageItem":
            parts = [part for part in (item.get("parts") or []) if isinstance(part, Mapping)]
            messages.append({"role": "user", "content": _user_content(parts)})
            index += 1
            continue

        if item_type == "FunctionCallOutputItem":
            messages.append(
                {
                    "role": "tool",
                    "content": str(item.get("output") or ""),
                    "tool_call_id": str(item.get("call_id") or ""),
                }
            )
            index += 1
            continue

        if item_type in _ASSISTANT_GROUP_TYPES:
            group: list[Mapping[str, Any]] = [item]
            index += 1
            while index < len(items):
                next_type = str(items[index].get("item_type") or "").strip()
                if next_type not in _ASSISTANT_GROUP_TYPES:
                    break
                group.append(items[index])
                index += 1
            assistant_message = _assistant_group_message(group)
            if assistant_message is not None:
                messages.append(assistant_message)
            continue

        # 未知识别不了的 Item：跳过而不是让整次请求失败，但要计数，
        # 避免宿主新增 Item 类型时上下文悄悄缺块。
        dropped += 1
        dropped_types.add(item_type or "<空类型>")
        index += 1

    _record_dropped(stats, dropped, dropped_types)
    return messages


def _record_dropped(
    stats: dict[str, Any] | None,
    dropped: int,
    dropped_types: set[str],
) -> None:
    """把「被丢弃的未识别 Item」统计写回调用方提供的字典。

    Args:
        stats: 调用方提供的统计字典；为 ``None`` 时不做任何事。
        dropped: 被丢弃的 Item 数量。
        dropped_types: 被丢弃的 Item 类型集合。
    """

    if stats is None:
        return
    stats["dropped_items"] = dropped
    stats["dropped_types"] = sorted(dropped_types)


def build_chat_tools(tool_options: Any) -> list[dict[str, Any]] | None:
    """把工具定义快照规范化为 OpenAI ``tools``。

    Args:
        tool_options: 快照中的 ``tool_options``（已是 OpenAI function 结构）。

    Returns:
        list[dict[str, Any]] | None: 工具列表；无工具时返回 ``None``。
    """

    if not isinstance(tool_options, Sequence) or isinstance(tool_options, (str, bytes)):
        return None

    tools: list[dict[str, Any]] = []
    for raw_tool in tool_options:
        tool = _as_mapping(raw_tool)
        function = _as_mapping(tool.get("function"))
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping) or not parameters:
            parameters = {"type": "object", "properties": {}, "required": []}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "parameters": dict(parameters),
                },
            }
        )
    return tools or None


def build_response_format(response_format: Any) -> dict[str, Any] | None:
    """把内部响应格式快照转换为 OpenAI ``response_format``。

    Args:
        response_format: 快照中的 ``response_format``。

    Returns:
        dict[str, Any] | None: OpenAI 响应格式；文本模式或缺失时返回 ``None``。
    """

    payload = _as_mapping(response_format)
    format_type = str(payload.get("format_type") or "").strip().lower()
    if format_type == "json_object":
        return {"type": "json_object"}
    if format_type == "json_schema":
        schema = payload.get("schema")
        if isinstance(schema, Mapping) and schema:
            return {"type": "json_schema", "json_schema": dict(schema)}
    return None


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """把模型返回的工具参数解析为字典。

    Args:
        raw_arguments: 可能是 JSON 字符串，也可能已经是字典。

    Returns:
        dict[str, Any]: 解析后的参数字典；无法解析时返回空字典。
    """

    if isinstance(raw_arguments, Mapping):
        return dict(raw_arguments)
    if not isinstance(raw_arguments, str):
        return {}

    text = _JSON_FENCE_PATTERN.sub("", raw_arguments.strip())
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def extract_response_content(message: Mapping[str, Any]) -> tuple[str, str]:
    """抽取助手正文与推理内容。

    Args:
        message: OpenAI 响应中的 ``choices[0].message``。

    Returns:
        tuple[str, str]: ``(正文, 推理内容)``。
    """

    content = message.get("content")
    if isinstance(content, list):
        collected: list[str] = []
        for part in content:
            part_payload = _as_mapping(part)
            text = _part_text(part_payload)
            if text:
                collected.append(text)
        content_text = "".join(collected)
    elif isinstance(content, str):
        content_text = content
    else:
        content_text = ""

    reasoning = ""
    for key in ("reasoning_content", "reasoning"):
        candidate = message.get(key)
        if isinstance(candidate, str) and candidate:
            reasoning = candidate
            break
    return content_text, reasoning


def build_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把响应中的工具调用转换为宿主可反序列化的结构。

    宿主 ``deserialize_tool_calls_snapshot`` 接受
    ``{"id": ..., "function": {"name": ..., "arguments": <dict>}}``。

    Args:
        message: OpenAI 响应中的 ``choices[0].message``。

    Returns:
        list[dict[str, Any]]: 规范化后的工具调用列表。
    """

    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, Sequence) or isinstance(raw_tool_calls, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_tool_call in raw_tool_calls:
        tool_call = _as_mapping(raw_tool_call)
        function = _as_mapping(tool_call.get("function"))
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        call_id = str(tool_call.get("id") or "").strip() or f"call_{len(normalized)}"
        entry: dict[str, Any] = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": parse_tool_arguments(function.get("arguments")),
            },
        }
        extra_content = tool_call.get("extra_content")
        if isinstance(extra_content, Mapping) and extra_content:
            entry["extra_content"] = dict(extra_content)
        normalized.append(entry)
    return normalized


def _as_int(value: Any) -> int:
    """把任意值安全地转为整数。

    Args:
        value: 待转换的值。

    Returns:
        int: 转换结果；失败时为 ``0``。
    """

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    """按顺序返回第一个「存在」的字段值。

    Args:
        payload: 来源映射。
        *keys: 候选键，按优先级排列。

    Returns:
        Any: 第一个非 ``None`` 的值；全部缺失时返回 ``None``。
    """

    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def build_usage(usage: Any, *, model_name: str, provider_name: str) -> dict[str, Any] | None:
    """把网关返回的 usage 规范化为宿主可识别的结构。

    同时兼容 Chat Completions（``prompt_tokens``/``completion_tokens``）
    与 Responses（``input_tokens``/``output_tokens``）两套字段。

    Args:
        usage: 响应中的 ``usage`` 字段。
        model_name: 本次模型名。
        provider_name: 本次 Provider 名。

    Returns:
        dict[str, Any] | None: 规范化用量；缺失时返回 ``None``。
    """

    payload = _as_mapping(usage)
    if not payload:
        return None

    prompt_tokens = _as_int(_first_present(payload, "prompt_tokens", "input_tokens"))
    completion_tokens = _as_int(_first_present(payload, "completion_tokens", "output_tokens"))
    total_tokens = _as_int(payload.get("total_tokens")) or prompt_tokens + completion_tokens

    details = _as_mapping(payload.get("prompt_tokens_details")) or _as_mapping(payload.get("input_tokens_details"))
    hit_tokens = _first_present(payload, "prompt_cache_hit_tokens", "cached_tokens")
    if hit_tokens is None:
        hit_tokens = details.get("cached_tokens")
    miss_tokens = payload.get("prompt_cache_miss_tokens")

    normalized: dict[str, Any] = {
        "model_name": str(model_name or ""),
        "provider_name": str(provider_name or ""),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if hit_tokens is not None or miss_tokens is not None:
        hit_value = _as_int(hit_tokens)
        miss_value = _as_int(miss_tokens)
        if not miss_value and hit_value:
            miss_value = max(prompt_tokens - hit_value, 0)
        normalized["prompt_cache_hit_tokens"] = hit_value
        normalized["prompt_cache_miss_tokens"] = miss_value
    return normalized


# ── Responses API（/responses）────────────────────────────────


@dataclass(slots=True)
class ResponsesOutput:
    """从 Responses 响应中投影出的可移植结果。"""

    content: str
    reasoning: str
    tool_calls: list[dict[str, Any]]
    response_id: str
    status: str
    incomplete_reason: str


def _plain_text(parts: Sequence[Mapping[str, Any]]) -> str:
    """把只允许文本的片段序列拼成字符串（忽略非文本片段）。

    Args:
        parts: 内容片段序列。

    Returns:
        str: 拼接后的文本。
    """

    collected: list[str] = []
    for part in parts:
        text = _part_text(part)
        if text:
            collected.append(text)
    return "".join(collected)


def _responses_user_content(parts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把用户消息内容转换为 Responses ``input`` 内容块。

    Args:
        parts: 内容片段序列。

    Returns:
        list[dict[str, Any]]: ``input_text`` / ``input_image`` 内容块。
    """

    content: list[dict[str, Any]] = []
    for part in parts:
        text = _part_text(part)
        if text is not None:
            if text.strip():
                content.append({"type": "input_text", "text": text})
            continue

        image = _normalize_image_part(part)
        if image is None:
            content.append({"type": "input_text", "text": "[图片内容不可用]"})
            continue
        image_format, image_base64 = image
        content.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:image/{image_format};base64,{image_base64}",
            }
        )
    return content


def build_responses_input(
    context_items: Iterable[Any],
    *,
    has_tools: bool,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """把 Context Item 快照转换为 Responses ``input`` 列表。

    与 Chat 路径不同，这里不折叠相邻的模型输出 Item：
    ``function_call`` 必须作为独立 Input Item 出现，``assistant`` 消息也单独一条。

    Args:
        context_items: 快照中的 ``context_items`` 列表。
        has_tools: 本次请求是否携带工具定义；为 ``False`` 时清洗工具链。
        stats: 可选统计字典；传入时写回 ``dropped_items`` / ``dropped_types``。

    Returns:
        list[dict[str, Any]]: Responses Input Items。
    """

    items = [item for item in context_items if isinstance(item, Mapping)]
    if not has_tools:
        items = [item for item in items if str(item.get("item_type") or "") not in _TOOL_CHAIN_TYPES]

    input_items: list[dict[str, Any]] = []
    dropped = 0
    dropped_types: set[str] = set()
    for item in items:
        item_type = str(item.get("item_type") or "").strip()
        parts = [part for part in (item.get("parts") or []) if isinstance(part, Mapping)]

        if item_type == "SystemMessageItem":
            text = _plain_text(parts)
            if text:
                input_items.append({"role": "system", "content": text})
            continue

        if item_type == "UserMessageItem":
            input_items.append({"role": "user", "content": _responses_user_content(parts)})
            continue

        if item_type == "AssistantMessageItem":
            text = _plain_text(parts)
            if text:
                input_items.append({"role": "assistant", "content": text})
            continue

        if item_type == "FunctionCallItem":
            tool_call = _as_mapping(item.get("tool_call"))
            call_id = str(tool_call.get("call_id") or "").strip()
            func_name = str(tool_call.get("func_name") or "").strip()
            if not call_id or not func_name:
                continue
            arguments = tool_call.get("args")
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, Mapping) else "{}",
                }
            )
            continue

        if item_type == "FunctionCallOutputItem":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(item.get("call_id") or ""),
                    "output": str(item.get("output") or ""),
                }
            )
            continue

        # reasoning / Provider 原生活动 / 未知 Item 在 Responses 侧没有可移植投影。
        dropped += 1
        dropped_types.add(item_type or "<空类型>")

    _record_dropped(stats, dropped, dropped_types)
    return input_items


def build_responses_tools(tool_options: Any) -> list[dict[str, Any]] | None:
    """把工具定义快照转换为 Responses 原生工具定义（扁平结构）。

    Args:
        tool_options: 快照中的 ``tool_options``（OpenAI function 结构）。

    Returns:
        list[dict[str, Any]] | None: 工具列表；无工具时返回 ``None``。
    """

    if not isinstance(tool_options, Sequence) or isinstance(tool_options, (str, bytes)):
        return None

    tools: list[dict[str, Any]] = []
    for raw_tool in tool_options:
        function = _as_mapping(_as_mapping(raw_tool).get("function"))
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping) or not parameters:
            parameters = {"type": "object", "properties": {}, "required": []}
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": str(function.get("description") or ""),
                "parameters": dict(parameters),
            }
        )
    return tools or None


def build_responses_text_config(response_format: Any) -> dict[str, Any] | None:
    """把内部响应格式快照转换为 Responses ``text`` 配置。

    Args:
        response_format: 快照中的 ``response_format``。

    Returns:
        dict[str, Any] | None: 形如 ``{"format": {...}}`` 的配置；无需设置时返回 ``None``。
    """

    payload = _as_mapping(response_format)
    format_type = str(payload.get("format_type") or "").strip().lower()
    format_config: dict[str, Any] | None = None
    if format_type == "json_object":
        format_config = {"type": "json_object"}
    elif format_type == "json_schema":
        schema = payload.get("schema")
        if isinstance(schema, Mapping) and schema:
            format_config = {"type": "json_schema", **dict(schema)}
    return {"format": format_config} if format_config else None


def _responses_error_text(payload: Mapping[str, Any]) -> str:
    """从失败响应中提取可读原因。

    Args:
        payload: Responses 响应体。

    Returns:
        str: 可读的原因文本。
    """

    for key in ("error", "incomplete_details"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            message = value.get("message") or value.get("reason")
            if message:
                return str(message)
            if value:
                return json.dumps(dict(value), ensure_ascii=False)
        elif value:
            return str(value)
    return str(payload.get("status") or "unknown")


def parse_responses_output(payload: Mapping[str, Any]) -> ResponsesOutput:
    """把 Responses 响应体投影为可移植结果。

    Args:
        payload: 网关返回的 Responses JSON。

    Returns:
        ResponsesOutput: 正文、推理、工具调用与状态。
    """

    raw_output = payload.get("output")
    items = (
        [item for item in raw_output if isinstance(item, Mapping)]
        if isinstance(raw_output, Sequence) and not isinstance(raw_output, (str, bytes))
        else []
    )

    content_parts: list[str] = []
    summary_parts: list[str] = []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in items:
        item_type = str(item.get("type") or "").strip()

        if item_type == "message":
            for raw_part in item.get("content") or []:
                part = _as_mapping(raw_part)
                part_type = str(part.get("type") or "").strip()
                if part_type == "output_text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        content_parts.append(text)
                elif part_type == "refusal":
                    refusal = part.get("refusal")
                    if isinstance(refusal, str) and refusal:
                        content_parts.append(refusal)
            continue

        if item_type == "reasoning":
            for raw_summary in item.get("summary") or []:
                text = _as_mapping(raw_summary).get("text")
                if isinstance(text, str) and text.strip():
                    summary_parts.append(text.strip())
            for raw_part in item.get("content") or []:
                part = _as_mapping(raw_part)
                if str(part.get("type") or "").strip() != "reasoning_text":
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
            continue

        if item_type == "function_call":
            call_id = str(item.get("call_id") or "").strip()
            func_name = str(item.get("name") or "").strip()
            if not call_id or not func_name:
                continue
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": parse_tool_arguments(item.get("arguments")),
                    },
                }
            )
            continue

    status = str(payload.get("status") or "")
    return ResponsesOutput(
        content="".join(content_parts),
        reasoning="\n".join(summary_parts) if summary_parts else "\n".join(text_parts),
        tool_calls=tool_calls,
        response_id=str(payload.get("id") or ""),
        status=status,
        incomplete_reason=_responses_error_text(payload) if status in {"failed", "incomplete", "cancelled"} else "",
    )
