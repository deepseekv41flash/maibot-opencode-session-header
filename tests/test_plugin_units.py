"""OpenCode 会话头注入插件的离线单元测试。

只覆盖纯逻辑（端点识别、种子提取、哈希、线格式转换），
不依赖 MaiBot 宿主，可直接用标准库运行：

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai_wire import (  # noqa: E402
    build_chat_messages,
    build_chat_tools,
    build_responses_input,
    build_responses_text_config,
    build_responses_tools,
    build_response_format,
    build_tool_calls,
    build_usage,
    parse_responses_output,
    parse_tool_arguments,
)
from session_identity import (  # noqa: E402
    derive_session_token,
    extract_host,
    extract_session_seed,
    is_opencode_endpoint,
    normalize_headers,
    strip_session_prefix,
)


class EndpointMatchTests(unittest.TestCase):
    """端点识别。"""

    def test_matches_opencode_domains(self) -> None:
        for url in (
            "https://opencode.ai/zen/v1",
            "https://opencode.ai",
            "https://zen.opencode.ai/v1",
            "https://api.opencode.ai/zen/go/v1",
            "opencode.ai/zen/v1",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_opencode_endpoint(url))

    def test_ignores_other_providers(self) -> None:
        for url in (
            "https://api.deepseek.com/v1",
            "https://api.openai.com/v1",
            "https://opencode.ai.evil.example/v1",
            "https://notopencode.ai/v1",
            "",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_opencode_endpoint(url))

    def test_extra_prefix_support(self) -> None:
        self.assertTrue(
            is_opencode_endpoint(
                "https://gw.example.com/opencode/v1",
                extra_base_url_prefixes=["https://gw.example.com/opencode"],
            )
        )
        self.assertFalse(
            is_opencode_endpoint(
                "https://gw.example.com/other/v1",
                extra_base_url_prefixes=["https://gw.example.com/opencode"],
            )
        )

    def test_extract_host(self) -> None:
        self.assertEqual(extract_host("https://opencode.ai/zen/v1"), "opencode.ai")
        self.assertEqual(extract_host("opencode.ai/zen"), "opencode.ai")
        self.assertEqual(extract_host(""), "")


class SessionSeedTests(unittest.TestCase):
    """会话种子提取。"""

    def test_priority_follows_candidate_order(self) -> None:
        seed = extract_session_seed([{"Session-Id": "abc", "X-Opencode-Session": "xyz"}])
        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertEqual(seed.value, "xyz")
        self.assertIn("x-opencode-session", seed.source)

    def test_source_order_breaks_ties(self) -> None:
        seed = extract_session_seed([{"session-id": "first"}, {"session-id": "second"}])
        assert seed is not None
        self.assertEqual(seed.value, "first")

    def test_prefix_is_stripped(self) -> None:
        seed = extract_session_seed([{"session-id": "session-6f1c9a"}])
        assert seed is not None
        self.assertEqual(seed.value, "6f1c9a")
        self.assertEqual(strip_session_prefix("sess_42"), "42")
        self.assertEqual(strip_session_prefix("plain"), "plain")

    def test_no_candidates(self) -> None:
        self.assertIsNone(extract_session_seed([{"authorization": "Bearer x"}]))

    def test_header_normalization(self) -> None:
        self.assertEqual(normalize_headers({" Session-Id ": " 7 "}), {"session-id": "7"})
        self.assertEqual(normalize_headers(None), {})
        self.assertEqual(normalize_headers({"X-Empty": "   "}), {})


class TokenTests(unittest.TestCase):
    """令牌派生。"""

    def test_stable_and_truncated(self) -> None:
        first = derive_session_token("abc", length=8)
        second = derive_session_token("abc", length=8)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertTrue(first.isalnum())

    def test_prefix_stripping_keeps_token_stable(self) -> None:
        with_prefix = extract_session_seed([{"session-id": "session-abc"}])
        plain = extract_session_seed([{"session-id": "abc"}])
        assert with_prefix is not None and plain is not None
        self.assertEqual(
            derive_session_token(with_prefix.value),
            derive_session_token(plain.value),
        )

    def test_length_and_alphabet(self) -> None:
        token = derive_session_token("abc", length=16, alphabet="urlsafe")
        self.assertEqual(len(token), 16)
        self.assertTrue(all(char.isalnum() or char in "-_" for char in token))

    def test_length_is_clamped(self) -> None:
        self.assertEqual(len(derive_session_token("abc", length=999)), 32)
        self.assertEqual(len(derive_session_token("abc", length=0)), 4)

    def test_salt_changes_token(self) -> None:
        self.assertNotEqual(
            derive_session_token("abc", salt="a"),
            derive_session_token("abc", salt="b"),
        )

    def test_seed_never_leaks(self) -> None:
        seed = "session-0123456789abcdef"
        token = derive_session_token(seed)
        self.assertNotIn("0123456789abcdef", token)

    def test_empty_seed_rejected(self) -> None:
        with self.assertRaises(ValueError):
            derive_session_token("   ")


class WireTests(unittest.TestCase):
    """Context Item ↔ OpenAI 线格式。"""

    @staticmethod
    def _item(item_type: str, **extra: object) -> dict:
        return {"item_type": item_type, "meta": {"item_id": "i", "timestamp": "t"}, **extra}

    def test_basic_conversation(self) -> None:
        items = [
            self._item("SystemMessageItem", parts=[{"type": "text", "text": "你是猫娘"}]),
            self._item("UserMessageItem", parts=[{"type": "text", "text": "在吗"}]),
            self._item("AssistantMessageItem", parts=[{"type": "text", "text": "在的"}]),
        ]
        messages = build_chat_messages(items, has_tools=False)
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "你是猫娘"},
                {"role": "user", "content": "在吗"},
                {"role": "assistant", "content": "在的"},
            ],
        )

    def test_image_part_becomes_data_url(self) -> None:
        items = [
            self._item(
                "UserMessageItem",
                parts=[
                    {"type": "text", "text": "看这个"},
                    {"type": "image", "image_format": "png", "image_base64": "QUJD"},
                ],
            )
        ]
        messages = build_chat_messages(items, has_tools=False)
        content = messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "看这个"})
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,QUJD")

    def test_tool_chain_folds_into_assistant(self) -> None:
        items = [
            self._item("UserMessageItem", parts=[{"type": "text", "text": "查天气"}]),
            self._item("ReasoningItem", representation="summary", summary_parts=["想一下"], text_parts=[]),
            self._item(
                "AssistantMessageItem",
                parts=[{"type": "text", "text": "我查一下"}],
            ),
            self._item(
                "FunctionCallItem",
                tool_call={"call_id": "call_1", "func_name": "get_weather", "args": {"city": "北京"}},
            ),
            self._item("FunctionCallOutputItem", call_id="call_1", output="晴", tool_name="get_weather"),
        ]
        messages = build_chat_messages(items, has_tools=True)
        self.assertEqual(len(messages), 3)
        assistant = messages[1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"], "我查一下")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_1")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(messages[2], {"role": "tool", "content": "晴", "tool_call_id": "call_1"})

    def test_tool_chain_dropped_without_tools(self) -> None:
        items = [
            self._item("UserMessageItem", parts=[{"type": "text", "text": "hi"}]),
            self._item(
                "FunctionCallItem",
                tool_call={"call_id": "call_1", "func_name": "f", "args": {}},
            ),
            self._item("FunctionCallOutputItem", call_id="call_1", output="ok"),
        ]
        messages = build_chat_messages(items, has_tools=False)
        self.assertEqual(messages, [{"role": "user", "content": "hi"}])

    def test_tools_and_response_format(self) -> None:
        tools = build_chat_tools(
            [
                {
                    "type": "function",
                    "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
                },
                {"type": "function", "function": {"description": "无名"}},
            ]
        )
        self.assertEqual(len(tools or []), 1)
        self.assertEqual(tools[0]["function"]["name"], "f")
        self.assertIsNone(build_chat_tools([]))
        self.assertEqual(build_response_format({"format_type": "json_object"}), {"type": "json_object"})
        self.assertIsNone(build_response_format({"format_type": "text"}))
        self.assertIsNone(build_response_format(None))

    def test_parse_tool_arguments(self) -> None:
        self.assertEqual(parse_tool_arguments('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_tool_arguments('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_tool_arguments("not json"), {})
        self.assertEqual(parse_tool_arguments({"a": 2}), {"a": 2})

    def test_build_tool_calls_from_response(self) -> None:
        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "reply", "arguments": '{"text": "在"}'},
                }
            ],
        }
        calls = build_tool_calls(message)
        self.assertEqual(calls[0]["function"]["arguments"], {"text": "在"})
        self.assertEqual(build_tool_calls({"content": "纯文本"}), [])


class ResponsesWireTests(unittest.TestCase):
    """Responses API 线格式。"""

    @staticmethod
    def _item(item_type: str, **extra: object) -> dict:
        return {"item_type": item_type, "meta": {"item_id": "i", "timestamp": "t"}, **extra}

    def test_input_keeps_function_call_separate(self) -> None:
        items = [
            self._item("SystemMessageItem", parts=[{"type": "text", "text": "你是猫娘"}]),
            self._item("UserMessageItem", parts=[{"type": "text", "text": "查天气"}]),
            self._item("AssistantMessageItem", parts=[{"type": "text", "text": "我查一下"}]),
            self._item(
                "FunctionCallItem",
                tool_call={"call_id": "call_1", "func_name": "get_weather", "args": {"city": "北京"}},
            ),
            self._item("FunctionCallOutputItem", call_id="call_1", output="晴"),
        ]
        input_items = build_responses_input(items, has_tools=True)
        self.assertEqual(input_items[0], {"role": "system", "content": "你是猫娘"})
        self.assertEqual(input_items[1], {"role": "user", "content": [{"type": "input_text", "text": "查天气"}]})
        self.assertEqual(input_items[2], {"role": "assistant", "content": "我查一下"})
        self.assertEqual(
            input_items[3],
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "北京"}',
            },
        )
        self.assertEqual(
            input_items[4],
            {"type": "function_call_output", "call_id": "call_1", "output": "晴"},
        )

    def test_input_drops_tool_chain_without_tools(self) -> None:
        items = [
            self._item("UserMessageItem", parts=[{"type": "text", "text": "hi"}]),
            self._item("FunctionCallItem", tool_call={"call_id": "c", "func_name": "f", "args": {}}),
            self._item("FunctionCallOutputItem", call_id="c", output="ok"),
        ]
        input_items = build_responses_input(items, has_tools=False)
        self.assertEqual(len(input_items), 1)
        self.assertEqual(input_items[0]["role"], "user")

    def test_input_image_block(self) -> None:
        items = [
            self._item(
                "UserMessageItem",
                parts=[
                    {"type": "text", "text": "看这个"},
                    {"type": "image", "image_format": "jpg", "image_base64": "QUJD"},
                ],
            )
        ]
        input_items = build_responses_input(items, has_tools=False)
        content = input_items[0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "看这个"})
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["image_url"], "data:image/jpeg;base64,QUJD")

    def test_reasoning_items_are_skipped(self) -> None:
        items = [
            self._item("ReasoningItem", representation="summary", summary_parts=["想"], text_parts=[]),
            self._item("UserMessageItem", parts=[{"type": "text", "text": "hi"}]),
        ]
        input_items = build_responses_input(items, has_tools=False)
        self.assertEqual(len(input_items), 1)

    def test_responses_tools_are_flat(self) -> None:
        tools = build_responses_tools(
            [
                {
                    "type": "function",
                    "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
                }
            ]
        )
        self.assertEqual(
            tools,
            [{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}],
        )
        self.assertIsNone(build_responses_tools([]))

    def test_text_config(self) -> None:
        self.assertEqual(
            build_responses_text_config({"format_type": "json_object"}),
            {"format": {"type": "json_object"}},
        )
        self.assertIsNone(build_responses_text_config({"format_type": "text"}))
        self.assertIsNone(build_responses_text_config(None))

    def test_parse_completed_output(self) -> None:
        payload = {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "想一下"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "喵~"}]},
                {
                    "type": "function_call",
                    "call_id": "call_7",
                    "name": "reply",
                    "arguments": '{"text": "在"}',
                },
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 4,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 8},
            },
        }
        parsed = parse_responses_output(payload)
        self.assertEqual(parsed.content, "喵~")
        self.assertEqual(parsed.reasoning, "想一下")
        self.assertEqual(parsed.response_id, "resp_1")
        self.assertEqual(parsed.incomplete_reason, "")
        self.assertEqual(parsed.tool_calls[0]["id"], "call_7")
        self.assertEqual(parsed.tool_calls[0]["function"]["arguments"], {"text": "在"})

    def test_parse_failed_output_reports_reason(self) -> None:
        parsed = parse_responses_output(
            {"id": "resp_2", "status": "failed", "error": {"message": "boom"}, "output": []}
        )
        self.assertEqual(parsed.status, "failed")
        self.assertEqual(parsed.incomplete_reason, "boom")

    def test_parse_ignores_unknown_items(self) -> None:
        parsed = parse_responses_output(
            {"status": "completed", "output": [{"type": "web_search_call", "id": "ws_1"}]}
        )
        self.assertEqual(parsed.content, "")
        self.assertEqual(parsed.tool_calls, [])

    def test_usage_from_responses_keys(self) -> None:
        usage = build_usage(
            {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 64},
            },
            model_name="m",
            provider_name="p",
        )
        assert usage is not None
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 20)
        self.assertEqual(usage["total_tokens"], 120)
        self.assertEqual(usage["prompt_cache_hit_tokens"], 64)
        self.assertEqual(usage["prompt_cache_miss_tokens"], 36)

    def test_usage_from_chat_keys_still_works(self) -> None:
        usage = build_usage(
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            model_name="m",
            provider_name="p",
        )
        assert usage is not None
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertNotIn("prompt_cache_hit_tokens", usage)


if __name__ == "__main__":
    unittest.main()
