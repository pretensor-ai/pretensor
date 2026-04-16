"""Tests for LLM runtime utilities (fences, interface types)."""

from __future__ import annotations

from pretensor.intelligence.llm_runtime import (
    ChatMessage,
    LlmBudgetExceededError,
    LlmUsage,
    parse_json_array,
    strip_json_fence,
    strip_markdown_fence,
)


def test_strip_markdown_fence_yaml() -> None:
    raw = "```yaml\n- a: 1\n```"
    assert strip_markdown_fence(raw) == "- a: 1"


def test_strip_markdown_fence_plain() -> None:
    assert strip_markdown_fence("plain") == "plain"


def test_strip_json_fence_delegates() -> None:
    raw = '```json\n["a"]\n```'
    assert strip_json_fence(raw) == '["a"]'


def test_parse_json_array_basic() -> None:
    assert parse_json_array('[1, 2, 3]') == [1, 2, 3]


def test_parse_json_array_with_fence() -> None:
    assert parse_json_array('```json\n["x"]\n```') == ["x"]


def test_chat_message_frozen() -> None:
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_llm_usage_frozen() -> None:
    u = LlmUsage(input_tokens=10, output_tokens=20)
    assert u.input_tokens == 10
    assert u.output_tokens == 20


def test_budget_exceeded_error_is_exception() -> None:
    assert issubclass(LlmBudgetExceededError, Exception)
