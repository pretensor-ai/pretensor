"""Unit tests for L3 LLM client adapters.

We never hit a real provider in unit tests — all transport is faked via
``httpx.MockTransport``. The goal is to exercise envelope construction,
header / endpoint correctness, error wrapping, and response parsing.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from pretensor.benchmark.l3.agent import (
    AgentMessage,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
)
from pretensor.benchmark.l3.llm_client import (
    AnthropicHttpClient,
    LlmCallError,
    OpenAIHttpClient,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _anthropic_ok_response(text: str = "SELECT 1") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_x",
            "type": "message",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        },
    )


def _openai_ok_response(text: str = "SELECT 1") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_x",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 14, "completion_tokens": 5},
        },
    )


def _make_client(
    handler: Any,
    cls: Any,
    api_key: str = "test-key",
) -> Any:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return cls(api_key=api_key, client=http)


# ---------------------------------------------------------------------------
# AnthropicHttpClient
# ---------------------------------------------------------------------------


def test_anthropic_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LlmCallError, match="ANTHROPIC_API_KEY"):
        AnthropicHttpClient()


def test_anthropic_uses_explicit_api_key_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["key"] = request.headers.get("x-api-key")
        return _anthropic_ok_response()

    client = _make_client(handler, AnthropicHttpClient, api_key="explicit-key")
    client.complete(system="S", user="U", model="claude-haiku-4-5", temperature=0.0)
    assert captured["key"] == "explicit-key"


def test_anthropic_posts_to_messages_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["version"] = request.headers.get("anthropic-version")
        return _anthropic_ok_response()

    client = _make_client(handler, AnthropicHttpClient)
    client.complete(system="S", user="U", model="claude-haiku-4-5", temperature=0.0)
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["version"] == "2023-06-01"


def test_anthropic_request_body_carries_system_user_model_temperature() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return _anthropic_ok_response()

    client = _make_client(handler, AnthropicHttpClient)
    client.complete(
        system="SYS", user="USER", model="claude-haiku-4-5", temperature=0.0
    )
    body = captured["body"]
    assert body["system"] == "SYS"
    assert body["model"] == "claude-haiku-4-5"
    assert body["temperature"] == 0.0
    assert body["messages"] == [{"role": "user", "content": "USER"}]


def test_anthropic_parses_text_blocks() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _anthropic_ok_response("SELECT title FROM film")

    client = _make_client(handler, AnthropicHttpClient)
    response = client.complete(
        system="S", user="U", model="claude-haiku-4-5", temperature=0.0
    )
    assert response.text == "SELECT title FROM film"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 7
    assert response.latency_ms >= 0


def test_anthropic_concatenates_multiple_text_blocks() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "SELECT "},
                    {"type": "text", "text": "1"},
                ],
            },
        )

    client = _make_client(handler, AnthropicHttpClient)
    response = client.complete(
        system="S", user="U", model="claude-haiku-4-5", temperature=0.0
    )
    assert response.text == "SELECT 1"


def test_anthropic_wraps_http_error_in_llm_call_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    client = _make_client(handler, AnthropicHttpClient)
    with pytest.raises(LlmCallError, match="429"):
        client.complete(system="S", user="U", model="m", temperature=0.0)


def test_anthropic_wraps_transport_error_in_llm_call_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = _make_client(handler, AnthropicHttpClient)
    with pytest.raises(LlmCallError, match="transport"):
        client.complete(system="S", user="U", model="m", temperature=0.0)


def test_anthropic_raises_when_no_text_blocks_in_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": []})

    client = _make_client(handler, AnthropicHttpClient)
    with pytest.raises(LlmCallError, match="no text blocks"):
        client.complete(system="S", user="U", model="m", temperature=0.0)


def test_anthropic_raises_on_invalid_json_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = _make_client(handler, AnthropicHttpClient)
    with pytest.raises(LlmCallError, match="not JSON"):
        client.complete(system="S", user="U", model="m", temperature=0.0)


# ---------------------------------------------------------------------------
# OpenAIHttpClient
# ---------------------------------------------------------------------------


def test_openai_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LlmCallError, match="OPENAI_API_KEY"):
        OpenAIHttpClient()


def test_openai_posts_to_chat_completions_endpoint() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        return _openai_ok_response()

    client = _make_client(handler, OpenAIHttpClient)
    client.complete(system="S", user="U", model="gpt-4o-mini", temperature=0.0)
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"


def test_openai_request_body_packs_messages() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return _openai_ok_response()

    client = _make_client(handler, OpenAIHttpClient)
    client.complete(system="SYS", user="USER", model="gpt-4o-mini", temperature=0.0)
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert captured["body"]["temperature"] == 0.0
    assert captured["body"]["model"] == "gpt-4o-mini"


def test_openai_parses_choice_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _openai_ok_response("SELECT title FROM film")

    client = _make_client(handler, OpenAIHttpClient)
    response = client.complete(
        system="S", user="U", model="gpt-4o-mini", temperature=0.0
    )
    assert response.text == "SELECT title FROM film"
    assert response.prompt_tokens == 14
    assert response.completion_tokens == 5


def test_openai_wraps_http_error_in_llm_call_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    client = _make_client(handler, OpenAIHttpClient)
    with pytest.raises(LlmCallError, match="500"):
        client.complete(system="S", user="U", model="m", temperature=0.0)


def test_openai_wraps_transport_error_in_llm_call_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    client = _make_client(handler, OpenAIHttpClient)
    with pytest.raises(LlmCallError, match="transport"):
        client.complete(system="S", user="U", model="m", temperature=0.0)


def test_openai_raises_when_choices_missing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = _make_client(handler, OpenAIHttpClient)
    with pytest.raises(LlmCallError, match="choices"):
        client.complete(system="S", user="U", model="m", temperature=0.0)


def test_openai_raises_when_content_not_string() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": 42}}]},
        )

    client = _make_client(handler, OpenAIHttpClient)
    with pytest.raises(LlmCallError, match="content"):
        client.complete(system="S", user="U", model="m", temperature=0.0)


def test_openai_raises_on_invalid_json_response() -> None:
    """Non-JSON 200 responses are wrapped as LlmCallError, not raw ValueError.

    Mirrors the Anthropic equivalent so both clients fail the same way
    when a provider returns success status with an unparseable body.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = _make_client(handler, OpenAIHttpClient)
    with pytest.raises(LlmCallError, match="not JSON"):
        client.complete(system="S", user="U", model="m", temperature=0.0)


# ---------------------------------------------------------------------------
# AnthropicHttpClient.agent_complete (tool-use)
# ---------------------------------------------------------------------------


def _anthropic_text_step(text: str = "SELECT 1") -> httpx.Response:
    """Anthropic Messages response with a single text block (final step)."""
    return httpx.Response(
        200,
        json={
            "id": "msg_x",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        },
    )


def _anthropic_tool_use_step() -> httpx.Response:
    """Anthropic Messages response with a single tool_use block."""
    return httpx.Response(
        200,
        json={
            "id": "msg_x",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Looking up customers."},
                {
                    "type": "tool_use",
                    "id": "toolu_01ABC",
                    "name": "query",
                    "input": {"q": "customer"},
                },
            ],
            "usage": {"input_tokens": 25, "output_tokens": 9},
        },
    )


def test_agent_complete_text_step_returns_final_text() -> None:
    """Stop reason ``end_turn`` produces an :class:`AgentStep` with text."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return _anthropic_text_step("SELECT 1")

    client = _make_client(handler, AnthropicHttpClient)
    step = client.agent_complete(
        system="SYS",
        messages=[AgentMessage(role="user", text="ask")],
        tools=[
            AgentTool(
                name="query", description="search", input_schema={"type": "object"}
            ),
        ],
        model="claude-haiku-4-5",
        temperature=0.0,
    )
    assert step.text == "SELECT 1"
    assert step.tool_calls == ()
    assert step.prompt_tokens == 12
    assert step.completion_tokens == 7
    # The wire body carries tools[] + system + messages[] in Anthropic shape.
    body = captured["body"]
    assert body["system"] == "SYS"
    assert body["tools"] == [
        {"name": "query", "description": "search", "input_schema": {"type": "object"}}
    ]
    assert body["messages"] == [{"role": "user", "content": "ask"}]


def test_agent_complete_tool_use_step_returns_tool_calls() -> None:
    """Stop reason ``tool_use`` decodes ``tool_use`` blocks into tool_calls."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return _anthropic_tool_use_step()

    client = _make_client(handler, AnthropicHttpClient)
    step = client.agent_complete(
        system="SYS",
        messages=[AgentMessage(role="user", text="ask")],
        tools=[AgentTool(name="query", description="", input_schema={})],
        model="m",
        temperature=0.0,
    )
    # Text BLOCK present alongside the tool_use; agent loop carries it forward.
    assert step.text == "Looking up customers."
    assert len(step.tool_calls) == 1
    call = step.tool_calls[0]
    assert call.id == "toolu_01ABC"
    assert call.name == "query"
    assert call.arguments == {"q": "customer"}


def test_agent_complete_serialises_assistant_and_tool_result_history() -> None:
    """Multi-turn history is rendered to Anthropic's content-block format.

    Verifies the wire payload after one tool round-trip:
    - assistant turn with text + tool_use block
    - user turn with tool_result block referencing the same id
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return _anthropic_text_step("SELECT 2")

    client = _make_client(handler, AnthropicHttpClient)
    history = [
        AgentMessage(role="user", text="ask"),
        AgentMessage(
            role="assistant",
            text="Calling tool.",
            tool_calls=(
                AgentToolCall(id="toolu_1", name="query", arguments={"q": "x"}),
            ),
        ),
        AgentMessage(
            role="user",
            tool_results=(
                AgentToolResult(id="toolu_1", content='{"hits":[]}', is_error=False),
            ),
        ),
    ]
    client.agent_complete(
        system="SYS",
        messages=history,
        tools=[],
        model="m",
        temperature=0.0,
    )
    msgs = captured["body"]["messages"]
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "ask"}
    assert msgs[1]["role"] == "assistant"
    blocks = msgs[1]["content"]
    assert blocks[0] == {"type": "text", "text": "Calling tool."}
    assert blocks[1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "query",
        "input": {"q": "x"},
    }
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": '{"hits":[]}',
        }
    ]


def test_agent_complete_raises_when_tool_use_input_is_not_a_dict() -> None:
    """A malformed Anthropic ``tool_use.input`` (non-object) fails loud.

    Silently coercing non-dict input to ``{}`` would forward an
    empty-args tool call and mask a provider-side regression.
    Verifies the runner sees ``LlmCallError`` instead, surfaced into
    the per-item record where an auditor can spot it.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_bad",
                        "name": "query",
                        "input": "this should have been an object",
                    }
                ],
            },
        )

    client = _make_client(handler, AnthropicHttpClient)
    with pytest.raises(LlmCallError, match="not a JSON object"):
        client.agent_complete(
            system="SYS",
            messages=[AgentMessage(role="user", text="?")],
            tools=[AgentTool(name="query", description="", input_schema={})],
            model="m",
            temperature=0.0,
        )


def test_agent_complete_marks_tool_result_is_error_in_payload() -> None:
    """``AgentToolResult.is_error=True`` is forwarded so the LLM sees the failure."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return _anthropic_text_step("SELECT 1")

    client = _make_client(handler, AnthropicHttpClient)
    client.agent_complete(
        system="SYS",
        messages=[
            AgentMessage(role="user", text="ask"),
            AgentMessage(
                role="assistant",
                tool_calls=(AgentToolCall(id="t1", name="q", arguments={}),),
            ),
            AgentMessage(
                role="user",
                tool_results=(AgentToolResult(id="t1", content="boom", is_error=True),),
            ),
        ],
        tools=[],
        model="m",
        temperature=0.0,
    )
    last_msg = captured["body"]["messages"][-1]
    assert last_msg["content"][0]["is_error"] is True
