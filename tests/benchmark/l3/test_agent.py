"""Unit tests for the provider-agnostic agent loop.

The loop is a tight finite-state machine: emit message → call LLM →
either finish (text) or run tools, then loop. Tests exercise the
finishing and tool-routing branches, the iteration cap, and the
trace shape that the runner serialises into the JSON envelope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from pretensor.benchmark.l3.agent import (
    DEFAULT_MAX_ITERATIONS,
    AgentLoopError,
    AgentMessage,
    AgentStep,
    AgentTool,
    AgentToolCall,
    ToolInvocationOutcome,
    run_agent_loop,
)


@dataclass
class _ScriptedAgentClient:
    """Returns a pre-recorded :class:`AgentStep` per ``agent_complete`` call.

    Steps are popped from the front of the list so test setup reads
    top-down in execution order; an empty list at call time raises so a
    test can't quietly succeed by under-asserting iterations.
    """

    steps: list[AgentStep]
    captured_messages: list[list[AgentMessage]] = field(default_factory=list)
    captured_tools: list[list[AgentTool]] = field(default_factory=list)

    def agent_complete(
        self,
        *,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        model: str,
        temperature: float,
    ) -> AgentStep:
        # Snapshot the state the loop sent so tests can assert history shape.
        self.captured_messages.append(list(messages))
        self.captured_tools.append(list(tools))
        if not self.steps:
            raise AssertionError("scripted client ran out of steps")
        return self.steps.pop(0)


def _ok_outcome(text: str) -> ToolInvocationOutcome:
    return ToolInvocationOutcome(content=text, is_error=False)


def _make_invoker(
    *,
    by_name: dict[str, Callable[[dict[str, Any]], ToolInvocationOutcome]],
) -> Callable[[str, dict[str, Any]], ToolInvocationOutcome]:
    def _invoke(name: str, args: dict[str, Any]) -> ToolInvocationOutcome:
        try:
            return by_name[name](args)
        except KeyError:
            return ToolInvocationOutcome(content="unknown tool", is_error=True)

    return _invoke


def _tool(name: str) -> AgentTool:
    return AgentTool(name=name, description="", input_schema={})


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


def test_returns_immediately_when_first_step_has_text_and_no_tools() -> None:
    client = _ScriptedAgentClient(
        steps=[
            AgentStep(
                text="SELECT 1",
                tool_calls=(),
                prompt_tokens=10,
                completion_tokens=2,
                latency_ms=50,
            )
        ]
    )
    result = run_agent_loop(
        client=client,
        system="sys",
        user="ask",
        tools=[_tool("query")],
        invoke_tool=_make_invoker(by_name={}),
        model="m",
        temperature=0.0,
    )
    assert result.text == "SELECT 1"
    assert result.iterations == 1
    assert result.trace == []
    assert result.total_prompt_tokens == 10
    assert result.total_completion_tokens == 2
    assert result.total_llm_latency_ms == 50


def test_runs_tool_calls_and_appends_results_to_history() -> None:
    """Loop drives a tool call, then the LLM produces final SQL."""
    client = _ScriptedAgentClient(
        steps=[
            AgentStep(
                text=None,
                tool_calls=(
                    AgentToolCall(
                        id="call_1", name="query", arguments={"q": "customer"}
                    ),
                ),
                prompt_tokens=20,
                completion_tokens=5,
                latency_ms=80,
            ),
            AgentStep(
                text="SELECT * FROM customer",
                tool_calls=(),
                prompt_tokens=30,
                completion_tokens=8,
                latency_ms=70,
            ),
        ]
    )
    payload = json.dumps({"hits": [{"table": "customer"}]})
    invoker = _make_invoker(by_name={"query": lambda _args: _ok_outcome(payload)})

    result = run_agent_loop(
        client=client,
        system="sys",
        user="how many customers?",
        tools=[_tool("query")],
        invoke_tool=invoker,
        model="m",
        temperature=0.0,
    )

    assert result.text == "SELECT * FROM customer"
    assert result.iterations == 2
    assert len(result.trace) == 1
    entry = result.trace[0]
    assert entry.tool == "query"
    assert entry.args == {"q": "customer"}
    assert entry.response_size == len(payload.encode("utf-8"))
    assert entry.is_error is False
    # Token totals sum across both steps.
    assert result.total_prompt_tokens == 50
    assert result.total_completion_tokens == 13
    assert result.total_llm_latency_ms == 150

    # The second LLM call must have seen the assistant turn AND the
    # tool_results turn — otherwise the model can't act on the tool output.
    second_call_history = client.captured_messages[1]
    assert len(second_call_history) == 3, second_call_history
    assert second_call_history[0].role == "user"
    assert second_call_history[0].text == "how many customers?"
    assert second_call_history[1].role == "assistant"
    assert second_call_history[1].tool_calls[0].name == "query"
    assert second_call_history[2].role == "user"
    assert second_call_history[2].tool_results[0].id == "call_1"
    assert second_call_history[2].tool_results[0].content == payload


def test_loop_carries_assistant_text_alongside_tool_calls_into_history() -> None:
    """A 'thinks-out-loud' assistant turn keeps its text in the next request.

    When the model emits both a text block and a tool_use block in the
    same step, the loop must preserve the text in the assistant turn
    it appends — otherwise the next request loses the model's
    reasoning. The LLM-client layer already serialises this shape;
    this test pins the loop's behaviour end-to-end.
    """
    client = _ScriptedAgentClient(
        steps=[
            AgentStep(
                text="I'll check the customer table first.",
                tool_calls=(
                    AgentToolCall(id="c1", name="query", arguments={"q": "customer"}),
                ),
                prompt_tokens=15,
                completion_tokens=8,
                latency_ms=40,
            ),
            AgentStep(
                text="SELECT * FROM customer",
                tool_calls=(),
                prompt_tokens=20,
                completion_tokens=6,
                latency_ms=40,
            ),
        ]
    )
    invoker = _make_invoker(by_name={"query": lambda _a: _ok_outcome("{}")})

    result = run_agent_loop(
        client=client,
        system="sys",
        user="how many customers?",
        tools=[_tool("query")],
        invoke_tool=invoker,
        model="m",
        temperature=0.0,
    )

    assert result.text == "SELECT * FROM customer"
    second_history = client.captured_messages[1]
    assistant_turn = second_history[1]
    assert assistant_turn.role == "assistant"
    assert assistant_turn.text == "I'll check the customer table first."
    assert assistant_turn.tool_calls[0].name == "query"


def test_records_tool_error_in_trace_but_continues() -> None:
    """A tool-level error is surfaced to the LLM; the loop does not abort."""
    client = _ScriptedAgentClient(
        steps=[
            AgentStep(
                text=None,
                tool_calls=(AgentToolCall(id="x", name="oops", arguments={}),),
                prompt_tokens=None,
                completion_tokens=None,
                latency_ms=10,
            ),
            AgentStep(
                text="SELECT 1",
                tool_calls=(),
                prompt_tokens=None,
                completion_tokens=None,
                latency_ms=10,
            ),
        ]
    )
    invoker = _make_invoker(by_name={})  # nothing → fallback returns is_error
    result = run_agent_loop(
        client=client,
        system="sys",
        user="?",
        tools=[_tool("oops")],
        invoke_tool=invoker,
        model="m",
        temperature=0.0,
    )
    assert result.iterations == 2
    assert result.trace[0].is_error is True


def test_truncates_long_tool_results_in_followup_message() -> None:
    """Tool result longer than the cap is trimmed before the next LLM call."""
    big_payload = "x" * 12_000
    client = _ScriptedAgentClient(
        steps=[
            AgentStep(
                text=None,
                tool_calls=(AgentToolCall(id="c", name="big", arguments={}),),
                prompt_tokens=None,
                completion_tokens=None,
                latency_ms=1,
            ),
            AgentStep(
                text="SELECT 1",
                tool_calls=(),
                prompt_tokens=None,
                completion_tokens=None,
                latency_ms=1,
            ),
        ]
    )
    invoker = _make_invoker(by_name={"big": lambda _a: _ok_outcome(big_payload)})

    result = run_agent_loop(
        client=client,
        system="sys",
        user="?",
        tools=[_tool("big")],
        invoke_tool=invoker,
        model="m",
        temperature=0.0,
    )
    # The trace records the FULL response_size (auditable), but the
    # message handed back to the LLM is truncated.
    assert result.trace[0].response_size == len(big_payload.encode("utf-8"))
    second_history = client.captured_messages[1]
    forwarded = second_history[-1].tool_results[0].content
    assert len(forwarded) < len(big_payload)
    assert forwarded.endswith("…[truncated]")


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_raises_when_iteration_cap_exceeded() -> None:
    """Loop refuses to run forever when the LLM keeps requesting tools."""
    forever_step = AgentStep(
        text=None,
        tool_calls=(AgentToolCall(id="c", name="t", arguments={}),),
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=1,
    )
    client = _ScriptedAgentClient(steps=[forever_step] * (DEFAULT_MAX_ITERATIONS + 1))
    invoker = _make_invoker(by_name={"t": lambda _a: _ok_outcome("ok")})
    with pytest.raises(AgentLoopError, match="iterations"):
        run_agent_loop(
            client=client,
            system="sys",
            user="?",
            tools=[_tool("t")],
            invoke_tool=invoker,
            model="m",
            temperature=0.0,
        )


def test_raises_on_empty_step() -> None:
    """A step with no text AND no tool calls is a provider bug — fail loud."""
    client = _ScriptedAgentClient(
        steps=[
            AgentStep(
                text=None,
                tool_calls=(),
                prompt_tokens=None,
                completion_tokens=None,
                latency_ms=1,
            )
        ]
    )
    with pytest.raises(AgentLoopError, match="empty step"):
        run_agent_loop(
            client=client,
            system="sys",
            user="?",
            tools=[_tool("query")],
            invoke_tool=_make_invoker(by_name={}),
            model="m",
            temperature=0.0,
        )


def test_max_iterations_respected_when_cap_is_one() -> None:
    """Custom cap of 1 stops the loop after the first round-trip."""
    forever_step = AgentStep(
        text=None,
        tool_calls=(AgentToolCall(id="c", name="t", arguments={}),),
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=1,
    )
    client = _ScriptedAgentClient(steps=[forever_step, forever_step])
    invoker = _make_invoker(by_name={"t": lambda _a: _ok_outcome("ok")})
    with pytest.raises(AgentLoopError):
        run_agent_loop(
            client=client,
            system="sys",
            user="?",
            tools=[_tool("t")],
            invoke_tool=invoker,
            model="m",
            temperature=0.0,
            max_iterations=1,
        )
    # The client was called exactly once before raising.
    assert len(client.captured_messages) == 1
