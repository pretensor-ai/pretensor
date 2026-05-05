"""LLM tool-use agent loop for the L3 pretensor runner.

The runner asks the LLM to translate one NL question into PostgreSQL,
giving the model the MCP tool set discovered from ``pretensor serve``.
The loop here is provider-agnostic: it owns the message history, calls
the LLM (via :class:`AgentLlmClient`), routes tool invocations through
a caller-supplied callback, and stops as soon as the model returns
final text (the SQL).

A hard cap on iterations keeps a confused model from running away
forever. Per-call telemetry (latency, tokens, tool trace) is returned
so the runner can populate the JSON envelope without a second pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

__all__ = [
    "AgentLlmClient",
    "AgentLoopError",
    "AgentLoopResult",
    "AgentMessage",
    "AgentStep",
    "AgentTool",
    "AgentToolCall",
    "AgentToolResult",
    "DEFAULT_MAX_ITERATIONS",
    "ToolCallTraceEntry",
    "ToolInvocationOutcome",
    "ToolInvoker",
    "run_agent_loop",
]


DEFAULT_MAX_ITERATIONS = 16
"""Cap on the LLM ↔ tool round-trips per question.

Pagila's hardest gold question solves in ≤4 round-trips today; 16
leaves headroom for messier schemas without letting a confused model
spin forever. Hitting the cap is recorded in ``notes[]`` and the
question is marked failed — never silently accepted.
"""


_TOOL_RESULT_TRUNCATE_BYTES = 8000
"""Soft cap on the tool result handed back to the LLM, in UTF-8 bytes.

The MCP server can return large payloads (e.g. ``cypher`` over a
big graph). Trimming protects the next request's prompt budget; the
tool trace stores the *full* ``response_size`` (also in bytes) so
an auditor can see when truncation kicked in. Both values share the
same unit so comparing them is unambiguous.
"""

_TRUNCATION_SUFFIX = "…[truncated]"


@dataclass(frozen=True, slots=True)
class AgentTool:
    """An MCP tool exposed to the agent.

    Mirrors :class:`mcp.types.Tool` but lives in the L3 namespace so
    callers don't have to import the MCP package to read the trace.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """One tool invocation requested by the LLM.

    ``id`` is the provider's identifier (Anthropic ``tool_use_id``,
    OpenAI ``tool_call_id``); we forward it back unchanged so the
    next request can pair tool results with their calls.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    """Result of executing one :class:`AgentToolCall`."""

    id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """One turn of the LLM conversation, provider-agnostic.

    Role is either ``"assistant"`` (LLM produced this turn) or
    ``"user"`` (we produced this turn — initial question, or a batch
    of tool results). A single assistant turn may contain text and
    one or more tool calls; a single user turn either holds the
    initial prompt text or a batch of tool results.
    """

    role: Literal["user", "assistant"]
    text: str | None = None
    tool_calls: tuple[AgentToolCall, ...] = ()
    tool_results: tuple[AgentToolResult, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One round-trip with the LLM.

    Either ``text`` is set (the model finished and returned a final
    answer — for L3, the SQL) or ``tool_calls`` is non-empty (the
    model wants to call tools before producing text). Token counts
    are optional; some providers omit them.
    """

    text: str | None
    tool_calls: tuple[AgentToolCall, ...]
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int


class AgentLlmClient(Protocol):
    """Single-step tool-use surface for the L3 pretensor runner.

    Implementations translate :class:`AgentMessage` history to the
    provider's wire format (Anthropic content blocks, OpenAI
    ``tool_calls``, etc.) and back. The agent loop does not see the
    provider-specific shape.
    """

    def agent_complete(
        self,
        *,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        model: str,
        temperature: float,
    ) -> AgentStep: ...


class AgentLoopError(RuntimeError):
    """Raised when the loop cannot make further progress.

    Specifically: the iteration cap was hit without the LLM returning
    final text, OR the LLM produced an empty step (no text and no
    tool calls — providers shouldn't but it's not impossible).
    """


@dataclass(frozen=True, slots=True)
class ToolCallTraceEntry:
    """One row of the per-question tool-call trace.

    Mirrors AC-mandated shape: ``{tool, args, response_size}``.
    ``is_error`` flags MCP-level failures (subprocess error, unknown
    tool, handler exception); the runner uses it to surface failed
    runs without losing the rest of the trace.
    """

    tool: str
    args: dict[str, Any]
    response_size: int
    is_error: bool


@dataclass(slots=True)
class AgentLoopResult:
    """Final outcome of the loop for one question.

    ``text`` is the LLM's last textual reply (the SQL, with no fence
    cleanup yet — the runner does that). ``trace`` is the ordered list
    of tool calls; ``iterations`` is how many LLM round-trips ran. Token
    and latency totals are sums across all steps.
    """

    text: str
    trace: list[ToolCallTraceEntry] = field(default_factory=list)
    iterations: int = 0
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_llm_latency_ms: int = 0


@dataclass(frozen=True, slots=True)
class ToolInvocationOutcome:
    """Outcome of one tool invocation, ready to attach back to the LLM.

    Plain ``content`` + ``is_error`` so the runner doesn't have to know
    about provider-specific tool_use ids — the loop pairs the outcome
    with the originating call when it builds the next user turn.
    """

    content: str
    is_error: bool = False


ToolInvoker = Callable[[str, dict[str, Any]], ToolInvocationOutcome]
"""Sync callable the loop uses to run one tool.

The runner provides this — typically wrapping an
:class:`McpClient.call_tool` plus serialisation. Returning a
:class:`ToolInvocationOutcome` (rather than a raw dict) lets the runner
mark tool-level errors that the LLM should still see.
"""


def run_agent_loop(
    *,
    client: AgentLlmClient,
    system: str,
    user: str,
    tools: list[AgentTool],
    invoke_tool: ToolInvoker,
    model: str,
    temperature: float,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> AgentLoopResult:
    """Run one question to completion against ``client`` and ``invoke_tool``.

    Each iteration sends the full conversation history to the LLM. If
    the model returns text, the loop exits with that text. If the model
    returns tool calls, each call is executed (via ``invoke_tool``),
    the results are appended as a user turn, and the loop continues.
    """
    messages: list[AgentMessage] = [AgentMessage(role="user", text=user)]
    trace: list[ToolCallTraceEntry] = []
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_latency_ms = 0

    for iteration in range(1, max_iterations + 1):
        step = client.agent_complete(
            system=system,
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
        )
        total_latency_ms += step.latency_ms
        if step.prompt_tokens is not None:
            total_prompt_tokens = (total_prompt_tokens or 0) + step.prompt_tokens
        if step.completion_tokens is not None:
            total_completion_tokens = (
                total_completion_tokens or 0
            ) + step.completion_tokens

        if step.text is not None and not step.tool_calls:
            return AgentLoopResult(
                text=step.text,
                trace=trace,
                iterations=iteration,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_llm_latency_ms=total_latency_ms,
            )

        if not step.tool_calls:
            raise AgentLoopError(
                f"LLM returned an empty step at iteration {iteration} "
                "(no text and no tool calls)."
            )

        # Record the assistant turn (text + tool calls) so the next request
        # carries the full history. ``step.text`` may be non-empty when the
        # model "thinks out loud" before calling a tool — keep it.
        messages.append(
            AgentMessage(
                role="assistant",
                text=step.text,
                tool_calls=step.tool_calls,
            )
        )

        results: list[AgentToolResult] = []
        for call in step.tool_calls:
            outcome = invoke_tool(call.name, call.arguments)
            content_bytes = outcome.content.encode("utf-8")
            trace.append(
                ToolCallTraceEntry(
                    tool=call.name,
                    args=dict(call.arguments),
                    response_size=len(content_bytes),
                    is_error=outcome.is_error,
                )
            )
            content = outcome.content
            if len(content_bytes) > _TOOL_RESULT_TRUNCATE_BYTES:
                # Slice on bytes to keep the truncation cap unambiguous;
                # decode with ``errors="ignore"`` so a multibyte
                # codepoint straddling the boundary doesn't raise.
                content = (
                    content_bytes[:_TOOL_RESULT_TRUNCATE_BYTES].decode(
                        "utf-8", errors="ignore"
                    )
                    + _TRUNCATION_SUFFIX
                )
            results.append(
                AgentToolResult(id=call.id, content=content, is_error=outcome.is_error)
            )

        messages.append(AgentMessage(role="user", tool_results=tuple(results)))

    raise AgentLoopError(
        f"Agent did not produce final text within {max_iterations} iterations."
    )
