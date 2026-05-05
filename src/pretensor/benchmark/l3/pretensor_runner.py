"""``run_l3_pretensor`` — agent + Pretensor MCP runner.

Same gold question set, same SQL grader, same JSON envelope as the
baseline (``pretensor.benchmark.l3.runner.run_l3_baseline``); the
difference is what the agent gets fed.

The baseline hands the LLM a frozen DDL dump. This runner hands it a
live MCP session against a ``pretensor serve`` subprocess, lets the LLM
discover the tool catalogue, and routes its tool calls through the MCP
client. Per-question records pick up an extra ``tool_calls`` field so
each agent decision is auditable from the JSON alone.

The runner is otherwise sync: the MCP client is the only async surface
and lives behind a sync wrapper (``StdioMcpClient``), so the call shape
matches the baseline runner one-to-one.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import secrets
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast

from pretensor.benchmark.fixtures import load_dataset
from pretensor.benchmark.l3.agent import (
    DEFAULT_MAX_ITERATIONS,
    AgentLlmClient,
    AgentLoopError,
    AgentTool,
    ToolCallTraceEntry,
    ToolInvocationOutcome,
    run_agent_loop,
)
from pretensor.benchmark.l3.db import (
    QueryExecutionError,
    execute_query,
    resolve_database_url,
)
from pretensor.benchmark.l3.gold import L3GoldEntry, load_l3_gold
from pretensor.benchmark.l3.llm_client import AnthropicHttpClient, LlmCallError
from pretensor.benchmark.l3.mcp_client import McpClient, StdioMcpClient
from pretensor.benchmark.l3.prompt import (
    build_pretensor_system_prompt,
    pretensor_prompt_template_hash,
    prompt_hash,
    strip_sql_fences,
)
from pretensor.benchmark.l3.sql_equivalence import gold_is_ordered, rows_equivalent
from pretensor.benchmark.results import BenchmarkResult, Metric, write_json
from pretensor.benchmark.runner import Dataset

__all__ = ["DEFAULT_TEMPERATURE", "run_l3_pretensor"]


DEFAULT_TEMPERATURE = 0.0
"""LLM temperature; matches the baseline so the comparison is honest."""

_DETERMINISTIC_RAN_AT = "1970-01-01T00:00:00Z"
"""Pinned timestamp; mirrors the baseline runner's reproducibility convention."""

_AGENT_ERROR_MAX_CHARS = 500
"""Truncation limit for per-item ``error`` strings (matches the baseline)."""

_SUCCESS_RATE_METRIC = "nl2sql_success_rate_pretensor"
_LATENCY_METRIC = "mean_latency_ms_pretensor"

_SEMANTIC_SEARCH_TOOL = "semantic_search"
"""The MCP tool that signals the embeddings extra is loaded server-side.

Per the L3 contract the runner stays embeddings-agnostic: it inspects
the catalogue advertised by ``pretensor serve`` and reports whatever
state it found. If the embeddings extra was installed in the serve
process, ``semantic_search`` is registered and shows up here.
"""


def run_l3_pretensor(
    dataset: Dataset,
    out: Path | None,
    graph_dir: Path,
    *,
    model: str,
    seed: int | None,
    llm_client: AgentLlmClient | None = None,
    mcp_client: McpClient | None = None,
) -> None:
    """Run the L3 pretensor (agent + MCP) variant for ``dataset``.

    ``llm_client`` and ``mcp_client`` are Python-API-only seams — the
    CLI never exposes them. Tests inject fakes to drive the runner
    without spawning a subprocess or hitting a real LLM provider.
    Production callers leave both ``None`` and the runner builds an
    :class:`AnthropicHttpClient` against ``ANTHROPIC_API_KEY`` plus a
    :class:`StdioMcpClient` against ``pretensor serve`` over ``graph_dir``.

    The runner refuses to run when ``graph_dir`` does not contain an
    indexed graph — for L3 parity, the baseline and pretensor variants
    must see the same data, and silently re-indexing would defeat the
    comparison. The user is told to run ``pretensor index`` first.
    """
    fixture = load_dataset(dataset)
    _questions_path, questions_bytes, questions = load_l3_gold(fixture)
    dsn = resolve_database_url(dataset)
    _require_indexed_graph_dir(graph_dir)

    resolved_seed = seed if seed is not None else secrets.randbits(63)
    system_prompt = build_pretensor_system_prompt(dataset.value)
    rendered_hash = prompt_hash(system_prompt)
    template_hash = pretensor_prompt_template_hash()

    client: AgentLlmClient
    if llm_client is not None:
        client = llm_client
    else:
        # Mirror the baseline runner's friendly-exit pattern: convert the
        # missing-key LlmCallError into a LookupError so the CLI's
        # _handle_input_error path renders a one-line message rather than
        # a Python traceback.
        try:
            client = AnthropicHttpClient()
        except LlmCallError as exc:
            raise LookupError(str(exc)) from exc

    with _resolve_mcp_client(mcp_client, graph_dir=graph_dir) as mcp:
        tools = mcp.list_tools()
        embeddings_enabled = any(t.name == _SEMANTIC_SEARCH_TOOL for t in tools)

        notes = _build_notes(tools=tools, embeddings_enabled=embeddings_enabled)

        per_item: list[dict[str, Any]] = []
        latencies_ms: list[int] = []
        successes = 0

        for entry in questions:
            record = _evaluate_one(
                entry=entry,
                system_prompt=system_prompt,
                client=client,
                tools=tools,
                mcp=mcp,
                model=model,
                temperature=DEFAULT_TEMPERATURE,
                dsn=dsn,
            )
            per_item.append(record)
            latencies_ms.append(int(record["latency_ms"]))
            if record["pretensor_pass"]:
                successes += 1

    per_item.sort(key=lambda r: cast(str, r["id"]))

    metrics: dict[str, Metric] = {
        _SUCCESS_RATE_METRIC: Metric(
            value=(successes / len(questions)) if questions else None,
            direction="higher_is_better",
        ),
        _LATENCY_METRIC: Metric(
            value=(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else None,
            direction="lower_is_better",
        ),
    }

    fixture_sha = "sha256:" + hashlib.sha256(questions_bytes).hexdigest()

    extra: dict[str, Any] = {
        "runner": "pretensor",
        "provider": _provider_name(client),
        "model": model,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": resolved_seed,
        "prompt_template_hash": template_hash,
        "prompt_hash": rendered_hash,
        "tool_catalogue": [t.name for t in tools],
        "max_iterations": DEFAULT_MAX_ITERATIONS,
    }

    result = BenchmarkResult(
        level="l3",
        dataset=dataset.value,
        pretensor_version=_resolve_version(),
        embeddings_enabled=embeddings_enabled,
        ran_at=_DETERMINISTIC_RAN_AT,
        fixture_sha=fixture_sha,
        metrics=metrics,
        per_item=per_item,
        notes=notes,
        extra=extra,
    )

    if out is None:
        sys.stdout.write(json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n")
        return
    _atomic_write_json(result, out)


def _evaluate_one(
    *,
    entry: L3GoldEntry,
    system_prompt: str,
    client: AgentLlmClient,
    tools: list[AgentTool],
    mcp: McpClient,
    model: str,
    temperature: float,
    dsn: str,
) -> dict[str, Any]:
    """Grade one question end-to-end through the agent + MCP loop.

    Wraps :func:`_populate_record` in a try/finally so ``latency_ms`` is
    always set to the full per-question wall-clock — including SQL
    execution and equivalence checking, not just the LLM round-trips.
    The aggregated LLM-only timing is recorded separately as
    ``llm_latency_ms`` for cost / regression diagnostics.
    """
    record: dict[str, Any] = {
        "id": entry.id,
        "question": entry.question,
        "gold_sql": entry.expected_sql,
        "agent_sql": "",
        "tool_calls": [],
        "execution_success": False,
        "result_equivalence": False,
        "latency_ms": 0,
        "llm_latency_ms": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "iterations": 0,
        "row_count_gold": 0,
        "row_count_agent": 0,
        "pretensor_pass": False,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        _populate_record(
            record,
            entry=entry,
            system_prompt=system_prompt,
            client=client,
            tools=tools,
            mcp=mcp,
            model=model,
            temperature=temperature,
            dsn=dsn,
        )
    finally:
        record["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return record


def _populate_record(
    record: dict[str, Any],
    *,
    entry: L3GoldEntry,
    system_prompt: str,
    client: AgentLlmClient,
    tools: list[AgentTool],
    mcp: McpClient,
    model: str,
    temperature: float,
    dsn: str,
) -> None:
    """Walk one question through the agent loop → SQL → equivalence pipeline.

    Mutates ``record`` in place. Errors at any stage short-circuit the
    function and are surfaced in ``record["error"]``; the surrounding
    ``_evaluate_one`` always sets ``latency_ms`` regardless of which
    branch we exited through.
    """

    def invoke(name: str, arguments: dict[str, Any]) -> ToolInvocationOutcome:
        outcome = mcp.call_tool(name, arguments)
        return ToolInvocationOutcome(content=outcome.content, is_error=outcome.is_error)

    try:
        loop_result = run_agent_loop(
            client=client,
            system=system_prompt,
            user=entry.question,
            tools=tools,
            invoke_tool=invoke,
            model=model,
            temperature=temperature,
        )
    except LlmCallError as exc:
        record["error"] = _truncate(f"LLM call failed: {exc}")
        return
    except AgentLoopError as exc:
        record["error"] = _truncate(f"agent loop failed: {exc}")
        return

    record["llm_latency_ms"] = loop_result.total_llm_latency_ms
    record["prompt_tokens"] = loop_result.total_prompt_tokens
    record["completion_tokens"] = loop_result.total_completion_tokens
    record["iterations"] = loop_result.iterations
    record["tool_calls"] = [_trace_to_dict(t) for t in loop_result.trace]

    agent_sql = strip_sql_fences(loop_result.text).strip()
    record["agent_sql"] = agent_sql
    if not agent_sql:
        record["error"] = "agent returned an empty SQL string"
        return

    try:
        gold_rows, _ = execute_query(dsn, entry.expected_sql)
    except QueryExecutionError as exc:
        record["error"] = _truncate(f"gold SQL failed to execute: {exc}")
        return
    record["row_count_gold"] = len(gold_rows)

    try:
        agent_rows, _ = execute_query(dsn, agent_sql, enforce_select_only=True)
    except QueryExecutionError as exc:
        record["error"] = _truncate(f"agent SQL failed to execute: {exc}")
        return
    record["row_count_agent"] = len(agent_rows)
    record["execution_success"] = True

    equivalence = rows_equivalent(
        gold_rows, agent_rows, ordered=gold_is_ordered(entry.expected_sql)
    )
    record["result_equivalence"] = equivalence.equivalent
    if not equivalence.equivalent:
        record["error"] = _truncate(f"row equivalence failed: {equivalence.reason}")
    record["pretensor_pass"] = (
        record["execution_success"] and record["result_equivalence"]
    )


def _trace_to_dict(entry: ToolCallTraceEntry) -> dict[str, Any]:
    """Render one tool-call trace entry for the JSON envelope.

    ``is_error`` is always present (matching every other boolean in
    the per-item record). A consumer reading the envelope can do
    ``entry["is_error"]`` without a KeyError on success entries.
    """
    return {
        "tool": entry.tool,
        "args": entry.args,
        "response_size": entry.response_size,
        "is_error": entry.is_error,
    }


def _build_notes(*, tools: list[AgentTool], embeddings_enabled: bool) -> list[str]:
    """Compose the ``notes[]`` block.

    Always carries the LLM-determinism caveat so an auditor reads the
    same disclaimer the baseline runner emits. Adds catalogue + embeddings
    notes so a reader can infer the run's tool surface from the envelope
    alone (without re-running ``pretensor serve``).
    """
    notes = [
        "L3 is LLM-bound; the --seed value governs harness state only "
        "(the Anthropic / OpenAI APIs do not accept a seed parameter). "
        "Same-seed reruns at temperature 0 may still differ by remaining "
        "LLM nondeterminism — including the order or choice of MCP tool "
        "calls inside the agent loop."
    ]
    notes.append(
        f"MCP tool catalogue ({len(tools)} tools): "
        + ", ".join(sorted(t.name for t in tools))
    )
    if not embeddings_enabled:
        notes.append(
            "embeddings extra not detected on the serve subprocess "
            f"(no '{_SEMANTIC_SEARCH_TOOL}' tool in the catalogue); "
            "the agent ran without semantic search."
        )
    return notes


def _require_indexed_graph_dir(graph_dir: Path) -> None:
    """Refuse to run when ``graph_dir`` clearly has no indexed graph.

    Looks for a ``graphs/`` subdirectory containing at least one
    ``*.kuzu`` file. The check is deliberately shallow — verifying
    that an *index* is fresh would couple this runner to the graph-dir
    layout details, which the indexing subsystem owns. The message
    points the user at the canonical fix (`pretensor index ...`) so an
    operator can self-serve.
    """
    graphs = graph_dir / "graphs"
    if not graphs.is_dir() or not any(graphs.glob("*.kuzu")):
        raise FileNotFoundError(
            f"L3 pretensor runner requires an indexed graph at {graph_dir}/graphs/. "
            "Run `pretensor index <dsn>` first to populate it."
        )


@contextmanager
def _resolve_mcp_client(
    injected: McpClient | None, *, graph_dir: Path
) -> Iterator[McpClient]:
    """Yield the injected client when given, else stand up a stdio client.

    The injected branch deliberately does NOT enter a context manager
    on the caller's behalf — tests own the lifecycle of their fakes.
    The stdio branch enters/exits :class:`StdioMcpClient` so the
    subprocess is reaped on every code path through the runner.

    A :class:`McpClientError` raised by :class:`StdioMcpClient` (broken
    binary, transport handshake failure) propagates unchanged — it is a
    distinct condition from the missing-graph-dir error raised by
    :func:`_require_indexed_graph_dir`. The CLI handles both cleanly
    via its input-error branch; collapsing them here would lose the
    distinction operators need to debug a broken serve install.
    """
    if injected is not None:
        yield injected
        return
    with StdioMcpClient(graph_dir=graph_dir) as stdio_client_:
        yield stdio_client_


def _truncate(s: str) -> str:
    return s if len(s) <= _AGENT_ERROR_MAX_CHARS else s[:_AGENT_ERROR_MAX_CHARS] + "…"


def _atomic_write_json(result: BenchmarkResult, out: Path) -> None:
    """Write JSON to a temp sibling and atomically rename into place."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    write_json(result, tmp)
    os.replace(tmp, out)


def _provider_name(client: AgentLlmClient) -> str:
    """Best-effort provider tag for the envelope.

    Mirrors the baseline runner's helper so an auditor can compare the
    two envelopes without reaching for provider-specific docs.
    """
    if isinstance(client, AnthropicHttpClient):
        return "anthropic"
    return "custom"


def _resolve_version() -> str:
    try:
        return importlib.metadata.version("pretensor")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+unknown"
