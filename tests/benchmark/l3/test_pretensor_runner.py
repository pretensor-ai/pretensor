"""Unit tests for ``run_l3_pretensor``.

A scripted ``AgentLlmClient`` and an in-memory ``McpClient`` drive the
runner without spawning ``pretensor serve`` or hitting an LLM API.
The end-to-end path against a real subprocess is exercised by
``tests/e2e/test_l3_pretensor.py`` (gated on PRETENSOR_E2E=1).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import pretensor.benchmark.l3.pretensor_runner as runner_mod
from pretensor.benchmark.l3.agent import (
    AgentMessage,
    AgentStep,
    AgentTool,
    AgentToolCall,
)
from pretensor.benchmark.l3.llm_client import LlmCallError
from pretensor.benchmark.l3.mcp_client import McpToolResult
from pretensor.benchmark.l3.pretensor_runner import run_l3_pretensor
from pretensor.benchmark.results import read_json
from pretensor.benchmark.runner import Dataset

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _ScriptedAgentClient:
    """Returns canned :class:`AgentStep` sequences keyed by user question.

    The runner sends one ``user`` message per question (the gold prompt),
    so we key responses on that text. Each question gets its own queue
    of steps; the LLM may respond with tool calls before producing
    final text, so the queue lets a single question take multiple
    round-trips.
    """

    steps_by_question: dict[str, deque[AgentStep]] = field(default_factory=dict)
    raises: dict[str, Exception] = field(default_factory=dict)

    def agent_complete(
        self,
        *,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        model: str,
        temperature: float,
    ) -> AgentStep:
        question = _initial_user_text(messages)
        if question in self.raises:
            raise self.raises[question]
        queue = self.steps_by_question.get(question)
        if not queue:
            raise AssertionError(f"no scripted steps for question: {question!r}")
        return queue.popleft()


@dataclass
class _InMemoryMcpClient:
    """In-memory stand-in for :class:`StdioMcpClient`.

    Provides ``list_tools`` and ``call_tool`` without spawning a process.
    ``responses`` maps tool name → canned content; default is an empty
    JSON object so the runner doesn't crash when a script forgets to
    register a tool.
    """

    tools: list[AgentTool]
    responses: dict[str, str] = field(default_factory=dict)
    is_error_for: set[str] = field(default_factory=set)

    def list_tools(self) -> list[AgentTool]:
        return list(self.tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        return McpToolResult(
            content=self.responses.get(name, "{}"),
            is_error=name in self.is_error_for,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _final_step(text: str) -> AgentStep:
    return AgentStep(
        text=text,
        tool_calls=(),
        prompt_tokens=10,
        completion_tokens=4,
        latency_ms=42,
    )


def _initial_user_text(messages: list[AgentMessage]) -> str:
    """Pull the original NL question off the front of the conversation."""
    if not messages or messages[0].role != "user" or messages[0].text is None:
        raise AssertionError("loop did not lead with the user prompt")
    return messages[0].text


def _load_pagila_questions() -> list[dict[str, Any]]:
    """Load the real Pagila gold set so tests pin the AC contract."""
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "scripts" / "data" / "pagila_nl2sql_bench.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_indexed_graph_dir(tmp_path: Path) -> Path:
    """Create a graph_dir layout that satisfies ``_require_indexed_graph_dir``."""
    graph_dir = tmp_path / ".pretensor"
    (graph_dir / "graphs").mkdir(parents=True, exist_ok=True)
    (graph_dir / "graphs" / "pagila.kuzu").touch()
    return graph_dir


def _patch_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gold_rows_by_sql: dict[str, list[tuple[Any, ...]]] | None = None,
    agent_rows_by_sql: dict[str, list[tuple[Any, ...]]] | None = None,
    raise_on_agent_sql: dict[str, Exception] | None = None,
    raise_on_gold_sql: dict[str, Exception] | None = None,
) -> None:
    """Stub ``execute_query`` and the per-dataset DB env var.

    Mirrors the baseline runner's test helper so the two test suites
    look the same to a future reader; differences live only in what
    each runner records per-item. ``raise_on_gold_sql`` and
    ``raise_on_agent_sql`` let tests inject :class:`QueryExecutionError`
    on either branch independently so the runner's "gold failed" and
    "agent failed" recovery paths are exercised in isolation.
    """
    monkeypatch.setenv("PAGILA_DATABASE_URL", "postgresql://stub/pagila")
    gold_map = gold_rows_by_sql or {}
    agent_map = agent_rows_by_sql or {}
    agent_raises = raise_on_agent_sql or {}
    gold_raises = raise_on_gold_sql or {}

    def fake_execute_query(
        dsn: str,
        sql: str,
        *,
        enforce_select_only: bool = False,
        statement_timeout_ms: int = 30_000,
    ) -> tuple[list[tuple[Any, ...]], list[str]]:
        if enforce_select_only:
            if sql in agent_raises:
                raise agent_raises[sql]
            return list(agent_map.get(sql, [])), []
        if sql in gold_raises:
            raise gold_raises[sql]
        return list(gold_map.get(sql, [])), []

    monkeypatch.setattr(runner_mod, "execute_query", fake_execute_query)


def _gold_only_client(
    fixture_questions: list[dict[str, Any]],
) -> _ScriptedAgentClient:
    """Return a client that emits each question's gold SQL with no tool calls."""
    return _ScriptedAgentClient(
        steps_by_question={
            q["question"]: deque([_final_step(q["expected_sql"])])
            for q in fixture_questions
        }
    )


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_runner_emits_required_envelope_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All AC-required envelope keys are present on a successful run."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [("answer",)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    client = _gold_only_client(fixture_questions)
    mcp = _InMemoryMcpClient(
        tools=[AgentTool(name="query", description="search", input_schema={})]
    )

    out = tmp_path / "result.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="claude-haiku-4-5",
        seed=42,
        llm_client=client,
        mcp_client=mcp,
    )
    result = read_json(out)
    assert result.level == "l3"
    assert result.dataset == "pagila"
    assert "nl2sql_success_rate_pretensor" in result.metrics
    assert "mean_latency_ms_pretensor" in result.metrics
    success = result.metrics["nl2sql_success_rate_pretensor"]
    assert success.direction == "higher_is_better"
    assert success.value == 1.0
    latency = result.metrics["mean_latency_ms_pretensor"]
    assert latency.direction == "lower_is_better"

    assert result.extra["runner"] == "pretensor"
    assert result.extra["model"] == "claude-haiku-4-5"
    assert result.extra["temperature"] == 0.0
    assert result.extra["seed"] == 42
    assert result.extra["tool_catalogue"] == ["query"]
    assert isinstance(result.extra["prompt_template_hash"], str)
    assert isinstance(result.extra["prompt_hash"], str)


def test_per_item_records_have_ac_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-item shape includes tool_calls, the new pretensor-only field."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    client = _gold_only_client(fixture_questions)
    mcp = _InMemoryMcpClient(tools=[])

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=mcp,
    )
    result = read_json(out)
    required = {
        "id",
        "question",
        "gold_sql",
        "agent_sql",
        "tool_calls",
        "execution_success",
        "result_equivalence",
        "latency_ms",
        "iterations",
        "pretensor_pass",
        "error",
    }
    for record in result.per_item:
        missing = required - set(record.keys())
        assert not missing, f"missing AC-required fields {missing}"
        assert isinstance(record["tool_calls"], list)


def test_tool_calls_recorded_in_per_item_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the LLM uses tools, the trace shows ``{tool, args, response_size}``.

    Covers AC #4: the trace can mix ``query``, ``context``, and
    ``traverse`` together — the runner does not filter or reorder
    tool calls; whatever the LLM emitted lands in the per-item
    record in invocation order.
    """
    fixture_questions = _load_pagila_questions()
    target = fixture_questions[0]
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    other_steps: dict[str, deque[AgentStep]] = {
        q["question"]: deque([_final_step(q["expected_sql"])])
        for q in fixture_questions[1:]
    }
    target_question = target["question"]
    other_steps[target_question] = deque(
        [
            AgentStep(
                text=None,
                tool_calls=(
                    AgentToolCall(id="t1", name="query", arguments={"q": "customer"}),
                    AgentToolCall(
                        id="t2", name="context", arguments={"table": "customer"}
                    ),
                    AgentToolCall(
                        id="t3",
                        name="traverse",
                        arguments={"from_table": "customer", "to_table": "rental"},
                    ),
                ),
                prompt_tokens=30,
                completion_tokens=6,
                latency_ms=80,
            ),
            _final_step(target["expected_sql"]),
        ]
    )
    client = _ScriptedAgentClient(steps_by_question=other_steps)
    mcp = _InMemoryMcpClient(
        tools=[
            AgentTool(name="query", description="search", input_schema={}),
            AgentTool(name="context", description="ctx", input_schema={}),
            AgentTool(name="traverse", description="traverse", input_schema={}),
        ],
        responses={
            "query": json.dumps({"hits": []}),
            "context": json.dumps({"columns": []}),
            "traverse": json.dumps({"path": []}),
        },
    )

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=mcp,
    )
    result = read_json(out)
    record = next(r for r in result.per_item if r["id"] == target["id"])
    tool_names = [call["tool"] for call in record["tool_calls"]]
    assert tool_names == ["query", "context", "traverse"]
    assert record["tool_calls"][0]["args"] == {"q": "customer"}
    assert all(call["response_size"] > 0 for call in record["tool_calls"])
    assert record["iterations"] == 2


def test_tool_call_trace_records_is_error_when_mcp_call_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When MCP returns ``is_error=True`` the per-item trace carries it through.

    Covers the ``_trace_to_dict`` branch that adds ``is_error: true``
    to the JSON envelope. Without this, an MCP-level fault during the
    agent loop would be invisible in the auditable per-item record.
    """
    fixture_questions = _load_pagila_questions()
    target = fixture_questions[0]
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    other_steps: dict[str, deque[AgentStep]] = {
        q["question"]: deque([_final_step(q["expected_sql"])])
        for q in fixture_questions[1:]
    }
    other_steps[target["question"]] = deque(
        [
            AgentStep(
                text=None,
                tool_calls=(
                    AgentToolCall(id="t1", name="query", arguments={"q": "noop"}),
                ),
                prompt_tokens=10,
                completion_tokens=2,
                latency_ms=5,
            ),
            _final_step(target["expected_sql"]),
        ]
    )
    client = _ScriptedAgentClient(steps_by_question=other_steps)
    mcp = _InMemoryMcpClient(
        tools=[AgentTool(name="query", description="", input_schema={})],
        responses={"query": '{"error": "broken"}'},
        is_error_for={"query"},
    )

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=mcp,
    )
    result = read_json(out)
    record = next(r for r in result.per_item if r["id"] == target["id"])
    assert len(record["tool_calls"]) == 1
    trace_entry = record["tool_calls"][0]
    assert trace_entry["tool"] == "query"
    assert trace_entry["is_error"] is True


def test_embeddings_enabled_reflects_semantic_search_tool_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``embeddings_enabled`` flips when ``semantic_search`` is in the catalogue."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )
    client = _gold_only_client(fixture_questions)
    mcp_with = _InMemoryMcpClient(
        tools=[
            AgentTool(name="query", description="", input_schema={}),
            AgentTool(name="semantic_search", description="", input_schema={}),
        ]
    )
    mcp_without = _InMemoryMcpClient(
        tools=[AgentTool(name="query", description="", input_schema={})]
    )
    graph_dir = _seed_indexed_graph_dir(tmp_path)

    out_with = tmp_path / "with.json"
    out_without = tmp_path / "without.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out_with,
        graph_dir,
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=mcp_with,
    )
    # _gold_only_client rebuilt: every question's queue is empty after the
    # first run. Build a fresh one for the second pass.
    client2 = _gold_only_client(fixture_questions)
    run_l3_pretensor(
        Dataset.PAGILA,
        out_without,
        graph_dir,
        model="m",
        seed=1,
        llm_client=client2,
        mcp_client=mcp_without,
    )

    assert read_json(out_with).embeddings_enabled is True
    assert read_json(out_without).embeddings_enabled is False


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_runner_records_llm_call_failure_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One per-question LLM failure must not abort the whole run."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    failing_question = fixture_questions[0]["question"]
    client = _gold_only_client(fixture_questions)
    client.raises[failing_question] = LlmCallError("simulated 500")

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=_InMemoryMcpClient(tools=[]),
    )
    result = read_json(out)
    failed = next(
        item for item in result.per_item if item["question"] == failing_question
    )
    assert failed["pretensor_pass"] is False
    assert failed["execution_success"] is False
    assert failed["agent_sql"] == ""
    assert failed["error"] is not None
    assert "LLM" in failed["error"]


def test_runner_records_agent_sql_execution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SQL execution error on the agent path lands in ``error`` per-item."""
    from pretensor.benchmark.l3.db import QueryExecutionError

    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}

    bad_question = fixture_questions[0]["question"]
    bad_sql = "SELECT bogus FROM nowhere"

    other_steps: dict[str, deque[AgentStep]] = {
        q["question"]: deque([_final_step(q["expected_sql"])])
        for q in fixture_questions[1:]
    }
    other_steps[bad_question] = deque([_final_step(bad_sql)])
    client = _ScriptedAgentClient(steps_by_question=other_steps)

    _patch_environment(
        monkeypatch,
        gold_rows_by_sql=gold_map,
        agent_rows_by_sql={
            q["expected_sql"]: [(q["id"],)] for q in fixture_questions[1:]
        },
        raise_on_agent_sql={bad_sql: QueryExecutionError("relation does not exist")},
    )

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=_InMemoryMcpClient(tools=[]),
    )
    result = read_json(out)
    bad = next(item for item in result.per_item if item["question"] == bad_question)
    assert bad["execution_success"] is False
    assert bad["pretensor_pass"] is False
    assert "agent SQL failed" in (bad["error"] or "")


def test_runner_marks_pretensor_pass_false_when_rows_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both SQLs run cleanly but rows diverge → pass=False, error explains.

    Parity with the baseline runner's equivalent. This branch fires on
    every imperfect benchmark run, so leaving it untested would hide a
    regression in the equivalence layer.
    """
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}

    differing_question = fixture_questions[0]["question"]
    differing_sql = fixture_questions[0]["expected_sql"]
    agent_map = dict(gold_map)
    # Agent SQL runs but yields different rows for the differing question.
    agent_map[differing_sql] = [("wrong-answer",)]

    client = _gold_only_client(fixture_questions)
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=agent_map
    )

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=_InMemoryMcpClient(tools=[]),
    )
    result = read_json(out)
    bad_item = next(
        item for item in result.per_item if item["question"] == differing_question
    )
    assert bad_item["execution_success"] is True
    assert bad_item["result_equivalence"] is False
    assert bad_item["pretensor_pass"] is False
    assert "row equivalence" in (bad_item["error"] or "")


def test_tool_call_trace_always_includes_is_error_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``is_error`` is present on every trace entry — successes too.

    Pins the trace shape contract so a consumer doing
    ``entry["is_error"]`` never raises ``KeyError`` on a successful
    tool call. The baseline used to omit the key for successes;
    this regression test prevents that from coming back.
    """
    fixture_questions = _load_pagila_questions()
    target = fixture_questions[0]
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    other_steps: dict[str, deque[AgentStep]] = {
        q["question"]: deque([_final_step(q["expected_sql"])])
        for q in fixture_questions[1:]
    }
    other_steps[target["question"]] = deque(
        [
            AgentStep(
                text=None,
                tool_calls=(
                    AgentToolCall(id="ok", name="query", arguments={"q": "x"}),
                    AgentToolCall(id="bad", name="bork", arguments={}),
                ),
                prompt_tokens=10,
                completion_tokens=2,
                latency_ms=5,
            ),
            _final_step(target["expected_sql"]),
        ]
    )
    client = _ScriptedAgentClient(steps_by_question=other_steps)
    mcp = _InMemoryMcpClient(
        tools=[
            AgentTool(name="query", description="", input_schema={}),
            AgentTool(name="bork", description="", input_schema={}),
        ],
        responses={"query": "{}", "bork": "{}"},
        is_error_for={"bork"},
    )

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=mcp,
    )
    result = read_json(out)
    record = next(r for r in result.per_item if r["id"] == target["id"])
    assert len(record["tool_calls"]) == 2
    success_entry, error_entry = record["tool_calls"]
    # Both entries carry the key — no KeyError for the consumer.
    assert success_entry["is_error"] is False
    assert error_entry["is_error"] is True


def test_runner_records_gold_sql_execution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure on the GOLD execution path is per-item, not run-aborting.

    Mirrors the baseline runner's equivalent test so a future reader
    sees the same recovery semantics on both runners. A bug in the
    gold corpus shows up as ``pretensor_pass=False`` with an
    explanatory error; the rest of the questions continue.
    """
    from pretensor.benchmark.l3.db import QueryExecutionError

    fixture_questions = _load_pagila_questions()
    bad_question = fixture_questions[0]["question"]
    bad_gold_sql = fixture_questions[0]["expected_sql"]
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}

    client = _gold_only_client(fixture_questions)
    _patch_environment(
        monkeypatch,
        gold_rows_by_sql=gold_map,
        agent_rows_by_sql=dict(gold_map),
        raise_on_gold_sql={
            bad_gold_sql: QueryExecutionError("simulated gold-corpus bug"),
        },
    )

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=_InMemoryMcpClient(tools=[]),
    )
    result = read_json(out)
    bad_item = next(
        item for item in result.per_item if item["question"] == bad_question
    )
    assert bad_item["execution_success"] is False
    assert bad_item["pretensor_pass"] is False
    assert "gold SQL failed" in (bad_item["error"] or "")
    # Captured BEFORE the gold attempt, so it must round-trip.
    assert bad_item["agent_sql"] == bad_gold_sql

    # Other questions still pass — one bad gold doesn't poison the run.
    expected_success = (len(fixture_questions) - 1) / len(fixture_questions)
    success = result.metrics["nl2sql_success_rate_pretensor"]
    assert success.value == pytest.approx(expected_success)


def test_runner_records_agent_loop_iteration_cap_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AgentLoopError`` from the loop hits the per-item ``error`` field.

    The runner catches :class:`AgentLoopError` separately from
    :class:`LlmCallError`; verify the iteration-cap path lands in the
    same per-item recovery branch and does not abort the run.
    """
    fixture_questions = _load_pagila_questions()
    target = fixture_questions[0]
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    # An infinite stream of tool-call steps for the target question
    # forces the loop to hit its iteration cap. Other questions get
    # the normal final-text step.
    other_steps: dict[str, deque[AgentStep]] = {
        q["question"]: deque([_final_step(q["expected_sql"])])
        for q in fixture_questions[1:]
    }
    forever_step = AgentStep(
        text=None,
        tool_calls=(AgentToolCall(id="loop", name="query", arguments={"q": "spin"}),),
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1,
    )
    # Use a deque that always yields the forever_step — copy it on each pop.
    target_queue: deque[AgentStep] = deque(
        [forever_step] * 32  # well past DEFAULT_MAX_ITERATIONS=16
    )
    other_steps[target["question"]] = target_queue
    client = _ScriptedAgentClient(steps_by_question=other_steps)

    mcp = _InMemoryMcpClient(
        tools=[AgentTool(name="query", description="", input_schema={})],
        responses={"query": "{}"},
    )

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=mcp,
    )
    result = read_json(out)
    bad = next(item for item in result.per_item if item["id"] == target["id"])
    assert bad["pretensor_pass"] is False
    assert bad["execution_success"] is False
    assert bad["agent_sql"] == ""
    assert "agent loop failed" in (bad["error"] or "")
    # Other questions still pass.
    other_passes = sum(
        1
        for item in result.per_item
        if item["id"] != target["id"] and item["pretensor_pass"]
    )
    assert other_passes == len(fixture_questions) - 1


def test_runner_records_empty_agent_sql_short_circuit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitespace-only LLM output is recorded and never reaches the DB.

    Parity with the baseline runner's equivalent test — the empty-SQL
    guard short-circuits before ``execute_query`` so row counts stay 0
    and the per-item ``error`` field explains the failure.
    """
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}

    target_question = fixture_questions[0]["question"]
    other_steps: dict[str, deque[AgentStep]] = {
        q["question"]: deque([_final_step(q["expected_sql"])])
        for q in fixture_questions[1:]
    }
    other_steps[target_question] = deque([_final_step("   \n   ")])
    client = _ScriptedAgentClient(steps_by_question=other_steps)

    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    out = tmp_path / "r.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=client,
        mcp_client=_InMemoryMcpClient(tools=[]),
    )
    result = read_json(out)
    item = next(
        record for record in result.per_item if record["question"] == target_question
    )
    assert item["agent_sql"] == ""
    assert item["execution_success"] is False
    assert item["pretensor_pass"] is False
    assert "empty SQL" in (item["error"] or "")
    assert item["row_count_gold"] == 0
    assert item["row_count_agent"] == 0


def test_runner_refuses_run_when_graph_dir_has_no_kuzu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing indexed graph short-circuits with a clear FileNotFoundError."""
    monkeypatch.setenv("PAGILA_DATABASE_URL", "postgresql://stub/pagila")
    empty_graph_dir = tmp_path / "empty.pretensor"
    empty_graph_dir.mkdir()

    client = _gold_only_client([])
    mcp = _InMemoryMcpClient(tools=[])
    out = tmp_path / "r.json"
    with pytest.raises(FileNotFoundError, match="pretensor index"):
        run_l3_pretensor(
            Dataset.PAGILA,
            out,
            empty_graph_dir,
            model="m",
            seed=1,
            llm_client=client,
            mcp_client=mcp,
        )


def test_runner_raises_lookup_error_when_anthropic_api_key_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing ANTHROPIC_API_KEY → LookupError, friendly CLI message path."""
    monkeypatch.setenv("PAGILA_DATABASE_URL", "postgresql://stub/pagila")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "r.json"
    with pytest.raises(LookupError, match="ANTHROPIC_API_KEY"):
        run_l3_pretensor(
            Dataset.PAGILA,
            out,
            _seed_indexed_graph_dir(tmp_path),
            model="claude-haiku-4-5",
            seed=1,
            llm_client=None,
            mcp_client=_InMemoryMcpClient(tools=[]),
        )


def test_runner_raises_when_database_url_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PAGILA_DATABASE_URL", raising=False)
    out = tmp_path / "r.json"
    with pytest.raises(LookupError, match="PAGILA_DATABASE_URL"):
        run_l3_pretensor(
            Dataset.PAGILA,
            out,
            _seed_indexed_graph_dir(tmp_path),
            model="m",
            seed=1,
            llm_client=_gold_only_client([]),
            mcp_client=_InMemoryMcpClient(tools=[]),
        )


# ---------------------------------------------------------------------------
# determinism + atomic write
# ---------------------------------------------------------------------------


def test_runner_two_runs_with_same_seed_and_fakes_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #5 verified under deterministic LLM + MCP stand-ins.

    Wall-clock latency (``per_item.latency_ms``, ``mean_latency_ms_pretensor``)
    is the only intentionally non-deterministic field. We pin ``_now`` to a
    constant so latency_ms is always 0 → byte-stable output. Anything else
    that flips between runs surfaces as a real determinism bug.
    """
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )
    # Pin the clock so latency_ms is always 0 → byte-stable across runs.
    monkeypatch.setattr(runner_mod, "_now", lambda: 0.0)

    graph_dir = _seed_indexed_graph_dir(tmp_path)
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        a,
        graph_dir,
        model="m",
        seed=42,
        llm_client=_gold_only_client(fixture_questions),
        mcp_client=_InMemoryMcpClient(tools=[]),
    )
    run_l3_pretensor(
        Dataset.PAGILA,
        b,
        graph_dir,
        model="m",
        seed=42,
        llm_client=_gold_only_client(fixture_questions),
        mcp_client=_InMemoryMcpClient(tools=[]),
    )
    assert a.read_bytes() == b.read_bytes()


def test_runner_writes_to_stdout_when_out_is_none(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--out`` omitted → JSON document goes to stdout."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    run_l3_pretensor(
        Dataset.PAGILA,
        None,
        _seed_indexed_graph_dir(tmp_path),
        model="m",
        seed=1,
        llm_client=_gold_only_client(fixture_questions),
        mcp_client=_InMemoryMcpClient(tools=[]),
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["level"] == "l3"
    assert payload["extra"]["runner"] == "pretensor"
