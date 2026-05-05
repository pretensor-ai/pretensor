"""End-to-end test for the L3 pretensor runner.

Gated by ``PRETENSOR_E2E=1`` (the conftest skips the whole module when
unset). Uses the real ``pretensor serve`` subprocess against an
already-indexed Pagila graph dir (the ``indexed_state`` fixture), and
a deterministic ``_GoldEchoAgentClient`` so the test does not require
an LLM API key. The point is to verify:

* the runner spawns and reaps ``pretensor serve`` (AC #6);
* MCP tool discovery + invocation work over real stdio (AC #1);
* per-question records carry tool-call traces (AC #3, #4);
* the JSON envelope conforms to the L3 schema (AC #2, #7).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pretensor.benchmark.l3 import run_l3_pretensor
from pretensor.benchmark.l3.agent import (
    AgentMessage,
    AgentStep,
    AgentTool,
    AgentToolCall,
)
from pretensor.benchmark.results import read_json
from pretensor.benchmark.runner import Dataset

# Three Pagila gold questions whose expected_sql runs cleanly against the
# e2e Pagila DDL. We keep the LLM dumb here — it always emits the gold
# SQL after a single MCP tool call. That keeps the run focused on
# infrastructure (real subprocess + real DB) rather than agent quality.
_E2E_QUESTIONS = (
    "What is the sum of all payment amounts?",
    "List each staff member's first and last name with their store id.",
    "How many customers have never rented a film?",
)


_TOOL_PLAN_PER_QUESTION: dict[str, tuple[tuple[str, dict[str, object]], ...]] = {
    "What is the sum of all payment amounts?": (("schema", {}),),
    "List each staff member's first and last name with their store id.": (
        ("query", {"q": "staff"}),
        ("context", {"table": "staff"}),
    ),
    "How many customers have never rented a film?": (
        ("traverse", {"from_table": "customer", "to_table": "rental"}),
    ),
}
"""Per-question tool plan exercised by ``_GoldEchoAgentClient``.

The three tools the AC #4 acceptance criterion calls out (``query``,
``context``, ``traverse``) are split across the gold questions so a
single e2e run produces a per-item trace that *as a set* contains
all of them. ``schema`` covers the no-arg path. Each tool here is one
the OSS pretensor server registers unconditionally — picking
embeddings-only tools (e.g. ``semantic_search``) would couple the
test to which optional extras are installed at run time.
"""


@dataclass
class _GoldEchoAgentClient:
    """Tool-using LLM stand-in: scripted MCP calls, then the gold SQL.

    For each covered question, the script is two steps: a tool-call
    step (per :data:`_TOOL_PLAN_PER_QUESTION`) followed by a final
    text step with the gold SQL. Uncovered questions get a single
    ``SELECT 0 WHERE FALSE`` step so they still fail equivalence
    cleanly rather than crashing.
    """

    responses: dict[str, str]
    _seq_by_question: dict[str, deque[AgentStep]] = field(default_factory=dict)

    def _steps_for(self, question: str) -> deque[AgentStep]:
        if question not in self._seq_by_question:
            sql = self.responses.get(question, "SELECT 0 WHERE FALSE")
            plan = _TOOL_PLAN_PER_QUESTION.get(question, (("schema", {}),))
            base_id = abs(hash(question)) % 10_000
            tool_calls = tuple(
                AgentToolCall(
                    id=f"toolu_{base_id:04d}_{i}",
                    name=name,
                    arguments=dict(args),
                )
                for i, (name, args) in enumerate(plan)
            )
            self._seq_by_question[question] = deque(
                [
                    AgentStep(
                        text=None,
                        tool_calls=tool_calls,
                        prompt_tokens=20,
                        completion_tokens=8,
                        latency_ms=12,
                    ),
                    AgentStep(
                        text=sql,
                        tool_calls=(),
                        prompt_tokens=15,
                        completion_tokens=6,
                        latency_ms=12,
                    ),
                ]
            )
        return self._seq_by_question[question]

    def agent_complete(
        self,
        *,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        model: str,
        temperature: float,
    ) -> AgentStep:
        if not messages or messages[0].text is None:
            raise AssertionError("loop did not lead with the user prompt")
        return self._steps_for(messages[0].text).popleft()


def _load_pagila_gold(repo_root: Path) -> dict[str, str]:
    path = repo_root / "scripts" / "data" / "pagila_nl2sql_bench.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {entry["question"]: entry["expected_sql"] for entry in raw}


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_l3_pretensor_end_to_end_against_pagila(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pagila_dsn: str,
    indexed_state: Path,
    repo_root: Path,
) -> None:
    """Runner spawns ``pretensor serve``, drives one tool call, writes valid JSON.

    Verifies AC #1, #2, #3, #4, #6 against a live Postgres + a real MCP
    subprocess:
    * AC #1 — runs end-to-end and exits cleanly.
    * AC #2 — JSON parses and matches the BenchmarkResult schema with
      pretensor-suffixed metric names.
    * AC #3 — per_item entries carry id / question / gold_sql / agent_sql /
      tool_calls / execution_success / result_equivalence / latency_ms /
      pretensor_pass.
    * AC #4 — model / temperature / seed / prompt_hash / tool_catalogue
      live in the envelope's ``extra`` block.
    * AC #6 — the runner code reaches a real DB through the env-var-resolved
      DSN AND a real MCP subprocess through stdio.
    """
    if shutil.which("pretensor") is None:
        pytest.skip("`pretensor` CLI not on PATH — `uv sync` first")

    monkeypatch.setenv("PAGILA_DATABASE_URL", pagila_dsn)
    gold_by_question = _load_pagila_gold(repo_root)
    client = _GoldEchoAgentClient(
        responses={q: gold_by_question[q] for q in _E2E_QUESTIONS}
    )

    out = tmp_path / "l3-pretensor.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        indexed_state,
        model="claude-haiku-4-5",
        seed=42,
        llm_client=client,
        # mcp_client=None → spawns real `pretensor serve` against indexed_state.
    )

    result = read_json(out)
    assert result.level == "l3"
    assert result.dataset == "pagila"
    assert isinstance(result.embeddings_enabled, bool)

    # Envelope shape (AC #4)
    assert result.extra["runner"] == "pretensor"
    assert result.extra["model"] == "claude-haiku-4-5"
    assert result.extra["temperature"] == 0.0
    assert result.extra["seed"] == 42
    assert isinstance(result.extra["prompt_hash"], str)
    assert isinstance(result.extra["prompt_template_hash"], str)
    assert isinstance(result.extra["tool_catalogue"], list)
    # AC #4: the served catalogue MUST expose at least query / context /
    # traverse / schema (the OSS pretensor server registers these
    # unconditionally). Asserting the catalogue contents directly is the
    # half of AC #4 the runner can verify on its own — what the agent
    # then chooses to invoke is LLM-bound and not asserted here.
    catalogue = set(result.extra["tool_catalogue"])
    assert {"query", "context", "traverse", "schema"} <= catalogue, catalogue

    # Metric shape (AC #2)
    success = result.metrics["nl2sql_success_rate_pretensor"]
    assert success.direction == "higher_is_better"
    assert success.value is not None
    assert 0.0 <= success.value <= 1.0
    latency = result.metrics["mean_latency_ms_pretensor"]
    assert latency.direction == "lower_is_better"
    assert latency.value is not None

    # per_item shape (AC #3)
    required_keys = {
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
    assert result.per_item, "expected at least one per-item entry"
    for record in result.per_item:
        missing = required_keys - set(record.keys())
        assert not missing, f"missing AC #3 fields: {missing}"

    # Each covered question must pass equivalence and exercise at
    # least one tool call.
    covered = set(_E2E_QUESTIONS)
    observed_tools: set[str] = set()
    for record in result.per_item:
        if record["question"] in covered:
            assert record["pretensor_pass"] is True, (
                f"covered question {record['id']} failed: {record.get('error')}"
            )
            assert record["execution_success"] is True
            assert len(record["tool_calls"]) >= 1, (
                f"covered question {record['id']} had no tool calls: {record}"
            )
            observed_tools.update(call["tool"] for call in record["tool_calls"])

    # AC #4: across the per-item traces (taken as a set) the agent
    # must have actually invoked query, context, AND traverse against
    # the live MCP subprocess. Catalogue membership alone is necessary
    # but not sufficient — this assertion is what proves the trace
    # carries those tools, not just the catalogue.
    assert {"query", "context", "traverse", "schema"} <= observed_tools, observed_tools


def test_l3_pretensor_atomic_write_no_temp_file_left_over(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pagila_dsn: str,
    indexed_state: Path,
    repo_root: Path,
) -> None:
    """``--out`` writes the final JSON with no leftover ``.tmp`` sibling.

    Plus: cleanly exiting after a successful run leaves no orphaned
    ``pretensor serve`` subprocess (AC #6) — the SDK's stdio_client
    context manager reaps it.
    """
    if shutil.which("pretensor") is None:
        pytest.skip("`pretensor` CLI not on PATH — `uv sync` first")

    monkeypatch.setenv("PAGILA_DATABASE_URL", pagila_dsn)
    gold_by_question = _load_pagila_gold(repo_root)
    client = _GoldEchoAgentClient(
        responses={q: gold_by_question[q] for q in _E2E_QUESTIONS}
    )

    out = tmp_path / "result.json"
    run_l3_pretensor(
        Dataset.PAGILA,
        out,
        indexed_state,
        model="claude-haiku-4-5",
        seed=7,
        llm_client=client,
    )
    siblings = sorted(p.name for p in tmp_path.iterdir())
    assert siblings == ["result.json"], siblings

    # AC #6: subprocess reaping must work in practice, not just in
    # theory. The runner is the only thing that should have spawned a
    # ``pretensor serve`` in this test process, so by now there must
    # be no surviving child of pid==os-current with that command line.
    leaked = subprocess.run(
        ["pgrep", "-f", "pretensor serve"],
        capture_output=True,
        text=True,
        check=False,
    )
    if leaked.returncode == 0:
        pytest.fail(
            f"pretensor serve subprocess(es) survived run: PIDs {leaked.stdout.strip()!r}"
        )
