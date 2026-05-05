"""End-to-end test for the L3 baseline runner against a real Pagila DB.

Gated by ``PRETENSOR_E2E=1`` (the conftest skips the whole module when
unset). Spawns Postgres via testcontainers, applies the Pagila DDL, points
the runner's per-dataset DSN env var at the container, and uses a
deterministic ``FakeLlmClient`` so the test does not require an LLM API
key. The point is to verify end-to-end SQL execution + row equivalence
against a real database, not LLM quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from pretensor.benchmark.l3 import run_l3_baseline
from pretensor.benchmark.l3.llm_client import LlmResponse
from pretensor.benchmark.results import read_json
from pretensor.benchmark.runner import Dataset

# Three hand-curated Pagila gold questions whose expected_sql is known to
# run cleanly against the e2e Pagila DDL. Mirroring them exactly in the
# fake client guarantees a 100% pass rate, isolating "did the runner
# wire correctly" from "did the agent succeed."
_E2E_QUESTIONS = (
    "What is the sum of all payment amounts?",
    "List each staff member's first and last name with their store id.",
    "How many customers have never rented a film?",
)


@dataclass
class _GoldEchoClient:
    """Fake LLM client that echoes the gold SQL for the questions we cover."""

    responses: dict[str, str]

    def complete(
        self, *, system: str, user: str, model: str, temperature: float
    ) -> LlmResponse:
        # Unknown questions get a syntactically valid no-op so they fail
        # equivalence rather than crashing the run — auditable failure mode.
        text = self.responses.get(user, "SELECT 0 WHERE FALSE")
        return LlmResponse(
            text=text, prompt_tokens=20, completion_tokens=10, latency_ms=15
        )


def _load_pagila_gold(repo_root: Path) -> dict[str, str]:
    path = repo_root / "scripts" / "data" / "pagila_nl2sql_bench.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {entry["question"]: entry["expected_sql"] for entry in raw}


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_l3_baseline_end_to_end_against_pagila(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pagila_dsn: str,
    repo_root: Path,
) -> None:
    """Runner exits cleanly, writes valid JSON, and reports per-question results.

    Verifies AC #1, #2, #3, #4, #6 against a live Postgres instance:
    * AC #1 — "runs end-to-end against a local Pagila instance and exits 0".
    * AC #2 — JSON parses and matches the BenchmarkResult schema.
    * AC #3 — per_item entries carry id / question / gold_sql / agent_sql /
      execution_success / result_equivalence / latency_ms / baseline_pass.
    * AC #4 — model / temperature / seed / prompt_hash live in the envelope's
      ``extra`` block.
    * AC #6 — runner code under ``src/pretensor/benchmark/l3/`` reaches a
      real DB through the env-var-resolved DSN.
    """
    monkeypatch.setenv("PAGILA_DATABASE_URL", pagila_dsn)
    gold_by_question = _load_pagila_gold(repo_root)
    client = _GoldEchoClient(responses={q: gold_by_question[q] for q in _E2E_QUESTIONS})

    out = tmp_path / "l3-baseline.json"
    run_l3_baseline(
        Dataset.PAGILA,
        out,
        model="claude-haiku-4-5",
        seed=42,
        llm_client=client,
    )

    result = read_json(out)
    assert result.level == "l3"
    assert result.dataset == "pagila"

    # Envelope (AC #4)
    assert result.extra["runner"] == "baseline"
    assert result.extra["model"] == "claude-haiku-4-5"
    assert result.extra["temperature"] == 0.0
    assert result.extra["seed"] == 42
    assert isinstance(result.extra["prompt_hash"], str)
    assert isinstance(result.extra["prompt_template_hash"], str)

    # Metric shape (AC #2)
    success = result.metrics["nl2sql_success_rate_baseline"]
    assert success.direction == "higher_is_better"
    assert success.value is not None
    assert 0.0 <= success.value <= 1.0
    latency = result.metrics["mean_latency_ms_baseline"]
    assert latency.direction == "lower_is_better"
    assert latency.value is not None

    # per_item shape (AC #3)
    required_keys = {
        "id",
        "question",
        "gold_sql",
        "agent_sql",
        "execution_success",
        "result_equivalence",
        "latency_ms",
        "baseline_pass",
        "error",
    }
    assert result.per_item, "expected at least one per-item entry"
    for record in result.per_item:
        assert required_keys <= set(record.keys()), (
            f"missing AC #3 fields: {required_keys - set(record.keys())}"
        )

    # The three covered questions must be passes (gold echo against
    # the same DB). The others are intentional non-passes to keep the
    # success rate auditable rather than 1.0 by construction.
    covered = {q for q in _E2E_QUESTIONS}
    for record in result.per_item:
        if record["question"] in covered:
            assert record["baseline_pass"] is True, (
                f"covered question {record['id']} failed: {record.get('error')}"
            )
            assert record["execution_success"] is True
            assert record["result_equivalence"] is True


def test_l3_baseline_atomic_write_no_temp_file_left_over(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pagila_dsn: str,
    repo_root: Path,
) -> None:
    monkeypatch.setenv("PAGILA_DATABASE_URL", pagila_dsn)
    gold_by_question = _load_pagila_gold(repo_root)
    client = _GoldEchoClient(responses={q: gold_by_question[q] for q in _E2E_QUESTIONS})

    out = tmp_path / "result.json"
    run_l3_baseline(
        Dataset.PAGILA, out, model="claude-haiku-4-5", seed=7, llm_client=client
    )
    siblings = sorted(p.name for p in tmp_path.iterdir())
    assert siblings == ["result.json"], siblings
