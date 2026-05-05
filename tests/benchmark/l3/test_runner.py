"""Unit tests for ``run_l3_baseline``.

A ``FakeLlmClient`` returns canned SQL keyed off the user-message question
text; ``execute_query`` is monkeypatched so the tests do not require a live
PostgreSQL instance. The end-to-end path against a real DB is exercised by
``tests/e2e/test_l3_baseline.py`` (gated on PRETENSOR_E2E=1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import pretensor.benchmark.l3.runner as runner_mod
from pretensor.benchmark.l3.llm_client import LlmCallError, LlmResponse
from pretensor.benchmark.l3.runner import run_l3_baseline
from pretensor.benchmark.results import read_json
from pretensor.benchmark.runner import Dataset

# ---------------------------------------------------------------------------
# fake LLM client + helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeLlmClient:
    """Deterministic stand-in for an LLM provider.

    ``responses`` maps the user-message text (the gold ``question``) to the
    canned SQL the fake client should emit. ``raises`` is a separate map
    that lets tests force per-question :class:`LlmCallError` for the
    error-handling path.
    """

    responses: dict[str, str]
    raises: dict[str, Exception]

    def complete(
        self, *, system: str, user: str, model: str, temperature: float
    ) -> LlmResponse:
        if user in self.raises:
            raise self.raises[user]
        text = self.responses.get(user, "SELECT 1")
        return LlmResponse(
            text=text, prompt_tokens=10, completion_tokens=5, latency_ms=42
        )


def _patch_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gold_rows_by_sql: dict[str, list[tuple[Any, ...]]] | None = None,
    agent_rows_by_sql: dict[str, list[tuple[Any, ...]]] | None = None,
    raise_on_agent_sql: dict[str, Exception] | None = None,
    raise_on_gold_sql: dict[str, Exception] | None = None,
) -> None:
    """Stub ``execute_query`` and the per-dataset DB env var.

    ``raise_on_gold_sql`` and ``raise_on_agent_sql`` let tests inject
    QueryExecutionError on either branch independently, so the runner's
    "gold failed" and "agent failed" recovery paths can be exercised in
    isolation.
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
            rows = agent_map.get(sql, [])
        else:
            if sql in gold_raises:
                raise gold_raises[sql]
            rows = gold_map.get(sql, [])
        return list(rows), []

    monkeypatch.setattr(runner_mod, "execute_query", fake_execute_query)


# ---------------------------------------------------------------------------
# happy-path: pretend the agent always returns gold SQL → 100% pass rate
# ---------------------------------------------------------------------------


def test_runner_emits_required_envelope_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All AC-required envelope keys are present on a successful run."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [("answer",)] for q in fixture_questions}
    agent_map = dict(gold_map)

    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})

    _patch_environment(
        monkeypatch,
        gold_rows_by_sql=gold_map,
        agent_rows_by_sql=agent_map,
    )

    out = tmp_path / "result.json"
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
    assert result.embeddings_enabled is False
    assert "nl2sql_success_rate_baseline" in result.metrics
    assert "mean_latency_ms_baseline" in result.metrics
    success_metric = result.metrics["nl2sql_success_rate_baseline"]
    assert success_metric.direction == "higher_is_better"
    assert success_metric.value == 1.0
    latency_metric = result.metrics["mean_latency_ms_baseline"]
    assert latency_metric.direction == "lower_is_better"

    assert result.extra["runner"] == "baseline"
    assert result.extra["model"] == "claude-haiku-4-5"
    assert result.extra["temperature"] == 0.0
    assert result.extra["seed"] == 42
    assert isinstance(result.extra["prompt_template_hash"], str)
    assert isinstance(result.extra["prompt_hash"], str)


def test_per_item_records_have_all_ac_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-item shape: id, question, gold_sql, agent_sql, execution_success,
    result_equivalence, latency_ms, baseline_pass, error."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    agent_map = dict(gold_map)

    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=agent_map
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    result = read_json(out)
    required = {
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
    for record in result.per_item:
        missing = required - set(record.keys())
        assert not missing, f"missing AC-required fields {missing}"


def test_per_item_sorted_by_id_for_byte_stable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    result = read_json(out)
    ids = [item["id"] for item in result.per_item]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_runner_records_llm_call_failure_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single LLM failure must not abort the run; it shows up per-item."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    agent_map = dict(gold_map)

    failing_question = fixture_questions[0]["question"]
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(
        responses=fake_responses,
        raises={failing_question: LlmCallError("simulated 500")},
    )

    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=agent_map
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    result = read_json(out)

    failed_item = next(
        item for item in result.per_item if item["question"] == failing_question
    )
    assert failed_item["baseline_pass"] is False
    assert failed_item["execution_success"] is False
    assert failed_item["agent_sql"] == ""
    assert failed_item["error"] is not None
    assert "LLM" in failed_item["error"]

    success_metric = result.metrics["nl2sql_success_rate_baseline"]
    assert success_metric.value is not None
    expected_success = (len(fixture_questions) - 1) / len(fixture_questions)
    assert success_metric.value == pytest.approx(expected_success)


def test_runner_records_agent_sql_execution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SQL execution error on the agent path is captured per-item."""
    from pretensor.benchmark.l3.db import QueryExecutionError

    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}

    bad_question = fixture_questions[0]["question"]
    bad_sql = "SELECT bogus FROM nowhere"
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    fake_responses[bad_question] = bad_sql
    client = FakeLlmClient(responses=fake_responses, raises={})

    _patch_environment(
        monkeypatch,
        gold_rows_by_sql=gold_map,
        agent_rows_by_sql={
            q["expected_sql"]: [(q["id"],)] for q in fixture_questions[1:]
        },
        raise_on_agent_sql={bad_sql: QueryExecutionError("relation does not exist")},
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    result = read_json(out)
    bad_item = next(
        item for item in result.per_item if item["question"] == bad_question
    )
    assert bad_item["execution_success"] is False
    assert bad_item["baseline_pass"] is False
    assert "agent SQL failed" in (bad_item["error"] or "")
    assert bad_item["agent_sql"] == bad_sql


def test_runner_records_gold_sql_execution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure on the GOLD execution path is recorded per-item, not crashing.

    A bug in the gold corpus (the human-authored ``expected_sql``) must
    not abort the whole run — it shows up as ``baseline_pass=False`` with
    an explanatory error, while the rest of the questions continue.
    """
    from pretensor.benchmark.l3.db import QueryExecutionError

    fixture_questions = _load_pagila_questions()
    bad_question = fixture_questions[0]["question"]
    bad_gold_sql = fixture_questions[0]["expected_sql"]
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    agent_map = dict(gold_map)

    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})

    _patch_environment(
        monkeypatch,
        gold_rows_by_sql=gold_map,
        agent_rows_by_sql=agent_map,
        raise_on_gold_sql={
            bad_gold_sql: QueryExecutionError("simulated gold-corpus bug"),
        },
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    result = read_json(out)
    bad_item = next(
        item for item in result.per_item if item["question"] == bad_question
    )
    assert bad_item["execution_success"] is False
    assert bad_item["baseline_pass"] is False
    assert "gold SQL failed" in (bad_item["error"] or "")
    # The agent's SQL was captured before gold was attempted, so it MUST
    # appear on the per-item record — that's what makes a gold bug
    # auditable rather than invisible.
    assert bad_item["agent_sql"] == bad_gold_sql

    # Other questions still pass — one bad gold doesn't poison the run.
    expected_success = (len(fixture_questions) - 1) / len(fixture_questions)
    success_metric = result.metrics["nl2sql_success_rate_baseline"]
    assert success_metric.value == pytest.approx(expected_success)


def test_runner_records_empty_agent_sql_short_circuit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty SQL string from the agent is recorded and does NOT touch the DB.

    The runner short-circuits before calling ``execute_query`` when the
    agent returned only whitespace; per-item records the empty agent_sql
    and the explanatory error.
    """
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    agent_map = dict(gold_map)

    empty_question = fixture_questions[0]["question"]
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    fake_responses[empty_question] = "   \n  "  # whitespace-only response
    client = FakeLlmClient(responses=fake_responses, raises={})
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=agent_map
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    result = read_json(out)
    item = next(
        record for record in result.per_item if record["question"] == empty_question
    )
    assert item["agent_sql"] == ""
    assert item["execution_success"] is False
    assert item["baseline_pass"] is False
    assert "empty SQL" in (item["error"] or "")
    # Crucially: row counts stay 0 since neither side was executed.
    assert item["row_count_gold"] == 0
    assert item["row_count_agent"] == 0


def test_runner_per_item_carries_llm_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LlmResponse latency / token counts make it into per_item records."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    result = read_json(out)
    sample = result.per_item[0]
    # FakeLlmClient sets these fixed values; runner must propagate.
    assert sample["llm_latency_ms"] == 42
    assert sample["prompt_tokens"] == 10
    assert sample["completion_tokens"] == 5


def test_runner_marks_pass_false_when_rows_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQL ran but rows don't match gold → pass=False, error explains."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}

    differing_question = fixture_questions[0]["question"]
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})

    agent_map = dict(gold_map)
    # Agent returns DIFFERENT rows for the differing question
    differing_sql = fixture_questions[0]["expected_sql"]
    agent_map[differing_sql] = [("wrong-answer",)]

    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=agent_map
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    result = read_json(out)
    bad_item = next(
        item for item in result.per_item if item["question"] == differing_question
    )
    assert bad_item["execution_success"] is True
    assert bad_item["result_equivalence"] is False
    assert bad_item["baseline_pass"] is False
    assert "row equivalence" in (bad_item["error"] or "")


# ---------------------------------------------------------------------------
# seed / determinism
# ---------------------------------------------------------------------------


def test_runner_records_a_seed_when_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=None, llm_client=client)
    result = read_json(out)
    assert isinstance(result.extra["seed"], int)
    assert result.extra["seed"] >= 0


def test_runner_two_runs_with_same_seed_and_fake_client_are_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #5 verified under deterministic LLM stand-in."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    run_l3_baseline(Dataset.PAGILA, a, model="m", seed=42, llm_client=client)
    run_l3_baseline(Dataset.PAGILA, b, model="m", seed=42, llm_client=client)
    assert a.read_bytes() == b.read_bytes()


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------


def test_runner_emits_seed_disclaimer_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec/AC tolerance documentation must be on the JSON envelope."""
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    result = read_json(out)
    assert any("seed" in note.lower() for note in result.notes)


# ---------------------------------------------------------------------------
# missing fixture inputs
# ---------------------------------------------------------------------------


def test_runner_raises_when_dataset_has_no_ddl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic datasets without a DDL bundle must error clearly."""
    monkeypatch.setenv("PAGILA_DATABASE_URL", "postgresql://stub/p")
    out = tmp_path / "r.json"
    with pytest.raises(FileNotFoundError, match="DDL bundle"):
        run_l3_baseline(
            Dataset.ANALYTICS_DWH,
            out,
            model="m",
            seed=1,
            llm_client=FakeLlmClient(responses={}, raises={}),
        )


def test_runner_raises_when_database_url_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing per-dataset env var produces a clear LookupError."""
    monkeypatch.delenv("PAGILA_DATABASE_URL", raising=False)
    out = tmp_path / "r.json"
    with pytest.raises(LookupError, match="PAGILA_DATABASE_URL"):
        run_l3_baseline(
            Dataset.PAGILA,
            out,
            model="m",
            seed=1,
            llm_client=FakeLlmClient(responses={}, raises={}),
        )


def test_runner_raises_lookup_error_when_anthropic_api_key_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing ANTHROPIC_API_KEY surfaces as LookupError, not a raw traceback.

    The default AnthropicHttpClient constructor raises ``LlmCallError``
    when the env var is unset. The runner converts that to
    ``LookupError`` so the CLI's missing-env-var handler treats it the
    same as missing ``PAGILA_DATABASE_URL`` (friendly exit 1, no
    traceback). Tests pass ``llm_client=None`` to exercise the default
    construction path.
    """
    monkeypatch.setenv("PAGILA_DATABASE_URL", "postgresql://stub/pagila")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "r.json"
    with pytest.raises(LookupError, match="ANTHROPIC_API_KEY"):
        run_l3_baseline(
            Dataset.PAGILA,
            out,
            model="claude-haiku-4-5",
            seed=1,
            llm_client=None,
        )


# ---------------------------------------------------------------------------
# atomic write + stdout fallback
# ---------------------------------------------------------------------------


def test_runner_writes_to_stdout_when_out_is_none(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    run_l3_baseline(Dataset.PAGILA, None, model="m", seed=1, llm_client=client)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["level"] == "l3"


def test_runner_atomic_write_no_temp_file_left_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_questions = _load_pagila_questions()
    gold_map = {q["expected_sql"]: [(q["id"],)] for q in fixture_questions}
    fake_responses = {q["question"]: q["expected_sql"] for q in fixture_questions}
    client = FakeLlmClient(responses=fake_responses, raises={})
    _patch_environment(
        monkeypatch, gold_rows_by_sql=gold_map, agent_rows_by_sql=dict(gold_map)
    )

    out = tmp_path / "r.json"
    run_l3_baseline(Dataset.PAGILA, out, model="m", seed=1, llm_client=client)
    siblings = sorted(p.name for p in tmp_path.iterdir())
    assert siblings == ["r.json"], siblings


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_pagila_questions() -> list[dict[str, Any]]:
    """Load the real Pagila question set so tests pin the AC contract."""
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "scripts" / "data" / "pagila_nl2sql_bench.json"
    return json.loads(path.read_text(encoding="utf-8"))
