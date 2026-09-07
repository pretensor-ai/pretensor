"""``run_l3_baseline`` — the agent + raw schema control runner.

Walks the gold question set for a dataset, asks the configured LLM to
translate each question into PostgreSQL, executes both the gold and the
agent SQL against the live database, and grades equivalence row-by-row.
Emits a :class:`BenchmarkResult` envelope whose ``extra`` block captures
the full agent configuration so a future re-run can be reproduced
byte-for-byte modulo LLM nondeterminism.

Atomic JSON write: the runner writes to ``<out>.tmp`` first and renames,
so a process kill mid-run never leaves a half-written file behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, cast

from pretensor.benchmark.fixtures import load_dataset
from pretensor.benchmark.l3.db import (
    QueryExecutionError,
    execute_query,
    resolve_database_url,
)
from pretensor.benchmark.l3.gold import L3GoldEntry, load_l3_gold
from pretensor.benchmark.l3.llm_client import (
    AnthropicHttpClient,
    LlmCallError,
    LlmClient,
    OpenAIHttpClient,
)
from pretensor.benchmark.l3.prompt import (
    MAX_DDL_CHARS,
    build_system_prompt,
    prompt_hash,
    prompt_template_hash,
    strip_sql_fences,
)
from pretensor.benchmark.l3.sql_equivalence import gold_is_ordered, rows_equivalent
from pretensor.benchmark.results import BenchmarkResult, Metric, write_json
from pretensor.benchmark.runner import Dataset
from pretensor.version import package_version

__all__ = ["DEFAULT_TEMPERATURE", "run_l3_baseline"]


DEFAULT_TEMPERATURE = 0.0
"""LLM temperature for the baseline runner.

Temp=0 minimises variance; the spec accepts that some LLM
nondeterminism remains and tracks it via the seed envelope field
plus a documented ``notes[]`` entry.
"""

_DETERMINISTIC_RAN_AT = "1970-01-01T00:00:00Z"
"""Pinned timestamp so identical inputs produce byte-identical envelopes.

Mirrors the ``_DETERMINISTIC_RAN_AT`` convention used in the L1 / L2
runners; real wall-clock would defeat reproducibility comparisons.
Module-private — callers have no reason to depend on this value.
"""

_now = time.perf_counter
"""Wall-clock source for per-item latency measurement.

Module-private so tests can monkeypatch it to a deterministic stub
(``lambda: 0.0``) without touching production behaviour — mirrors the
``_DETERMINISTIC_RAN_AT`` pattern used for the run timestamp.
"""

_AGENT_ERROR_MAX_CHARS = 500
"""Truncation limit for per-item ``error`` strings.

Database errors and LLM error responses can be huge; the auditor only
needs the leading edge to diagnose.
"""

_SUCCESS_RATE_METRIC = "nl2sql_success_rate_baseline"
_LATENCY_METRIC = "mean_latency_ms_baseline"


def run_l3_baseline(
    dataset: Dataset,
    out: Path | None,
    *,
    model: str,
    seed: int | None,
    llm_client: LlmClient | None = None,
) -> None:
    """Run the L3 baseline (agent + raw schema, no Pretensor) for ``dataset``.

    ``llm_client`` is a Python-API-only seam — the CLI never exposes it.
    Tests pass a fake client to avoid hitting a real provider; production
    callers leave it ``None`` and the runner builds an
    :class:`AnthropicHttpClient` against ``ANTHROPIC_API_KEY``.
    """
    fixture = load_dataset(dataset)
    if fixture.ddl_sql_path is None:
        raise FileNotFoundError(
            f"Dataset {dataset.value!r} has no DDL bundle "
            f"(scripts/data/{dataset.value}/schema.sql); cannot build the L3 prompt."
        )

    ddl_text = fixture.ddl_sql_path.read_text(encoding="utf-8")
    # ``load_l3_gold`` raises FileNotFoundError when ``questions_path`` is
    # absent and returns the resolved path + raw bytes explicitly. Reusing
    # the bytes for the fixture_sha avoids a second file read; relying on
    # the path lets us drop a narrowing ``assert`` (which would be
    # stripped under ``python -O``).
    _questions_path, questions_bytes, questions = load_l3_gold(fixture)
    dsn = resolve_database_url(dataset)

    resolved_seed = seed if seed is not None else secrets.randbits(63)

    system_prompt = build_system_prompt(dataset.value, ddl_text)
    rendered_hash = prompt_hash(system_prompt)
    template_hash = prompt_template_hash()

    client: LlmClient
    if llm_client is not None:
        client = llm_client
    else:
        # ``AnthropicHttpClient()`` raises ``LlmCallError`` when
        # ``ANTHROPIC_API_KEY`` is unset. Re-raise as ``LookupError`` so
        # the CLI's missing-env-var handler treats this the same as
        # missing ``PAGILA_DATABASE_URL`` and prints a friendly exit-1
        # message rather than a Python traceback.
        try:
            client = AnthropicHttpClient()
        except LlmCallError as exc:
            raise LookupError(str(exc)) from exc

    notes: list[str] = [
        "L3 is LLM-bound; the --seed value governs harness state only "
        "(the Anthropic / OpenAI APIs do not accept a seed parameter). "
        "Same-seed reruns at temperature 0 may still differ by remaining "
        "LLM nondeterminism."
    ]
    if len(ddl_text) > MAX_DDL_CHARS:
        notes.append(
            f"DDL for {dataset.value} is {len(ddl_text)} characters "
            f"(> soft limit {MAX_DDL_CHARS}); some models may truncate the "
            "schema at their context-window edge. The runner did NOT "
            "truncate the DDL — agent failures may reflect this rather "
            "than agent quality."
        )

    per_item: list[dict[str, Any]] = []
    latencies_ms: list[int] = []
    successes = 0

    for entry in questions:
        record = _evaluate_one(
            entry=entry,
            system_prompt=system_prompt,
            client=client,
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            dsn=dsn,
        )
        per_item.append(record)
        latencies_ms.append(int(record["latency_ms"]))
        if record["baseline_pass"]:
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
        "runner": "baseline",
        "provider": _provider_name(client),
        "model": model,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": resolved_seed,
        "prompt_template_hash": template_hash,
        "prompt_hash": rendered_hash,
    }

    result = BenchmarkResult(
        level="l3",
        dataset=dataset.value,
        pretensor_version=package_version(fallback="0.0.0+unknown"),
        embeddings_enabled=False,
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
    client: LlmClient,
    model: str,
    temperature: float,
    dsn: str,
) -> dict[str, Any]:
    """Grade one question end-to-end. Returns the per-item record.

    Wraps :func:`_populate_record` in a try/finally so ``latency_ms`` is
    always set to the full per-question wall-clock — including SQL
    execution and equivalence checking, not just the LLM call. The
    LLM-only timing the provider client measured is recorded separately
    as ``llm_latency_ms`` for cost / regression diagnostics.
    """
    record: dict[str, Any] = {
        "id": entry.id,
        "question": entry.question,
        "gold_sql": entry.expected_sql,
        "agent_sql": "",
        "execution_success": False,
        "result_equivalence": False,
        "latency_ms": 0,
        "llm_latency_ms": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "row_count_gold": 0,
        "row_count_agent": 0,
        "baseline_pass": False,
        "error": None,
    }
    t0 = _now()
    try:
        _populate_record(
            record,
            entry=entry,
            system_prompt=system_prompt,
            client=client,
            model=model,
            temperature=temperature,
            dsn=dsn,
        )
    finally:
        record["latency_ms"] = int((_now() - t0) * 1000)
    return record


def _populate_record(
    record: dict[str, Any],
    *,
    entry: L3GoldEntry,
    system_prompt: str,
    client: LlmClient,
    model: str,
    temperature: float,
    dsn: str,
) -> None:
    """Walk one question through the LLM → SQL → equivalence pipeline.

    Mutates ``record`` in place and returns ``None``. Errors at any
    stage short-circuit the function and are surfaced in
    ``record["error"]``; the surrounding ``_evaluate_one`` always sets
    ``latency_ms`` regardless of which branch we exited through.
    """
    try:
        response = client.complete(
            system=system_prompt,
            user=entry.question,
            model=model,
            temperature=temperature,
        )
    except LlmCallError as exc:
        record["error"] = _truncate(f"LLM call failed: {exc}")
        return

    record["llm_latency_ms"] = response.latency_ms
    record["prompt_tokens"] = response.prompt_tokens
    record["completion_tokens"] = response.completion_tokens

    agent_sql = strip_sql_fences(response.text).strip()
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
    record["baseline_pass"] = (
        record["execution_success"] and record["result_equivalence"]
    )


def _truncate(s: str) -> str:
    return s if len(s) <= _AGENT_ERROR_MAX_CHARS else s[:_AGENT_ERROR_MAX_CHARS] + "…"


def _atomic_write_json(result: BenchmarkResult, out: Path) -> None:
    """Write JSON to a temp sibling and atomically rename into place."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    write_json(result, tmp)
    os.replace(tmp, out)


def _provider_name(client: LlmClient) -> str:
    """Best-effort provider tag for the envelope.

    Looks at the client's class to keep tests free of an extra
    "provider" parameter on every fake. Falls back to ``"custom"``
    when neither built-in client matches.
    """
    if isinstance(client, AnthropicHttpClient):
        return "anthropic"
    if isinstance(client, OpenAIHttpClient):
        return "openai"
    return "custom"
