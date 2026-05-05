"""CLI tests for the ``pretensor benchmark`` scaffolding."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pretensor.cli.main import app

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _ANSI_ESCAPE_RE.sub("", text)).strip()


def test_benchmark_help_lists_three_levels() -> None:
    """``pretensor benchmark --help`` lists l1, l2, l3, and compare."""
    result = CliRunner().invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0, result.stdout
    plain = _normalize(result.stdout)
    for cmd in ("l1", "l2", "l3", "compare"):
        assert cmd in plain, f"Subcommand {cmd!r} missing from help: {plain!r}"


def test_benchmark_l1_help_documents_flags() -> None:
    """``benchmark l1 --help`` lists --dataset, --out, --graph-dir, --embeddings."""
    result = CliRunner().invoke(app, ["benchmark", "l1", "--help"])
    assert result.exit_code == 0, result.stdout
    plain = _normalize(result.stdout)
    for flag in ("--dataset", "--out", "--graph-dir", "--embeddings"):
        assert flag in plain, f"Flag {flag!r} missing from l1 help"


def test_benchmark_l2_help_documents_flags() -> None:
    """``benchmark l2 --help`` lists --dataset, --out, --graph-dir, --embeddings."""
    result = CliRunner().invoke(app, ["benchmark", "l2", "--help"])
    assert result.exit_code == 0, result.stdout
    plain = _normalize(result.stdout)
    for flag in ("--dataset", "--out", "--graph-dir", "--embeddings"):
        assert flag in plain, f"Flag {flag!r} missing from l2 help"


def test_benchmark_l3_help_documents_flags() -> None:
    """``benchmark l3 --help`` lists --dataset, --out, --graph-dir, --runner, --model, --seed."""
    result = CliRunner().invoke(app, ["benchmark", "l3", "--help"])
    assert result.exit_code == 0, result.stdout
    plain = _normalize(result.stdout)
    for flag in (
        "--dataset",
        "--out",
        "--graph-dir",
        "--runner",
        "--model",
        "--seed",
    ):
        assert flag in plain, f"Flag {flag!r} missing from l3 help"


def test_benchmark_l1_runs_to_completion(tmp_path: Path) -> None:
    """L1 runs end-to-end and writes the BenchmarkResult JSON to ``--out``."""
    out = tmp_path / "l1.json"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l1",
            "--dataset",
            "pagila",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert out.exists()
    body = json.loads(out.read_text())
    assert body["level"] == "l1"
    assert body["dataset"] == "pagila"
    assert "inferred_join_precision" in body["metrics"]


def test_benchmark_l2_runs(tmp_path: Path) -> None:
    """L2 runs end-to-end and writes a benchmark JSON to ``--out``."""
    out = tmp_path / "l2.json"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l2",
            "--dataset",
            "pagila",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert out.exists()
    body = json.loads(out.read_text())
    assert body["level"] == "l2"
    assert body["dataset"] == "pagila"
    # The three core metrics are required by AC #2.
    assert {
        "query_recall_at_5",
        "traverse_correctness",
        "compile_metric_correctness",
    } <= set(body["metrics"])


def test_benchmark_l3_baseline_requires_database_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--runner baseline reports a helpful error when the per-dataset DSN is unset."""
    monkeypatch.delenv("PAGILA_DATABASE_URL", raising=False)
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l3",
            "--runner",
            "baseline",
            "--dataset",
            "pagila",
            "--model",
            "claude-haiku-4-5",
            "--seed",
            "42",
            "--out",
            str(tmp_path / "l3-baseline.json"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stderr)
    assert "PAGILA_DATABASE_URL" in plain, plain


def test_benchmark_l3_baseline_reports_missing_anthropic_key_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ANTHROPIC_API_KEY exits 1 cleanly via the CLI input-error handler.

    Same friendly-exit treatment as missing ``PAGILA_DATABASE_URL`` —
    the user sees a single-line "set this env var" message, not a
    Python traceback. PAGILA_DATABASE_URL must be set, otherwise the
    DB-URL check fires first and we never reach the LLM client
    construction.
    """
    monkeypatch.setenv("PAGILA_DATABASE_URL", "postgresql://stub/pagila")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l3",
            "--runner",
            "baseline",
            "--dataset",
            "pagila",
            "--model",
            "claude-haiku-4-5",
            "--seed",
            "42",
            "--out",
            str(tmp_path / "l3-baseline.json"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stderr)
    assert "ANTHROPIC_API_KEY" in plain, plain
    # Belt-and-braces: traceback would surface a Python frame-line — assert it doesn't.
    assert "Traceback" not in plain, plain


def test_benchmark_l3_baseline_reports_missing_ddl_clearly(
    tmp_path: Path,
) -> None:
    """A dataset without a DDL bundle exits 1 with a clear message via the CLI.

    ``analytics_dwh`` is a synthetic fixture — no ``scripts/data/...
    /schema.sql`` is checked in for it — so the runner raises
    ``FileNotFoundError`` before any DB access. Verifies the CLI's
    ``_handle_input_error`` path catches this and renders an exit-1
    error rather than a Python traceback.
    """
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l3",
            "--runner",
            "baseline",
            "--dataset",
            "analytics_dwh",
            "--model",
            "claude-haiku-4-5",
            "--seed",
            "1",
            "--out",
            str(tmp_path / "l3-baseline.json"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stderr)
    assert "DDL bundle" in plain, plain


def test_benchmark_l3_pretensor_requires_database_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--runner pretensor`` reports a clear error when the per-dataset DSN is unset.

    Mirrors the baseline equivalent: the CLI's ``_handle_input_error``
    path surfaces the missing-env-var ``LookupError`` as exit 1 with
    a one-line message, not a Python traceback.
    """
    monkeypatch.delenv("PAGILA_DATABASE_URL", raising=False)
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l3",
            "--runner",
            "pretensor",
            "--dataset",
            "pagila",
            "--model",
            "claude-haiku-4-5",
            "--seed",
            "42",
            "--out",
            str(tmp_path / "l3-pretensor.json"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stderr)
    assert "PAGILA_DATABASE_URL" in plain, plain
    assert "Traceback" not in plain, plain


def test_benchmark_l3_pretensor_reports_mcp_subprocess_failure_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken ``pretensor serve`` binary surfaces as exit 1 with no traceback.

    Distinct from the missing-graph-dir path: the graph dir IS present
    and indexed, but the MCP subprocess can't be spawned. The CLI's
    ``_handle_input_error`` now also catches :class:`McpClientError`,
    so operators see a one-line "Failed to start MCP subprocess …"
    message, not a Python traceback.
    """
    monkeypatch.setenv("PAGILA_DATABASE_URL", "postgresql://stub/pagila")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Make `pretensor` unfindable so StdioMcpClient.__enter__ raises.
    monkeypatch.setenv("PATH", "/var/empty-no-binaries-here")
    graph = tmp_path / ".pretensor"
    (graph / "graphs").mkdir(parents=True)
    (graph / "graphs" / "pagila.kuzu").touch()
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l3",
            "--runner",
            "pretensor",
            "--dataset",
            "pagila",
            "--model",
            "claude-haiku-4-5",
            "--seed",
            "1",
            "--graph-dir",
            str(graph),
            "--out",
            str(tmp_path / "l3-pretensor.json"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stderr)
    assert "Failed to start MCP subprocess" in plain, plain
    assert "Traceback" not in plain, plain


def test_benchmark_l3_pretensor_reports_missing_graph_dir_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing ``.pretensor/graphs/*.kuzu`` exits 1 with a clear message.

    The pretensor runner refuses to run against an unindexed graph dir
    so the baseline / pretensor comparison never silently sees stale or
    re-indexed data. Verifies the CLI catches the resulting
    :class:`FileNotFoundError` and prints the operator-friendly hint
    (``pretensor index <dsn>``).
    """
    monkeypatch.setenv("PAGILA_DATABASE_URL", "postgresql://stub/pagila")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    empty_graph = tmp_path / ".pretensor"
    empty_graph.mkdir()
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l3",
            "--runner",
            "pretensor",
            "--dataset",
            "pagila",
            "--model",
            "claude-haiku-4-5",
            "--seed",
            "1",
            "--graph-dir",
            str(empty_graph),
            "--out",
            str(tmp_path / "l3-pretensor.json"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stderr)
    assert "pretensor index" in plain, plain
    assert "Traceback" not in plain, plain


def test_benchmark_l3_requires_runner_and_model() -> None:
    """``l3`` rejects calls missing --runner or --model (exit 2)."""
    r1 = CliRunner().invoke(
        app,
        ["benchmark", "l3", "--dataset", "pagila", "--model", "x"],
    )
    assert r1.exit_code == 2

    r2 = CliRunner().invoke(
        app,
        ["benchmark", "l3", "--dataset", "pagila", "--runner", "baseline"],
    )
    assert r2.exit_code == 2


def test_benchmark_l3_rejects_unknown_runner() -> None:
    """Unknown --runner values exit 2."""
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l3",
            "--runner",
            "bogus",
            "--dataset",
            "pagila",
            "--model",
            "claude-haiku-4-5",
        ],
    )
    assert result.exit_code == 2


def test_benchmark_rejects_unknown_dataset() -> None:
    """Unknown --dataset values exit with code 2 (per spec §CLI contract)."""
    result = CliRunner().invoke(
        app,
        ["benchmark", "l1", "--dataset", "bogus"],
    )
    assert result.exit_code == 2


def test_benchmark_l1_out_is_optional() -> None:
    """--out may be omitted; the JSON document is written to stdout."""
    result = CliRunner().invoke(
        app,
        ["benchmark", "l1", "--dataset", "adversarial"],
    )
    assert result.exit_code == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["level"] == "l1"
    assert body["dataset"] == "adversarial"


def test_benchmark_l1_accepts_embeddings_flag() -> None:
    """--embeddings / --no-embeddings are both accepted on l1."""
    for flag in ("--embeddings", "--no-embeddings"):
        result = CliRunner().invoke(
            app,
            ["benchmark", "l1", "--dataset", "adversarial", flag],
        )
        assert result.exit_code == 0, (
            f"l1 {flag!r} unexpectedly rejected: {result.stderr!r}"
        )


def test_benchmark_l2_accepts_embeddings_flag() -> None:
    """--embeddings / --no-embeddings are both accepted on l2.

    Asserts that the no-out path emits a valid JSON document on
    stdout — mirrors ``test_benchmark_l1_out_is_optional`` so the L2
    stdout branch is not silently broken.
    """
    for flag in ("--embeddings", "--no-embeddings"):
        result = CliRunner().invoke(
            app,
            ["benchmark", "l2", "--dataset", "pagila", flag],
        )
        assert result.exit_code == 0, (
            f"l2 {flag!r} unexpectedly rejected: {result.stderr!r}"
        )
        body = json.loads(result.stdout)
        assert body["level"] == "l2"
        assert body["dataset"] == "pagila"


def test_benchmark_accepts_every_spec_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every dataset key in the spec is accepted by the enum.

    Uses ``l3 --runner baseline`` rather than ``l1`` / ``l2`` so the
    assertion stays cheap — a full sweep through real metric computation
    across six fixtures runs into minutes. With no DB env vars set, the
    runner exits 1 via the friendly missing-env-var path; what we're
    testing here is that the dataset enum accepts each key (anything
    other than exit 1 means it was rejected at parse time as exit 2).
    """
    for var in (
        "PAGILA_DATABASE_URL",
        "TPCH_DATABASE_URL",
        "ADVENTUREWORKS_DATABASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    for dataset in (
        "pagila",
        "tpch",
        "analytics_dwh",
        "adversarial",
        "saas_multitenant",
        "adventureworks",
    ):
        result = CliRunner().invoke(
            app,
            [
                "benchmark",
                "l3",
                "--dataset",
                dataset,
                "--runner",
                "baseline",
                "--model",
                "claude-haiku-4-5",
            ],
        )
        # Friendly env-var / DDL error → exit 1 (not 2 = bad args).
        assert result.exit_code == 1, (
            f"dataset={dataset!r} rejected unexpectedly: "
            f"exit={result.exit_code} stderr={result.stderr!r}"
        )
