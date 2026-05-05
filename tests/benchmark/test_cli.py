"""CLI tests for the ``pretensor benchmark`` scaffolding."""

from __future__ import annotations

import re
from pathlib import Path

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


def test_benchmark_l1_unimplemented(tmp_path: Path) -> None:
    """Valid args surface a NotImplementedError naming the L1 scope."""
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l1",
            "--dataset",
            "pagila",
            "--out",
            str(tmp_path / "l1.json"),
        ],
    )
    assert result.exit_code == 1
    assert "L1 metrics not implemented" in _normalize(result.stderr)


def test_benchmark_l2_unimplemented(tmp_path: Path) -> None:
    """Valid args surface a NotImplementedError naming the L2 scope."""
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "l2",
            "--dataset",
            "pagila",
            "--out",
            str(tmp_path / "l2.json"),
        ],
    )
    assert result.exit_code == 1
    assert "L2 metrics not implemented" in _normalize(result.stderr)


def test_benchmark_l3_baseline_unimplemented(tmp_path: Path) -> None:
    """--runner baseline surfaces the baseline NotImplementedError."""
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
    assert "L3 baseline runner not implemented" in plain, plain
    assert "pretensor runner" not in plain, (
        "baseline runner must not mention the pretensor variant"
    )


def test_benchmark_l3_pretensor_unimplemented(tmp_path: Path) -> None:
    """--runner pretensor surfaces the pretensor NotImplementedError."""
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
            "--out",
            str(tmp_path / "l3-pretensor.json"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stderr)
    assert "L3 pretensor runner not implemented" in plain, plain
    assert "baseline runner" not in plain, (
        "pretensor runner must not mention the baseline variant"
    )


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
    """--out may be omitted; stub still runs and exits 1."""
    result = CliRunner().invoke(
        app,
        ["benchmark", "l1", "--dataset", "pagila"],
    )
    assert result.exit_code == 1
    assert "L1 metrics not implemented" in _normalize(result.stderr)


def test_benchmark_l1_accepts_embeddings_flag() -> None:
    """--embeddings / --no-embeddings are both accepted on l1."""
    for flag in ("--embeddings", "--no-embeddings"):
        result = CliRunner().invoke(
            app,
            ["benchmark", "l1", "--dataset", "pagila", flag],
        )
        assert result.exit_code == 1, (
            f"l1 {flag!r} unexpectedly rejected: {result.stderr!r}"
        )


def test_benchmark_l2_accepts_embeddings_flag() -> None:
    """--embeddings / --no-embeddings are both accepted on l2."""
    for flag in ("--embeddings", "--no-embeddings"):
        result = CliRunner().invoke(
            app,
            ["benchmark", "l2", "--dataset", "pagila", flag],
        )
        assert result.exit_code == 1, (
            f"l2 {flag!r} unexpectedly rejected: {result.stderr!r}"
        )


def test_benchmark_accepts_every_spec_dataset() -> None:
    """Every dataset key in the spec is accepted by the enum."""
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
            ["benchmark", "l1", "--dataset", dataset],
        )
        # Stub raises NotImplementedError → exit 1 (not 2 = bad args).
        assert result.exit_code == 1, (
            f"dataset={dataset!r} rejected unexpectedly: "
            f"exit={result.exit_code} stderr={result.stderr!r}"
        )
