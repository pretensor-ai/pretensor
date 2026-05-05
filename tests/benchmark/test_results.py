"""Tests for the benchmark results aggregator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pretensor.benchmark.results import (
    BenchmarkResult,
    ComparisonError,
    Metric,
    compare,
    read_json,
    write_csv,
    write_json,
)
from pretensor.cli.main import app

_FIXTURES = Path(__file__).parent / "fixtures"


def _sample_result(
    *,
    level: str = "l1",
    dataset: str = "pagila",
    metrics: dict[str, Metric] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        level=level,
        dataset=dataset,
        pretensor_version="0.1.0",
        embeddings_enabled=False,
        ran_at="2026-04-22T12:00:00Z",
        fixture_sha=(
            "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        ),
        metrics=metrics
        or {
            "inferred_join_precision": Metric(value=0.92, direction="higher_is_better"),
            "inferred_join_recall": Metric(value=0.88, direction="higher_is_better"),
        },
        per_item=[
            {
                "id": "public.customer->public.address",
                "expected": True,
                "predicted": True,
            },
        ],
        notes=[],
    )


def test_metric_round_trip_preserves_value_and_direction(tmp_path: Path) -> None:
    """Metric -> JSON -> Metric is byte-stable."""
    result = _sample_result()
    out = tmp_path / "result.json"
    write_json(result, out)
    loaded = read_json(out)
    assert loaded == result


def test_write_json_is_byte_stable(tmp_path: Path) -> None:
    """Two writes of the same result produce byte-identical files."""
    result = _sample_result()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_json(result, a)
    write_json(result, b)
    assert a.read_bytes() == b.read_bytes()


def test_write_json_sorts_keys_and_ends_with_newline(tmp_path: Path) -> None:
    """JSON output is sort_keys=True, indent=2, trailing newline."""
    result = _sample_result()
    out = tmp_path / "result.json"
    write_json(result, out)
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    parsed = json.loads(text)
    keys = list(parsed.keys())
    assert keys == sorted(keys), keys


def test_metric_value_can_be_none(tmp_path: Path) -> None:
    """A metric with value=None round-trips (unavailable metric)."""
    result = _sample_result(
        metrics={
            "semantic_search_recall_at_1": Metric(
                value=None, direction="higher_is_better"
            ),
        },
    )
    out = tmp_path / "r.json"
    write_json(result, out)
    loaded = read_json(out)
    assert loaded.metrics["semantic_search_recall_at_1"].value is None


def test_compare_no_change_has_no_regression() -> None:
    """Identical inputs -> no regression, no diffs."""
    a = _sample_result()
    b = _sample_result()
    report = compare(a, b)
    assert report.has_regression is False
    assert report.regressions == []
    assert report.improvements == []
    assert report.added == []
    assert report.removed == []


def test_compare_higher_is_better_drop_beyond_tolerance_is_regression() -> None:
    """A higher_is_better metric dropping by more than tolerance regresses."""
    baseline = _sample_result(
        metrics={"p": Metric(value=0.90, direction="higher_is_better")},
    )
    current = _sample_result(
        metrics={"p": Metric(value=0.80, direction="higher_is_better")},
    )
    report = compare(baseline, current, tolerance=0.001)
    assert report.has_regression is True
    assert len(report.regressions) == 1
    assert report.regressions[0].name == "p"


def test_compare_higher_is_better_drop_within_tolerance_is_not_regression() -> None:
    """A drop within tolerance is noise, not a regression."""
    baseline = _sample_result(
        metrics={"p": Metric(value=0.9000005, direction="higher_is_better")},
    )
    current = _sample_result(
        metrics={"p": Metric(value=0.9000000, direction="higher_is_better")},
    )
    report = compare(baseline, current, tolerance=1e-6)
    assert report.has_regression is False


def test_compare_higher_is_better_rise_is_improvement() -> None:
    baseline = _sample_result(
        metrics={"p": Metric(value=0.80, direction="higher_is_better")},
    )
    current = _sample_result(
        metrics={"p": Metric(value=0.90, direction="higher_is_better")},
    )
    report = compare(baseline, current, tolerance=1e-6)
    assert report.has_regression is False
    assert len(report.improvements) == 1
    assert report.improvements[0].name == "p"


def test_compare_lower_is_better_rise_is_regression() -> None:
    """A lower_is_better metric rising beyond tolerance regresses."""
    baseline = _sample_result(
        metrics={"latency_ms": Metric(value=100.0, direction="lower_is_better")},
    )
    current = _sample_result(
        metrics={"latency_ms": Metric(value=120.0, direction="lower_is_better")},
    )
    report = compare(baseline, current, tolerance=1e-6)
    assert report.has_regression is True
    assert report.regressions[0].name == "latency_ms"


def test_compare_lower_is_better_drop_is_improvement() -> None:
    baseline = _sample_result(
        metrics={"latency_ms": Metric(value=120.0, direction="lower_is_better")},
    )
    current = _sample_result(
        metrics={"latency_ms": Metric(value=100.0, direction="lower_is_better")},
    )
    report = compare(baseline, current, tolerance=1e-6)
    assert report.has_regression is False
    assert len(report.improvements) == 1


def test_compare_metric_missing_in_current_is_regression() -> None:
    """A metric present in baseline but missing in current is a regression."""
    baseline = _sample_result(
        metrics={
            "p": Metric(value=0.9, direction="higher_is_better"),
            "r": Metric(value=0.8, direction="higher_is_better"),
        },
    )
    current = _sample_result(
        metrics={"p": Metric(value=0.9, direction="higher_is_better")},
    )
    report = compare(baseline, current, tolerance=1e-6)
    assert report.has_regression is True
    assert report.removed == ["r"]


def test_compare_new_metric_in_current_is_added_not_regression() -> None:
    """A metric new in current is reported under `added`, not a regression."""
    baseline = _sample_result(
        metrics={"p": Metric(value=0.9, direction="higher_is_better")},
    )
    current = _sample_result(
        metrics={
            "p": Metric(value=0.9, direction="higher_is_better"),
            "new_metric": Metric(value=0.5, direction="higher_is_better"),
        },
    )
    report = compare(baseline, current, tolerance=1e-6)
    assert report.has_regression is False
    assert report.added == ["new_metric"]


def test_compare_metric_with_none_value_is_skipped_not_regression() -> None:
    """A metric whose value is None on either side is skipped (no flag)."""
    baseline = _sample_result(
        metrics={"p": Metric(value=None, direction="higher_is_better")},
    )
    current = _sample_result(
        metrics={"p": Metric(value=0.9, direction="higher_is_better")},
    )
    report = compare(baseline, current, tolerance=1e-6)
    assert report.has_regression is False


def test_compare_different_dataset_raises() -> None:
    a = _sample_result(dataset="pagila")
    b = _sample_result(dataset="tpch")
    with pytest.raises(ComparisonError, match="dataset"):
        compare(a, b)


def test_compare_different_level_raises() -> None:
    a = _sample_result(level="l1")
    b = _sample_result(level="l2")
    with pytest.raises(ComparisonError, match="level"):
        compare(a, b)


def test_format_diff_lists_regression_with_delta() -> None:
    """The human-readable diff names regressed metrics and shows the delta."""
    baseline = _sample_result(
        metrics={
            "inferred_join_precision": Metric(value=0.90, direction="higher_is_better")
        },
    )
    current = _sample_result(
        metrics={
            "inferred_join_precision": Metric(value=0.80, direction="higher_is_better")
        },
    )
    report = compare(baseline, current, tolerance=1e-6)
    text = report.format_diff()
    assert "inferred_join_precision" in text
    assert "0.90" in text or "0.9" in text
    assert "0.80" in text or "0.8" in text


def test_write_csv_one_row_per_result_with_metric_columns(tmp_path: Path) -> None:
    """Each result becomes one row; metric names become columns."""
    r1 = _sample_result(
        dataset="pagila",
        metrics={
            "inferred_join_precision": Metric(value=0.92, direction="higher_is_better"),
            "inferred_join_recall": Metric(value=0.88, direction="higher_is_better"),
        },
    )
    r2 = _sample_result(
        dataset="tpch",
        metrics={
            "inferred_join_precision": Metric(value=0.85, direction="higher_is_better"),
            "inferred_join_recall": Metric(value=0.80, direction="higher_is_better"),
        },
    )
    out = tmp_path / "results.csv"
    write_csv([r1, r2], out)
    text = out.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert len(lines) == 3
    header = lines[0].split(",")
    for col in (
        "level",
        "dataset",
        "pretensor_version",
        "embeddings_enabled",
        "ran_at",
        "inferred_join_precision",
        "inferred_join_recall",
    ):
        assert col in header, f"missing column {col!r}: {header!r}"


def test_write_csv_unioned_metric_columns_with_blanks(tmp_path: Path) -> None:
    """Metric column union: a row missing a metric writes an empty cell."""
    r1 = _sample_result(
        dataset="pagila",
        metrics={"a": Metric(value=0.5, direction="higher_is_better")},
    )
    r2 = _sample_result(
        dataset="tpch",
        metrics={"b": Metric(value=0.7, direction="higher_is_better")},
    )
    out = tmp_path / "u.csv"
    write_csv([r1, r2], out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["a"] == "0.5"
    assert rows[0]["b"] == ""
    assert rows[1]["a"] == ""
    assert rows[1]["b"] == "0.7"


def test_write_csv_empty_value_for_none_metric(tmp_path: Path) -> None:
    r = _sample_result(
        metrics={"unavailable": Metric(value=None, direction="higher_is_better")},
    )
    out = tmp_path / "n.csv"
    write_csv([r], out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["unavailable"] == ""


def test_cli_compare_exits_zero_when_no_regression() -> None:
    """`benchmark compare` with two identical files exits 0."""
    baseline = _FIXTURES / "baseline.json"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "compare",
            "--baseline",
            str(baseline),
            "--current",
            str(baseline),
        ],
    )
    assert result.exit_code == 0, result.stderr


def test_cli_compare_exits_one_on_regression() -> None:
    """`benchmark compare` exits 1 when a regression is detected."""
    baseline = _FIXTURES / "baseline.json"
    regressed = _FIXTURES / "regressed.json"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "compare",
            "--baseline",
            str(baseline),
            "--current",
            str(regressed),
        ],
    )
    assert result.exit_code == 1, result.stderr
    assert "inferred_join_precision" in result.stderr
    assert "REGRESSION" in result.stderr.upper()


def test_cli_compare_exits_two_on_dataset_mismatch(tmp_path: Path) -> None:
    """`benchmark compare` exits 2 when datasets disagree."""
    baseline = _FIXTURES / "baseline.json"
    other = tmp_path / "other.json"
    text = baseline.read_text()
    other.write_text(text.replace('"dataset": "pagila"', '"dataset": "tpch"'))
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "compare",
            "--baseline",
            str(baseline),
            "--current",
            str(other),
        ],
    )
    assert result.exit_code == 2, result.stderr
    assert "dataset" in result.stderr.lower()


def test_cli_compare_shows_in_help() -> None:
    """`pretensor benchmark --help` lists the compare subcommand."""
    result = CliRunner().invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "compare" in result.stdout


def test_cli_compare_exits_two_on_malformed_json(tmp_path: Path) -> None:
    """A non-JSON --current file exits 2 with a parse error on stderr."""
    baseline = _FIXTURES / "baseline.json"
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not valid JSON")
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "compare",
            "--baseline",
            str(baseline),
            "--current",
            str(bad),
        ],
    )
    assert result.exit_code == 2, result.stderr
    assert "current" in result.stderr.lower()


def test_cli_compare_exits_two_on_level_mismatch(tmp_path: Path) -> None:
    """`benchmark compare` exits 2 when levels disagree."""
    baseline = _FIXTURES / "baseline.json"
    other = tmp_path / "other.json"
    other.write_text(baseline.read_text().replace('"level": "l1"', '"level": "l2"'))
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "compare",
            "--baseline",
            str(baseline),
            "--current",
            str(other),
        ],
    )
    assert result.exit_code == 2, result.stderr
    assert "level" in result.stderr.lower()


def test_format_diff_no_regression_string() -> None:
    """The clean-path diff output contains the no-regression marker."""
    a = _sample_result()
    report = compare(a, a)
    assert "no regression" in report.format_diff()
