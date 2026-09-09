"""Tests for the release-gate script.

The gate orchestrates existing pieces (``run_l1`` / ``run_l2`` and the
``compare`` aggregator) and is otherwise pure I/O over the
``tests/benchmark/results/<tag>/`` directory tree. Tests stub the
benchmark-runner callable and the git-tag resolvers so the suite stays
hermetic — no subprocesses, no real git history.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import pretensor.benchmark.release_gate as release_gate
from pretensor.benchmark.release_gate import main
from pretensor.benchmark.results import BenchmarkResult, Metric, write_json
from pretensor.benchmark.runner import Dataset

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FAKE_FIXTURE_SHA = "sha256:" + "0" * 64


def _result(
    *,
    level: str,
    dataset: str,
    metrics: dict[str, Metric] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        level=level,
        dataset=dataset,
        pretensor_version="0.1.0",
        embeddings_enabled=False,
        ran_at="2026-04-22T12:00:00Z",
        fixture_sha=_FAKE_FIXTURE_SHA,
        metrics=metrics
        or {
            "inferred_join_precision": Metric(value=0.9, direction="higher_is_better"),
        },
        per_item=[],
        notes=[],
    )


def _stage(dir_: Path, results: list[BenchmarkResult]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for r in results:
        write_json(r, dir_ / f"{r.dataset}-{r.level}.json")


def _release_datasets() -> tuple[str, ...]:
    """Dataset values the gate evaluates by default.

    Derived from the production constant rather than hardcoded so a new
    release-gating dataset added to ``release_gate._RELEASE_DATASETS``
    automatically gets test coverage in the existing scenarios.
    """
    return tuple(d.value for d in release_gate._RELEASE_DATASETS)


def _full_baseline(dir_: Path, *, value: float = 0.9) -> None:
    """Stage every (dataset, level) pair that the gate iterates by default."""
    metrics = {
        "inferred_join_precision": Metric(value=value, direction="higher_is_better")
    }
    results = []
    for dataset in _release_datasets():
        for level in ("l1", "l2"):
            results.append(_result(level=level, dataset=dataset, metrics=dict(metrics)))
    _stage(dir_, results)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_first_tag_no_baseline_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No previous tag → gate exits 0 with a clear notice on stderr."""
    monkeypatch.setattr(release_gate, "_resolve_previous_tag", lambda: None)
    monkeypatch.setattr(release_gate, "_resolve_candidate_tag", lambda: "v0.1.0a0")
    # Nothing should be run; if it is, the test fails loudly.
    monkeypatch.setattr(
        release_gate,
        "_run_benchmark",
        _exploding_runner("first-tag path should not invoke runner"),
    )

    rc = main(["--results-base", str(tmp_path)])
    err = capsys.readouterr().err

    assert rc == 0
    assert "no baseline to compare" in err.lower()


def test_all_pass_exits_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Identical baseline + current → exit 0, no regression text."""
    base = tmp_path / "v0.1.0a0"
    cur = tmp_path / "v0.1.0a1"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)

    rc = main(
        [
            "--baseline-dir",
            str(base),
            "--current-dir",
            str(cur),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 0
    assert "REGRESSION" not in err


def test_one_regression_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A single dataset/level metric drop produces exit 1 with a labelled summary."""
    base = tmp_path / "prev"
    cur = tmp_path / "cur"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)
    # Overwrite pagila/l2 with a regressed metric.
    write_json(
        _result(
            level="l2",
            dataset="pagila",
            metrics={
                "inferred_join_precision": Metric(
                    value=0.5, direction="higher_is_better"
                )
            },
        ),
        cur / "pagila-l2.json",
    )

    rc = main(
        [
            "--baseline-dir",
            str(base),
            "--current-dir",
            str(cur),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "REGRESSION" in err
    assert "pagila/l2" in err
    # Other (passing) datasets should not appear in the regression block.
    assert "tpch/l2" not in err
    assert "adventureworks/l1" not in err


def test_passing_datasets_not_reported_as_regressed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One regressed dataset must not flag the others; aggregate still fails."""
    base = tmp_path / "prev"
    cur = tmp_path / "cur"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)
    write_json(
        _result(
            level="l1",
            dataset="adventureworks",
            metrics={
                "inferred_join_precision": Metric(
                    value=0.4, direction="higher_is_better"
                )
            },
        ),
        cur / "adventureworks-l1.json",
    )

    rc = main(
        [
            "--baseline-dir",
            str(base),
            "--current-dir",
            str(cur),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "adventureworks/l1" in err
    assert "pagila/l1" not in err
    assert "tpch/l2" not in err


def test_missing_baseline_for_one_dataset_is_not_a_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dataset added between tags has no prior JSON — skip it with a notice."""
    base = tmp_path / "prev"
    cur = tmp_path / "cur"
    # Stage baselines for two datasets only; the third (adventureworks)
    # is "new" — no baseline file was published with the previous tag.
    metrics = {
        "inferred_join_precision": Metric(value=0.9, direction="higher_is_better")
    }
    for dataset in ("pagila", "tpch"):
        for level in ("l1", "l2"):
            _stage(base, [_result(level=level, dataset=dataset, metrics=dict(metrics))])
    _full_baseline(cur, value=0.9)

    rc = main(
        [
            "--baseline-dir",
            str(base),
            "--current-dir",
            str(cur),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 0
    assert "no baseline" in err.lower()
    assert "adventureworks" in err


def test_benchmark_run_failure_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If a (dataset, level) run can't produce a JSON, the gate fails loudly."""
    base = tmp_path / "prev"
    cur_root = tmp_path / "cur"
    _full_baseline(base, value=0.9)

    monkeypatch.setattr(release_gate, "_resolve_previous_tag", lambda: "v0.1.0a0")
    monkeypatch.setattr(release_gate, "_resolve_candidate_tag", lambda: "v0.1.0a1")

    def _broken_runner(
        level: str,
        dataset: Dataset,
        out: Path,
        *,
        graph_dir: Path,
        embeddings: bool,
    ) -> None:
        raise RuntimeError(f"boom on {dataset.value}/{level}")

    monkeypatch.setattr(release_gate, "_run_benchmark", _broken_runner)

    def _resolve(tag: str, _results_base: Path) -> Path:
        return base if tag == "v0.1.0a0" else cur_root

    monkeypatch.setattr(release_gate, "_resolve_results_dir", _resolve)

    rc = main(["--results-base", str(tmp_path)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "failed to produce" in err.lower()


def test_l3_results_are_ignored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stray L3 JSON in the baseline / current dirs must not change the verdict."""
    base = tmp_path / "prev"
    cur = tmp_path / "cur"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)
    # Stage an L3 regression that the gate must ignore.
    write_json(
        _result(
            level="l3",
            dataset="pagila",
            metrics={
                "nl2sql_success_rate_pretensor": Metric(
                    value=0.5, direction="higher_is_better"
                )
            },
        ),
        base / "pagila-l3.json",
    )
    write_json(
        _result(
            level="l3",
            dataset="pagila",
            metrics={
                "nl2sql_success_rate_pretensor": Metric(
                    value=0.1, direction="higher_is_better"
                )
            },
        ),
        cur / "pagila-l3.json",
    )

    rc = main(
        [
            "--baseline-dir",
            str(base),
            "--current-dir",
            str(cur),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 0
    assert "/l3" not in err  # the gate must not even mention L3 pairs


def test_within_tolerance_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sub-1e-6 noise must not be flagged as a regression."""
    base = tmp_path / "prev"
    cur = tmp_path / "cur"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9 - 1e-9)

    rc = main(
        [
            "--baseline-dir",
            str(base),
            "--current-dir",
            str(cur),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 0
    assert "REGRESSION" not in err


def test_missing_current_file_yields_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--current-dir`` provided but a per-pair JSON is missing → exit 1."""
    base = tmp_path / "prev"
    cur = tmp_path / "cur"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)
    # Delete one current-side result; baseline still has it, so the
    # gate must complain rather than silently skip.
    (cur / "tpch-l1.json").unlink()

    rc = main(
        [
            "--baseline-dir",
            str(base),
            "--current-dir",
            str(cur),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "tpch/l1" in err
    assert "no current result" in err.lower()


def test_comparison_error_yields_error_outcome(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A schema-mismatched baseline (wrong dataset/level pair) → exit 1.

    ``compare()`` raises ``ComparisonError`` if the two inputs disagree on
    dataset or level. The gate matches per-pair JSON by filename, so the
    only way to provoke this in practice is a mislabelled JSON whose
    ``dataset`` field disagrees with its filename.
    """
    base = tmp_path / "prev"
    cur = tmp_path / "cur"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)
    # Overwrite pagila-l1.json with a payload whose dataset field is
    # 'tpch' — same filename, mismatched body. compare() will refuse.
    write_json(
        _result(
            level="l1",
            dataset="tpch",
            metrics={
                "inferred_join_precision": Metric(
                    value=0.9, direction="higher_is_better"
                )
            },
        ),
        cur / "pagila-l1.json",
    )

    rc = main(
        [
            "--baseline-dir",
            str(base),
            "--current-dir",
            str(cur),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "pagila/l1" in err
    assert "cannot compare across datasets" in err.lower()


def test_help_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` prints usage and exits 0; matches the project's CLI test pattern."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    out = capsys.readouterr().out
    assert excinfo.value.code == 0
    # Sanity: the help text mentions the gate's defining flags.
    assert "--baseline-dir" in out
    assert "--current-dir" in out
    assert "--datasets" in out


def test_corrupt_result_json_yields_error_outcome(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A truncated / non-JSON result file surfaces as an error, exit 1."""
    base = tmp_path / "prev"
    cur = tmp_path / "cur"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)
    # Corrupt one file in-place — partial JSON write or disk corruption.
    (cur / "pagila-l1.json").write_text("{not valid json", encoding="utf-8")

    rc = main(
        [
            "--baseline-dir",
            str(base),
            "--current-dir",
            str(cur),
        ]
    )
    err = capsys.readouterr().err

    assert rc == 1
    assert "pagila/l1" in err
    assert "failed to read result json" in err.lower()


def test_invalid_dataset_name_exits_via_argparse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown ``--datasets`` value is rejected by argparse with a usage line."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--datasets", "definitely_not_a_dataset"])
    err = capsys.readouterr().err
    # argparse exits 2 on argument errors and emits "usage:" + "error:" lines.
    assert excinfo.value.code == 2
    assert "definitely_not_a_dataset" in err
    assert "usage:" in err.lower()


def test_missing_archive_dir_for_previous_tag_exits_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Previous tag exists but its results dir is absent → exit 1, not silent pass.

    If the post-publish archive PR for the prior tag never landed, the
    per-tag results directory will be missing and every (dataset, level)
    pair would silently 'skip' — masquerading as a first-tag run and
    letting regressions through. The gate must refuse instead.
    """
    monkeypatch.setattr(release_gate, "_resolve_previous_tag", lambda: "v0.1.0a0")
    monkeypatch.setattr(release_gate, "_resolve_candidate_tag", lambda: "v0.1.0a1")
    monkeypatch.setattr(
        release_gate,
        "_run_benchmark",
        _exploding_runner("missing-archive path should not invoke runner"),
    )

    # results-base exists but the v0.1.0a0/ subdir does not.
    (tmp_path / "v0.1.0a1").mkdir()

    rc = main(["--results-base", str(tmp_path)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "v0.1.0a0" in err
    assert "absent" in err.lower() or "refusing" in err.lower()


def test_resolve_candidate_tag_no_env_no_tag_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_resolve_candidate_tag`` raises SystemExit when there's nothing to resolve.

    Covers the branch where ``GITHUB_REF_NAME`` is unset (or empty) AND
    ``git describe --tags --exact-match HEAD`` fails — operator forgot to
    pass ``--current-dir`` and isn't sitting at a tagged commit.
    """
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)

    class _Result:
        returncode = 1
        stdout = ""
        stderr = "fatal: no tag exactly matches"

    monkeypatch.setattr(
        release_gate.subprocess,
        "run",
        lambda *_a, **_kw: _Result(),
    )

    with pytest.raises(SystemExit) as excinfo:
        release_gate._resolve_candidate_tag()
    msg = str(excinfo.value)
    assert "GITHUB_REF_NAME" in msg
    assert "--current-dir" in msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exploding_runner(reason: str) -> Any:
    """Returns a runner that raises if invoked — use to assert "no run" paths."""

    def _explode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(reason)

    return _explode


# ---------------------------------------------------------------------------
# Accepted regressions (tests/benchmark/results/<tag>/accepted-regressions.toml)
# ---------------------------------------------------------------------------


def _stage_regression(
    tmp_path: Path,
    *,
    metric: str = "inferred_join_precision",
    direction: str = "higher_is_better",
    baseline_value: float = 0.9,
    current_value: float = 0.5,
) -> tuple[Path, Path]:
    """Stage a full passing baseline + current, then regress one pair.

    The regressed pair is ``adventureworks/l1`` on ``metric``.
    """
    base = tmp_path / "v0.1.0"
    cur = tmp_path / "v0.2.0"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)
    _stage(
        base,
        [
            _result(
                level="l1",
                dataset="adventureworks",
                metrics={metric: Metric(value=baseline_value, direction=direction)},  # type: ignore[arg-type]
            )
        ],
    )
    _stage(
        cur,
        [
            _result(
                level="l1",
                dataset="adventureworks",
                metrics={metric: Metric(value=current_value, direction=direction)},  # type: ignore[arg-type]
            )
        ],
    )
    return base, cur


def _write_accepted(cur: Path, body: str) -> None:
    (cur / release_gate._ACCEPTED_FILENAME).write_text(body)


_ACCEPTED_ENTRY = """
[[accepted]]
dataset = "adventureworks"
level = "l1"
metric = "inferred_join_precision"
accepted_value = 0.5
reason = "deliberate precision/recall trade-off documented in the changelog"
"""


def test_accepted_regression_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A documented regression no worse than accepted_value passes the gate."""
    base, cur = _stage_regression(tmp_path, current_value=0.5)
    _write_accepted(cur, _ACCEPTED_ENTRY)

    rc = main(["--baseline-dir", str(base), "--current-dir", str(cur)])
    err = capsys.readouterr().err

    assert rc == 0
    assert "RELEASE GATE: REGRESSION DETECTED" not in err
    assert "accepted regression" in err.lower()
    assert "adventureworks/l1" in err
    assert "inferred_join_precision" in err
    assert "deliberate precision/recall trade-off" in err


def test_accepted_regression_within_tolerance_of_accepted_value_passes(
    tmp_path: Path,
) -> None:
    """Float noise below the comparator tolerance does not defeat an acceptance."""
    base, cur = _stage_regression(tmp_path, current_value=0.5 - 1e-9)
    _write_accepted(cur, _ACCEPTED_ENTRY)

    rc = main(["--baseline-dir", str(base), "--current-dir", str(cur)])

    assert rc == 0


def test_regression_worse_than_accepted_value_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """accepted_value is a floor, not a blanket waiver: worse still blocks."""
    base, cur = _stage_regression(tmp_path, current_value=0.4)
    _write_accepted(cur, _ACCEPTED_ENTRY)

    rc = main(["--baseline-dir", str(base), "--current-dir", str(cur)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "RELEASE GATE: REGRESSION DETECTED" in err
    assert "inferred_join_precision" in err
    assert "accepted_value" in err
    # The entry did match a regression; it must not also be reported as stale.
    assert "matched no regression" not in err


def test_accepted_lower_is_better_direction(tmp_path: Path) -> None:
    """For lower_is_better metrics accepted_value is a ceiling."""
    base, cur = _stage_regression(
        tmp_path,
        metric="latency_ms",
        direction="lower_is_better",
        baseline_value=10.0,
        current_value=20.0,
    )
    _write_accepted(
        cur,
        """
[[accepted]]
dataset = "adventureworks"
level = "l1"
metric = "latency_ms"
accepted_value = 20.0
reason = "slower but correct"
""",
    )
    assert main(["--baseline-dir", str(base), "--current-dir", str(cur)]) == 0

    _write_accepted(
        cur,
        """
[[accepted]]
dataset = "adventureworks"
level = "l1"
metric = "latency_ms"
accepted_value = 15.0
reason = "slower but correct"
""",
    )
    assert main(["--baseline-dir", str(base), "--current-dir", str(cur)]) == 1


def test_accepted_entry_does_not_cover_other_pairs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An acceptance is scoped to exactly one (dataset, level, metric)."""
    base, cur = _stage_regression(tmp_path, current_value=0.5)
    _write_accepted(cur, _ACCEPTED_ENTRY.replace('"adventureworks"', '"pagila"'))

    rc = main(["--baseline-dir", str(base), "--current-dir", str(cur)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "RELEASE GATE: REGRESSION DETECTED" in err
    assert "adventureworks/l1" in err


def test_accepted_entry_matching_no_regression_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale or mistyped entry must not silently do nothing."""
    base = tmp_path / "v0.1.0"
    cur = tmp_path / "v0.2.0"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)
    _write_accepted(
        cur, _ACCEPTED_ENTRY.replace("inferred_join_precision", "typo_metric")
    )

    rc = main(["--baseline-dir", str(base), "--current-dir", str(cur)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "typo_metric" in err
    assert "matched no regression" in err


def test_accepted_entry_without_reason_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every acceptance must carry a non-empty justification."""
    base, cur = _stage_regression(tmp_path, current_value=0.5)
    _write_accepted(
        cur,
        _ACCEPTED_ENTRY.replace(
            'reason = "deliberate precision/recall trade-off documented in the changelog"',
            'reason = "  "',
        ),
    )

    rc = main(["--baseline-dir", str(base), "--current-dir", str(cur)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "reason" in err


def test_accepted_file_malformed_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unparseable or incomplete acceptance files block rather than pass."""
    base, cur = _stage_regression(tmp_path, current_value=0.5)

    _write_accepted(cur, "[[accepted]\nthis is not toml")
    rc = main(["--baseline-dir", str(base), "--current-dir", str(cur)])
    assert rc == 1
    assert release_gate._ACCEPTED_FILENAME in capsys.readouterr().err

    _write_accepted(cur, '[[accepted]]\ndataset = "adventureworks"\n')
    rc = main(["--baseline-dir", str(base), "--current-dir", str(cur)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "level" in err or "metric" in err


def test_removed_metric_cannot_be_accepted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A metric that disappeared is always a regression, acceptance or not."""
    base = tmp_path / "v0.1.0"
    cur = tmp_path / "v0.2.0"
    _full_baseline(base, value=0.9)
    _full_baseline(cur, value=0.9)
    _stage(
        cur,
        [
            _result(
                level="l1",
                dataset="adventureworks",
                metrics={"other": Metric(value=1.0, direction="higher_is_better")},
            )
        ],
    )
    _write_accepted(cur, _ACCEPTED_ENTRY)

    rc = main(["--baseline-dir", str(base), "--current-dir", str(cur)])
    err = capsys.readouterr().err

    assert rc == 1
    assert "RELEASE GATE: REGRESSION DETECTED" in err
