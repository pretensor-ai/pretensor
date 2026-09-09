"""Release gate for ``.github/workflows/release.yml``.

A tag cut may publish to PyPI only after this script exits 0. The gate
runs L1 and L2 against every release-gating dataset, compares the
resulting JSON against the previous tag's stored results under
``tests/benchmark/results/<previous-tag>/``, and refuses to publish on
any regression. L3 is informational and is never evaluated here.

See ``docs/specs/benchmark/spec.md`` §Release-gate rules and
``docs/contracts/architecture.md`` Invariant #13 for the contract.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pretensor.benchmark.results import (
    ComparisonError,
    ComparisonReport,
    MetricDiff,
    compare,
    read_json,
)
from pretensor.benchmark.runner import Dataset

__all__ = ["main"]


_DEFAULT_RESULTS_BASE = Path("tests/benchmark/results")
"""Root directory for ``<tag>/`` subdirectories, relative to the repo root."""

_RELEASE_DATASETS: tuple[Dataset, ...] = (
    Dataset.PAGILA,
    Dataset.TPCH,
    Dataset.ADVENTUREWORKS,
)
"""Datasets the gate evaluates by default — see spec §Release-gate rules.

Other fixtures (``analytics_dwh``, ``adversarial``, ``saas_multitenant``)
exist in the repo but are exercises for L1/L2 internals; the spec calls
out Pagila, TPC-H, and AdventureWorks as the three release-gating
datasets that ship with documented external provenance.
"""

_GATING_LEVELS: tuple[str, ...] = ("l1", "l2")
"""Levels the gate evaluates. L3 is informational and never enters here."""

_DEFAULT_TOLERANCE = 1e-6
"""Single global tolerance reused from ``compare()``.

When a per-metric tolerance table arrives (derived from the null-client
determinism leg), introduce the lookup at the call site in
:func:`_evaluate_pair` then. Until then, this single threshold suffices
and the gate avoids unused indirection layers.
"""

_DEFAULT_GRAPH_DIR = Path(".pretensor")
"""Where ``run_l1`` / ``run_l2`` look for an indexed graph."""


_ACCEPTED_FILENAME = "accepted-regressions.toml"
"""Per-candidate-tag file that documents deliberate metric regressions.

Lives next to the candidate's results (``tests/benchmark/results/<tag>/``)
and is committed *before* the tag is cut. Each ``[[accepted]]`` entry names
one ``(dataset, level, metric)`` and the value the release is willing to
ship at. The gate treats the entry as a floor (or ceiling, for
``lower_is_better`` metrics), never as a blanket waiver: a run that is
worse than ``accepted_value`` still blocks, and an entry that matches no
observed regression is an error so stale or mistyped entries cannot pass
silently. Every entry must carry a non-empty ``reason``.
"""


@dataclass(frozen=True, slots=True)
class _AcceptedRegression:
    """One documented, deliberately accepted metric regression."""

    dataset: str
    level: str
    metric: str
    accepted_value: float
    reason: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.dataset, self.level, self.metric)


_ACCEPTED_REQUIRED_KEYS: tuple[str, ...] = (
    "dataset",
    "level",
    "metric",
    "accepted_value",
    "reason",
)


def _load_accepted_regressions(path: Path) -> tuple[_AcceptedRegression, ...]:
    """Parse ``path`` into acceptance entries; ``()`` when the file is absent.

    Raises :class:`ValueError` on any structural problem so the caller can
    block the release with a message that names the file — a malformed
    acceptance must never degrade into "no acceptances, fail as usual"
    *or* into a pass.
    """
    if not path.is_file():
        return ()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ValueError(f"cannot parse {path}: {exc}") from exc

    raw_entries = data.get("accepted", [])
    if not isinstance(raw_entries, list):
        raise ValueError(f"{path}: 'accepted' must be an array of tables")

    entries: list[_AcceptedRegression] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(raw_entries):
        where = f"{path}: [[accepted]] entry #{index + 1}"
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be a table")
        missing = [k for k in _ACCEPTED_REQUIRED_KEYS if k not in raw]
        if missing:
            raise ValueError(
                f"{where} is missing required key(s): {', '.join(missing)}"
            )
        for key in ("dataset", "level", "metric", "reason"):
            if not isinstance(raw[key], str):
                raise ValueError(f"{where}: {key!r} must be a string")
        value = raw["accepted_value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: 'accepted_value' must be a number")
        if not raw["reason"].strip():
            raise ValueError(f"{where}: 'reason' must not be empty")
        entry = _AcceptedRegression(
            dataset=raw["dataset"],
            level=raw["level"],
            metric=raw["metric"],
            accepted_value=float(value),
            reason=raw["reason"].strip(),
        )
        if entry.key in seen:
            raise ValueError(
                f"{where}: duplicate entry for {entry.dataset}/{entry.level} "
                f"{entry.metric!r}"
            )
        seen.add(entry.key)
        entries.append(entry)
    return tuple(entries)


def _meets_accepted_value(
    diff: MetricDiff, entry: _AcceptedRegression, *, tolerance: float
) -> bool:
    """True when ``diff.current`` is no worse than the documented floor/ceiling."""
    if diff.current is None:
        return False
    if diff.direction == "higher_is_better":
        return diff.current >= entry.accepted_value - tolerance
    return diff.current <= entry.accepted_value + tolerance


# ---------------------------------------------------------------------------
# Tag + path resolution (module-level seams so tests can monkeypatch)
# ---------------------------------------------------------------------------


def _resolve_previous_tag() -> str | None:
    """Return the most recent tag before HEAD, or ``None`` if no prior tag exists.

    Uses ``git describe --abbrev=0 --tags HEAD^`` per the issue body. Any
    git failure (no parent commit, no prior tag, not a git repo) collapses
    to ``None`` so callers can branch cleanly on first-tag behaviour.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--abbrev=0", "--tags", "HEAD^"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    tag = result.stdout.strip()
    return tag or None


def _resolve_candidate_tag() -> str:
    """Return the tag for the candidate release.

    GitHub Actions sets ``GITHUB_REF_NAME`` to the tag on a ``push: tags``
    trigger — preferred because it's authoritative for the running
    workflow. Local invocation falls back to
    ``git describe --tags --exact-match HEAD`` so an operator can dry-run
    the gate at a tag they've already cut.
    """
    env_tag = os.environ.get("GITHUB_REF_NAME", "").strip()
    if env_tag:
        return env_tag
    result = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Unable to determine candidate tag: GITHUB_REF_NAME unset and "
            "HEAD does not point at a tag. Pass --current-dir to run "
            "the gate against a pre-staged result directory."
        )
    return result.stdout.strip()


def _resolve_results_dir(tag: str, results_base: Path) -> Path:
    """Map a tag to its per-tag results directory under ``results_base``."""
    return results_base / tag


# ---------------------------------------------------------------------------
# Benchmark invocation seam
# ---------------------------------------------------------------------------


def _run_benchmark(
    level: str,
    dataset: Dataset,
    out: Path,
    *,
    graph_dir: Path,
    embeddings: bool,
) -> None:
    """Run one ``(level, dataset)`` benchmark, writing JSON to ``out``.

    Imports ``run_l1`` / ``run_l2`` lazily so a test that monkeypatches
    this seam never pays the heavy dependency-graph cost (Kuzu, igraph,
    sqlglot, MCP SDK).
    """
    if level == "l1":
        from pretensor.benchmark.runner import run_l1

        run_l1(dataset, out, graph_dir, embeddings=embeddings)
        return
    if level == "l2":
        from pretensor.benchmark.runner import run_l2

        run_l2(dataset, out, graph_dir, embeddings=embeddings)
        return
    raise ValueError(f"unsupported gating level: {level!r}")


# ---------------------------------------------------------------------------
# Per-pair comparison
# ---------------------------------------------------------------------------


_PairStatus = Literal["pass", "accepted", "regression", "skipped", "error"]


@dataclass(frozen=True, slots=True)
class _PairOutcome:
    """One ``(dataset, level)`` evaluation's verdict."""

    dataset: str
    level: str
    status: _PairStatus
    detail: str = ""
    accepted_metrics: tuple[str, ...] = ()
    """Metric names whose regression was covered by an acceptance entry."""
    unmet_metrics: tuple[str, ...] = ()
    """Metric names that had an acceptance entry but regressed past it."""

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.level}"


def _evaluate_pair(
    *,
    dataset: Dataset,
    level: str,
    baseline_dir: Path,
    current_dir: Path,
    run_benchmarks: bool,
    graph_dir: Path,
    embeddings: bool,
    accepted: Sequence[_AcceptedRegression] = (),
) -> _PairOutcome:
    """Run (optionally) and compare one ``(dataset, level)`` pair."""
    fname = f"{dataset.value}-{level}.json"
    baseline_path = baseline_dir / fname
    current_path = current_dir / fname

    if run_benchmarks:
        try:
            current_dir.mkdir(parents=True, exist_ok=True)
            _run_benchmark(
                level,
                dataset,
                current_path,
                graph_dir=graph_dir,
                embeddings=embeddings,
            )
        except Exception as exc:  # noqa: BLE001 — surface ANY runner fault
            return _PairOutcome(
                dataset=dataset.value,
                level=level,
                status="error",
                detail=f"failed to produce current result: {exc}",
            )

    if not baseline_path.is_file():
        # New dataset added between tags — not a regression.
        return _PairOutcome(
            dataset=dataset.value,
            level=level,
            status="skipped",
            detail=f"no baseline at {baseline_path}",
        )
    if not current_path.is_file():
        return _PairOutcome(
            dataset=dataset.value,
            level=level,
            status="error",
            detail=f"no current result at {current_path}",
        )

    try:
        baseline_result = read_json(baseline_path)
        current_result = read_json(current_path)
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        return _PairOutcome(
            dataset=dataset.value,
            level=level,
            status="error",
            detail=f"failed to read result JSON: {exc}",
        )

    try:
        report = compare(
            baseline_result,
            current_result,
            tolerance=_DEFAULT_TOLERANCE,
        )
    except ComparisonError as exc:
        return _PairOutcome(
            dataset=dataset.value,
            level=level,
            status="error",
            detail=str(exc),
        )

    if not report.has_regression:
        return _PairOutcome(dataset=dataset.value, level=level, status="pass")

    return _apply_acceptances(
        report,
        dataset=dataset.value,
        level=level,
        accepted=accepted,
        tolerance=_DEFAULT_TOLERANCE,
    )


def _apply_acceptances(
    report: ComparisonReport,
    *,
    dataset: str,
    level: str,
    accepted: Sequence[_AcceptedRegression],
    tolerance: float,
) -> _PairOutcome:
    """Split a regressed report into accepted vs. still-blocking regressions.

    A removed metric is never acceptable: an acceptance names a value the
    metric may ship at, and a metric that no longer exists has none.
    """
    by_key = {e.key: e for e in accepted}
    accepted_lines: list[str] = []
    accepted_names: list[str] = []
    unmet_names: list[str] = []
    remaining: list[MetricDiff] = []
    not_met: list[str] = []
    for diff in report.regressions:
        entry = by_key.get((dataset, level, diff.name))
        if entry is None:
            remaining.append(diff)
            continue
        if _meets_accepted_value(diff, entry, tolerance=tolerance):
            accepted_names.append(diff.name)
            accepted_lines.append(
                f"{diff.name} ({diff.direction}): "
                f"{diff.baseline:.6g} → {diff.current:.6g} "
                f"(accepted_value {entry.accepted_value:.6g}) — {entry.reason}"
            )
        else:
            remaining.append(diff)
            unmet_names.append(diff.name)
            not_met.append(
                f"{diff.name}: current {diff.current:.6g} is worse than "
                f"accepted_value {entry.accepted_value:.6g} in {_ACCEPTED_FILENAME}"
            )

    if remaining or report.removed:
        filtered = ComparisonReport(
            has_regression=True,
            regressions=remaining,
            improvements=report.improvements,
            added=report.added,
            removed=report.removed,
        )
        detail = filtered.format_diff()
        if not_met:
            detail += "\nAcceptance not met:\n" + "".join(
                f"  {line}\n" for line in not_met
            )
        return _PairOutcome(
            dataset=dataset,
            level=level,
            status="regression",
            detail=detail,
            accepted_metrics=tuple(accepted_names),
            unmet_metrics=tuple(unmet_names),
        )
    return _PairOutcome(
        dataset=dataset,
        level=level,
        status="accepted",
        detail="\n".join(accepted_lines),
        accepted_metrics=tuple(accepted_names),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_datasets(value: str) -> tuple[Dataset, ...]:
    """Argparse ``type=`` for ``--datasets``.

    Empty / whitespace-only string → release-gating defaults. Unknown
    dataset names raise :class:`argparse.ArgumentTypeError`, which
    argparse converts into a uniform ``error: argument --datasets: …``
    message (with the standard usage line) — same surface as every
    other validation failure in this CLI.
    """
    if not value or not value.strip():
        return _RELEASE_DATASETS
    out: list[Dataset] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(Dataset(raw))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"unknown dataset {raw!r}: {exc}") from exc
    return tuple(out) or _RELEASE_DATASETS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pretensor.benchmark.release_gate",
        description=(
            "Block a tag publication on L1/L2 regression vs the previous tag's "
            "stored benchmark results."
        ),
    )
    parser.add_argument(
        "--results-base",
        type=Path,
        default=_DEFAULT_RESULTS_BASE,
        help="Root directory containing per-tag result subdirectories. "
        "Default: tests/benchmark/results.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Override the baseline directory (skips git previous-tag resolution).",
    )
    parser.add_argument(
        "--current-dir",
        type=Path,
        default=None,
        help="Override the candidate-tag directory (skips running benchmarks).",
    )
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=_RELEASE_DATASETS,
        help=("Comma-separated dataset list (default: pagila,tpch,adventureworks)."),
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=_DEFAULT_GRAPH_DIR,
        help=(
            "Graph directory passed through to run_l1 / run_l2. Default: .pretensor."
        ),
    )
    parser.add_argument(
        "--embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the embeddings-on leg of L1/L2 (requires the 'embeddings' "
            "extra). Default: --no-embeddings."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    datasets: tuple[Dataset, ...] = args.datasets
    results_base: Path = args.results_base

    # Resolve baseline directory.
    baseline_dir: Path
    if args.baseline_dir is not None:
        baseline_dir = args.baseline_dir
    else:
        previous_tag = _resolve_previous_tag()
        if previous_tag is None:
            print(
                "release-gate: no previous tag found — no baseline to compare. "
                "First-tag path; treating as pass.",
                file=sys.stderr,
            )
            return 0
        baseline_dir = _resolve_results_dir(previous_tag, results_base)
        # A previous tag exists but its archive directory is entirely
        # absent — most likely the post-publish archive PR for that tag
        # never landed. Treating this as "skipped, all clear" would
        # masquerade as a first-tag run and silently let regressions
        # through; refuse instead and force the operator to investigate.
        if not baseline_dir.is_dir():
            print(
                f"release-gate: previous tag {previous_tag!r} resolved but "
                f"its baseline directory {baseline_dir} is absent. The "
                "post-publish archive job for that tag likely failed or "
                "the archive PR was never merged. Refusing to proceed.",
                file=sys.stderr,
            )
            return 1

    # Resolve candidate directory + decide whether to run benchmarks.
    current_dir: Path
    run_benchmarks: bool
    if args.current_dir is not None:
        current_dir = args.current_dir
        run_benchmarks = False
    else:
        candidate_tag = _resolve_candidate_tag()
        current_dir = _resolve_results_dir(candidate_tag, results_base)
        run_benchmarks = True

    accepted_path = current_dir / _ACCEPTED_FILENAME
    try:
        accepted = _load_accepted_regressions(accepted_path)
    except ValueError as exc:
        print(f"release-gate: invalid acceptance file — {exc}", file=sys.stderr)
        return 1

    outcomes: list[_PairOutcome] = []
    for dataset in datasets:
        for level in _GATING_LEVELS:
            outcomes.append(
                _evaluate_pair(
                    dataset=dataset,
                    level=level,
                    baseline_dir=baseline_dir,
                    current_dir=current_dir,
                    run_benchmarks=run_benchmarks,
                    graph_dir=args.graph_dir,
                    embeddings=args.embeddings,
                    accepted=accepted,
                )
            )

    # An acceptance that matched nothing is either stale (the regression
    # went away — delete the entry) or mistyped (it would have silently
    # waived nothing while the real regression blocks). Either way, block
    # so the file stays an exact record of what shipped.
    used = {
        (o.dataset, o.level, name)
        for o in outcomes
        for name in (*o.accepted_metrics, *o.unmet_metrics)
    }
    for entry in accepted:
        if entry.key not in used:
            outcomes.append(
                _PairOutcome(
                    dataset=entry.dataset,
                    level=entry.level,
                    status="error",
                    detail=(
                        f"{accepted_path} entry for metric {entry.metric!r} "
                        "matched no regression — remove it or fix the "
                        "dataset/level/metric"
                    ),
                )
            )

    return _emit_summary(outcomes)


def _emit_summary(outcomes: Sequence[_PairOutcome]) -> int:
    """Print the verdict to stderr; return 0 on no regression, 1 otherwise."""
    regressions = [o for o in outcomes if o.status == "regression"]
    errors = [o for o in outcomes if o.status == "error"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    accepted = [o for o in outcomes if o.status == "accepted"]
    passed = [o for o in outcomes if o.status == "pass"]

    lines: list[str] = []
    if regressions or errors:
        lines.append("RELEASE GATE: REGRESSION DETECTED")
    elif accepted:
        lines.append(
            "release gate: all comparisons within tolerance or explicitly accepted"
        )
    else:
        lines.append("release gate: all comparisons within tolerance")

    if regressions:
        lines.append("")
        lines.append("Regressions:")
        for o in regressions:
            lines.append(f"  {o.label}:")
            for diff_line in o.detail.splitlines():
                lines.append(f"    {diff_line}")
    if errors:
        lines.append("")
        lines.append("Errors:")
        for o in errors:
            lines.append(f"  {o.label}: {o.detail}")
    if accepted:
        lines.append("")
        lines.append(f"Accepted regressions (documented in {_ACCEPTED_FILENAME}):")
        for o in accepted:
            lines.append(f"  {o.label}:")
            for diff_line in o.detail.splitlines():
                lines.append(f"    {diff_line}")
    if skipped:
        lines.append("")
        lines.append("Skipped (no baseline — new dataset since previous tag):")
        for o in skipped:
            lines.append(f"  {o.label}: {o.detail}")
    if passed and not (regressions or errors):
        lines.append("")
        lines.append(f"Passed: {len(passed)} comparison(s)")

    print("\n".join(lines), file=sys.stderr)
    return 0 if not (regressions or errors) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
