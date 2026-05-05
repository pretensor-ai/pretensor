"""Benchmark results aggregator — schema, I/O, and regression-gate comparator.

The JSON schema this module reads/writes is the canonical contract that
every benchmark level (L1, L2, L3) emits and that the release-gate script
consumes. See ``docs/specs/benchmark/spec.md`` §JSON output schema.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "BenchmarkResult",
    "ComparisonError",
    "ComparisonReport",
    "Direction",
    "Metric",
    "MetricDiff",
    "compare",
    "read_json",
    "write_csv",
    "write_json",
]

Direction = Literal["higher_is_better", "lower_is_better"]


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "null"
    return f"{value:+.6g}" if signed else f"{value:.6g}"


@dataclass(frozen=True, slots=True)
class Metric:
    """One metric value plus the direction that defines a regression.

    ``value`` is ``None`` when the metric is unavailable on this run
    (e.g. ``semantic_search`` Recall@K on a build without ``[embeddings]``).
    ``direction`` encodes regression semantics — a drop on a
    ``higher_is_better`` metric is a regression; a rise on a
    ``lower_is_better`` metric is a regression.
    """

    value: float | None
    direction: Direction

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "direction": self.direction}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metric:
        return cls(value=data["value"], direction=data["direction"])


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One benchmark run's full output — matches the spec's JSON schema.

    ``extra`` is a free-form per-level envelope for fields that don't
    belong in ``metrics`` or ``per_item`` but must be captured for
    reproducibility — e.g. L3 records the LLM model, temperature, seed,
    and prompt hash here. L1 / L2 leave it empty. The spec's JSON-schema
    section ("downstream consumers must tolerate additional keys")
    sanctions adding new top-level keys this way.
    """

    level: str
    dataset: str
    pretensor_version: str
    embeddings_enabled: bool
    ran_at: str
    fixture_sha: str
    metrics: dict[str, Metric]
    per_item: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "level": self.level,
            "dataset": self.dataset,
            "pretensor_version": self.pretensor_version,
            "embeddings_enabled": self.embeddings_enabled,
            "ran_at": self.ran_at,
            "fixture_sha": self.fixture_sha,
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
            "per_item": list(self.per_item),
            "notes": list(self.notes),
        }
        if self.extra:
            out["extra"] = dict(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        return cls(
            level=data["level"],
            dataset=data["dataset"],
            pretensor_version=data["pretensor_version"],
            embeddings_enabled=data["embeddings_enabled"],
            ran_at=data["ran_at"],
            fixture_sha=data["fixture_sha"],
            metrics={name: Metric.from_dict(m) for name, m in data["metrics"].items()},
            per_item=list(data["per_item"]),
            notes=list(data.get("notes", [])),
            extra=dict(data.get("extra", {})),
        )


def write_json(result: BenchmarkResult, path: Path) -> None:
    """Serialize ``result`` to ``path`` as deterministic, pretty-printed JSON.

    Output is byte-stable for identical inputs: keys are sorted, indent is 2
    spaces, and the file ends with a single trailing newline.
    """
    text = json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> BenchmarkResult:
    """Parse a JSON file into a ``BenchmarkResult``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkResult.from_dict(data)


class ComparisonError(ValueError):
    """Raised when two ``BenchmarkResult`` inputs cannot be compared."""


@dataclass(frozen=True, slots=True)
class MetricDiff:
    """One metric's before/after delta with its regression direction."""

    name: str
    baseline: float | None
    current: float | None
    direction: Direction
    delta: float | None


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Result of comparing two ``BenchmarkResult`` instances."""

    has_regression: bool = False
    regressions: list[MetricDiff] = field(default_factory=list)
    improvements: list[MetricDiff] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def format_diff(self) -> str:
        """Render a human-readable diff block suitable for stderr output."""
        lines: list[str] = []
        if self.has_regression:
            lines.append("REGRESSION DETECTED")
        else:
            lines.append("no regression")
        if self.regressions:
            lines.append("")
            lines.append("Regressions:")
            for d in self.regressions:
                arrow = "↓" if d.direction == "higher_is_better" else "↑"
                lines.append(
                    f"  {d.name} ({d.direction}): "
                    f"{_fmt(d.baseline)} → {_fmt(d.current)} "
                    f"(Δ{_fmt(d.delta, signed=True)}) {arrow}"
                )
        if self.removed:
            lines.append("")
            lines.append("Metrics removed (present in baseline, missing in current):")
            for name in self.removed:
                lines.append(f"  {name}")
        if self.added:
            lines.append("")
            lines.append("New metrics in current (not in baseline):")
            for name in self.added:
                lines.append(f"  {name}")
        if self.improvements:
            lines.append("")
            lines.append("Improvements:")
            for d in self.improvements:
                lines.append(
                    f"  {d.name}: {_fmt(d.baseline)} → {_fmt(d.current)} "
                    f"(Δ{_fmt(d.delta, signed=True)})"
                )
        return "\n".join(lines) + "\n"


def compare(
    baseline: BenchmarkResult,
    current: BenchmarkResult,
    *,
    tolerance: float = 1e-6,
) -> ComparisonReport:
    """Compare two benchmark results and classify metric changes.

    Raises ``ComparisonError`` if the two inputs disagree on ``dataset`` or
    ``level`` — comparing across datasets or levels is meaningless. A metric
    present in ``baseline`` but missing in ``current`` is a regression
    (``removed``). A metric new in ``current`` is recorded under ``added``
    but never counts as a regression. Metrics whose value is ``None`` on
    either side are skipped (no diff emitted).
    """
    if baseline.dataset != current.dataset:
        raise ComparisonError(
            f"cannot compare across datasets: baseline={baseline.dataset!r}, "
            f"current={current.dataset!r}"
        )
    if baseline.level != current.level:
        raise ComparisonError(
            f"cannot compare across levels: baseline={baseline.level!r}, "
            f"current={current.level!r}"
        )

    regressions: list[MetricDiff] = []
    improvements: list[MetricDiff] = []
    removed: list[str] = []
    added: list[str] = sorted(set(current.metrics) - set(baseline.metrics))

    for name in sorted(baseline.metrics):
        base_metric = baseline.metrics[name]
        cur_metric = current.metrics.get(name)
        if cur_metric is None:
            removed.append(name)
            continue
        if base_metric.value is None or cur_metric.value is None:
            continue
        delta = cur_metric.value - base_metric.value
        diff = MetricDiff(
            name=name,
            baseline=base_metric.value,
            current=cur_metric.value,
            direction=base_metric.direction,
            delta=delta,
        )
        if base_metric.direction == "higher_is_better":
            if delta < -tolerance:
                regressions.append(diff)
            elif delta > tolerance:
                improvements.append(diff)
        else:  # lower_is_better
            if delta > tolerance:
                regressions.append(diff)
            elif delta < -tolerance:
                improvements.append(diff)

    return ComparisonReport(
        has_regression=bool(regressions or removed),
        regressions=regressions,
        improvements=improvements,
        added=added,
        removed=removed,
    )


_CSV_META_COLUMNS = (
    "level",
    "dataset",
    "pretensor_version",
    "embeddings_enabled",
    "ran_at",
)


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
    """Write one row per ``BenchmarkResult`` as a flat CSV for spreadsheet review.

    Metric columns are the sorted union of metric names across ``results``.
    A row missing a metric (or whose value is ``None``) writes an empty cell.
    Metric ``direction`` is JSON-only metadata and is omitted from CSV;
    reviewers consult the JSON for direction.
    """
    metric_names = sorted({name for r in results for name in r.metrics})
    columns = list(_CSV_META_COLUMNS) + metric_names
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in results:
            row: dict[str, Any] = {
                "level": r.level,
                "dataset": r.dataset,
                "pretensor_version": r.pretensor_version,
                "embeddings_enabled": r.embeddings_enabled,
                "ran_at": r.ran_at,
            }
            for name in metric_names:
                metric = r.metrics.get(name)
                row[name] = (
                    "" if metric is None or metric.value is None else metric.value
                )
            writer.writerow(row)
