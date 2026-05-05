"""``pretensor benchmark`` subsystem — 3-level benchmark harness.

The CLI contract and metric definitions are specified in
``pretensor-ai/pretensor-specs`` at ``docs/specs/benchmark/spec.md``. This
package contains the CLI dispatch / I/O wiring, the results aggregator
(``BenchmarkResult``, ``compare``, JSON / CSV I/O), and the ``compare``
regression-gate. L1 / L2 metric computation and the L3 agent-task runners
land in follow-up changes.
"""

from __future__ import annotations

from pretensor.benchmark.cli import register_benchmark_command
from pretensor.benchmark.results import (
    BenchmarkResult,
    ComparisonError,
    ComparisonReport,
    Direction,
    Metric,
    MetricDiff,
    compare,
    read_json,
    write_csv,
    write_json,
)
from pretensor.benchmark.runner import (
    Dataset,
    RunnerKind,
    run_l1,
    run_l2,
    run_l3,
)

__all__ = [
    "BenchmarkResult",
    "ComparisonError",
    "ComparisonReport",
    "Dataset",
    "Direction",
    "Metric",
    "MetricDiff",
    "RunnerKind",
    "compare",
    "read_json",
    "register_benchmark_command",
    "run_l1",
    "run_l2",
    "run_l3",
    "write_csv",
    "write_json",
]
