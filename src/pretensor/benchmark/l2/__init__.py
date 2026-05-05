"""L2 MCP-tool-quality benchmark — metric implementations and runner.

See ``docs/specs/benchmark/spec.md`` §L2 metric definitions for the
contract this module fulfils. Metric values are wrapped in the
``{value, direction}`` envelope from
:mod:`pretensor.benchmark.results`.
"""

from __future__ import annotations

from pretensor.benchmark.l2.metrics import (
    compile_metric_correctness,
    query_recall_at_k,
    semantic_search_recall_at_k,
    top_k_with_ties,
    traverse_correctness,
)
from pretensor.benchmark.l2.runner import run_l2

__all__ = [
    "compile_metric_correctness",
    "query_recall_at_k",
    "run_l2",
    "semantic_search_recall_at_k",
    "top_k_with_ties",
    "traverse_correctness",
]
