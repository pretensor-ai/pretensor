"""Dispatch stubs for the three benchmark levels.

These functions are intentionally minimal: the initial scaffolding lands the
CLI contract and a stable Python entry point, while the metric
implementations ship in follow-up changes.
"""

from __future__ import annotations

import enum
from pathlib import Path


class Dataset(str, enum.Enum):
    """Closed set of benchmark fixture keys.

    Matches ``docs/specs/benchmark/spec.md`` §CLI contract. Unknown values
    are rejected by Typer with exit code 2 before reaching ``run_*``.
    """

    PAGILA = "pagila"
    TPCH = "tpch"
    ANALYTICS_DWH = "analytics_dwh"
    ADVERSARIAL = "adversarial"
    SAAS_MULTITENANT = "saas_multitenant"
    ADVENTUREWORKS = "adventureworks"


class RunnerKind(str, enum.Enum):
    """L3 runner selector — baseline (raw schema) vs pretensor (MCP)."""

    BASELINE = "baseline"
    PRETENSOR = "pretensor"


def run_l1(
    dataset: Dataset,
    out: Path | None,
    graph_dir: Path,
    *,
    embeddings: bool,
) -> None:
    """Run L1 (graph-quality) metrics for ``dataset``.

    Will compute inferred-join precision / recall, cluster-stability
    Jaccard, entity-resolution precision / recall on
    ``tests/fixtures/pairs/``, and role-classification F1 on
    ``adversarial.yaml``. ``embeddings=True`` enables the Layer-A
    embeddings path when the optional extra is installed.
    """
    raise NotImplementedError(
        "L1 metrics not implemented yet "
        "(inferred-join P/R, cluster Jaccard, pairs P/R, role F1)."
    )


def run_l2(
    dataset: Dataset,
    out: Path | None,
    graph_dir: Path,
    *,
    embeddings: bool,
) -> None:
    """Run L2 (MCP-tool-quality) metrics for ``dataset``.

    Will compute ``query`` + ``semantic_search`` Recall@K, ``traverse``
    path correctness, and ``compile_metric`` SQL correctness.
    ``embeddings=True`` enables the Layer-B ``semantic_search`` metric
    when the optional extra is installed.
    """
    raise NotImplementedError(
        "L2 metrics not implemented yet "
        "(query + semantic_search Recall@K, traverse, compile_metric)."
    )


def run_l3(
    dataset: Dataset,
    out: Path | None,
    graph_dir: Path,
    *,
    runner: RunnerKind,
    model: str,
    seed: int | None,
) -> None:
    """Run L3 (agent-task-success) for ``dataset`` via the chosen runner.

    ``runner=RunnerKind.BASELINE`` runs the agent against the raw schema.
    ``runner=RunnerKind.PRETENSOR`` runs the agent through the MCP server.
    """
    if runner is RunnerKind.BASELINE:
        raise NotImplementedError(
            "L3 baseline runner not implemented yet (agent + raw schema, no Pretensor)."
        )
    raise NotImplementedError(
        "L3 pretensor runner not implemented yet (agent + MCP via `pretensor serve`)."
    )
