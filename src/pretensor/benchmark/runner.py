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

    Computes inferred-join precision / recall against declared FKs,
    cluster-stability Jaccard across two reindex runs, and role-
    classification F1 against ``<dataset>_roles.yaml`` when present.
    ``embeddings=True`` enables the Layer-A embeddings path when the
    optional extra is installed; missing extra logs an install hint
    and falls back to the heuristic-only run.
    """
    # Imported lazily so the heavy intelligence-pipeline dependency
    # graph (Kuzu, igraph, etc.) does not load when other run_*
    # callers import this module.
    from pretensor.benchmark.l1 import run_l1 as _run_l1

    _run_l1(dataset, out, graph_dir, embeddings=embeddings)


def run_l2(
    dataset: Dataset,
    out: Path | None,
    graph_dir: Path,
    *,
    embeddings: bool,
) -> None:
    """Run L2 (MCP-tool-quality) metrics for ``dataset``.

    Computes ``query`` Recall@K, ``traverse`` path correctness, and
    ``compile_metric`` SQL correctness against the fixture's gold data.
    When ``embeddings=True`` and the ``semantic_search`` MCP tool is
    available, also emits ``semantic_search`` Recall@K (Layer-B gate).
    """
    # Imported lazily so the heavy MCP / Kuzu / sqlglot dependency
    # graph does not load when other run_* callers import this module.
    from pretensor.benchmark.l2 import run_l2 as _run_l2

    _run_l2(dataset, out, graph_dir, embeddings=embeddings)


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
    The baseline ignores ``graph_dir``; the pretensor runner spawns
    ``pretensor serve --graph-dir <graph_dir>``.
    """
    if runner is RunnerKind.BASELINE:
        # Imported lazily so the LLM-client / httpx dependency graph does
        # not load when other run_* callers import this module.
        from pretensor.benchmark.l3.runner import run_l3_baseline

        run_l3_baseline(dataset, out, model=model, seed=seed)
        return
    # Lazy import: the pretensor runner pulls in the MCP stdio client and
    # asyncio plumbing; other run_* callers should not pay that cost.
    from pretensor.benchmark.l3.pretensor_runner import run_l3_pretensor

    run_l3_pretensor(dataset, out, graph_dir, model=model, seed=seed)
