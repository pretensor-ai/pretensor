"""Run clustering, labeling, and join-path precomputation after a graph build.

The pipeline is expressed as a sequence of named :class:`PipelineStep` objects
executed by :class:`PipelineRunner`.  Cloud can inject additional steps (e.g.
``llm_refine``, ``feedback_score``, ``semantic_propose``) between the built-in
OSS steps by calling :func:`build_oss_pipeline` and registering extra steps
before calling ``runner.run(ctx)``.

OSS step order (resolved from dependencies):
    classify → cluster → label → join_paths
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

from pretensor.config import GraphConfig
from pretensor.core.store import KuzuStore
from pretensor.intelligence.cluster_labeler import ClusterLabeler
from pretensor.intelligence.clustering import Cluster, ClusteringEngine
from pretensor.intelligence.graph_export import GraphExporter
from pretensor.intelligence.join_paths import JoinPathEngine
from pretensor.intelligence.schema_classification import (
    classify_database_tables_async,
    compute_cluster_schema_patterns,
    load_fk_reference_pairs,
)
from pretensor.intelligence.steps import PipelineContext, PipelineRunner, PipelineStep

if TYPE_CHECKING:
    from pretensor.config import PretensorConfig

__all__ = [
    "run_intelligence_layer",
    "run_intelligence_layer_sync",
    "build_oss_pipeline",
]

logger = logging.getLogger(__name__)

_PROFILE_INDEX = os.environ.get("PRETENSOR_PROFILE_INDEX", "").lower() not in (
    "",
    "0",
    "false",
    "no",
)

# ---------------------------------------------------------------------------
# Context keys — stable names shared between steps
# ---------------------------------------------------------------------------
_CTX_STORE = "store"
_CTX_DATABASE_KEY = "database_key"
_CTX_CONFIG = "config"
_CTX_GRAPH = "graph"
_CTX_ROLE_BY_TABLE = "role_by_table"
_CTX_CLUSTERS = "clusters"
_CTX_PATTERNS = "patterns"


# ---------------------------------------------------------------------------
# Built-in OSS steps
# ---------------------------------------------------------------------------


class _ClassifyStep:
    """Classify each table in the database by role (fact/dimension/bridge/lookup)."""

    name = "classify"
    dependencies: list[str] = []

    async def execute(self, ctx: PipelineContext) -> None:
        store: KuzuStore = ctx.get(_CTX_STORE)
        database_key: str = ctx.get(_CTX_DATABASE_KEY)

        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        role_by_table = await classify_database_tables_async(store, database_key)
        if _PROFILE_INDEX:
            print(
                f"[profile] intelligence.classify_database_tables: {(time.perf_counter() - _t) * 1000:.0f}ms",
                flush=True,
            )
        ctx.set(_CTX_ROLE_BY_TABLE, role_by_table)


class _ClusterStep:
    """Run community detection on the FK graph to group tables into clusters."""

    name = "cluster"
    dependencies: list[str] = ["classify"]

    async def execute(self, ctx: PipelineContext) -> None:
        store: KuzuStore = ctx.get(_CTX_STORE)
        database_key: str = ctx.get(_CTX_DATABASE_KEY)
        cfg: GraphConfig = ctx.get(_CTX_CONFIG)
        graph = ctx.get(_CTX_GRAPH)

        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        clusters: list[Cluster] = ClusteringEngine(cfg).cluster(graph)
        if _PROFILE_INDEX:
            print(
                f"[profile] intelligence.clustering: {(time.perf_counter() - _t) * 1000:.0f}ms ({len(clusters)} clusters)",
                flush=True,
            )

        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        fk_pairs = load_fk_reference_pairs(store, database_key)
        role_by_table = ctx.get(_CTX_ROLE_BY_TABLE)
        patterns = compute_cluster_schema_patterns(clusters, role_by_table, fk_pairs)
        if _PROFILE_INDEX:
            print(
                f"[profile] intelligence.cluster_schema_patterns: {(time.perf_counter() - _t) * 1000:.0f}ms",
                flush=True,
            )

        ctx.set(_CTX_CLUSTERS, clusters)
        ctx.set(_CTX_PATTERNS, patterns)


class _LabelStep:
    """Assign domain labels to clusters and persist them to Kuzu."""

    name = "label"
    dependencies: list[str] = ["cluster"]

    async def execute(self, ctx: PipelineContext) -> None:
        store: KuzuStore = ctx.get(_CTX_STORE)
        database_key: str = ctx.get(_CTX_DATABASE_KEY)
        clusters: list[Cluster] = ctx.get(_CTX_CLUSTERS)
        patterns = ctx.get(_CTX_PATTERNS)

        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        labeler = ClusterLabeler(store)
        await labeler.label_and_persist(
            clusters, database_key, cluster_schema_patterns=patterns
        )
        if _PROFILE_INDEX:
            print(
                f"[profile] intelligence.cluster_labeling: {(time.perf_counter() - _t) * 1000:.0f}ms",
                flush=True,
            )


class _JoinPathsStep:
    """Precompute join paths between table pairs within and across clusters."""

    name = "join_paths"
    dependencies: list[str] = ["label"]

    async def execute(self, ctx: PipelineContext) -> None:
        store: KuzuStore = ctx.get(_CTX_STORE)
        database_key: str = ctx.get(_CTX_DATABASE_KEY)
        cfg: GraphConfig = ctx.get(_CTX_CONFIG)

        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        JoinPathEngine(store).precompute(database_key, cfg)
        if _PROFILE_INDEX:
            print(
                f"[profile] intelligence.join_paths_precompute: {(time.perf_counter() - _t) * 1000:.0f}ms",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Public factory and orchestration helpers
# ---------------------------------------------------------------------------

_OSS_STEPS: list[PipelineStep] = [
    _ClassifyStep(),
    _ClusterStep(),
    _LabelStep(),
    _JoinPathsStep(),
]


def build_oss_pipeline() -> PipelineRunner:
    """Return a :class:`PipelineRunner` pre-loaded with the four OSS steps.

    Cloud can append additional steps via :meth:`PipelineRunner.register`
    before calling ``runner.run(ctx)``.

    Returns:
        A :class:`PipelineRunner` with steps: classify → cluster → label →
        join_paths.
    """
    return PipelineRunner(list(_OSS_STEPS))


async def run_intelligence_layer(
    store: KuzuStore,
    database_key: str,
    *,
    config: GraphConfig | PretensorConfig | None = None,
) -> None:
    """Clear prior intelligence rows, cluster tables, label, precompute join paths.

    Delegates to :func:`build_oss_pipeline` so external callers get identical
    semantics while Cloud can extend the pipeline via :func:`build_oss_pipeline`
    directly.

    Args:
        store: The Kuzu graph store.
        database_key: Logical database name (``SchemaTable.database``).
        config: Optional tuning overrides; accepts :class:`GraphConfig` or
            :class:`PretensorConfig`. When a :class:`PretensorConfig` is given,
            its ``graph`` sub-field is used for clustering/join-path tuning.
            Defaults to :class:`GraphConfig` with OSS defaults.
    """
    from pretensor.config import PretensorConfig

    if isinstance(config, PretensorConfig):
        cfg = config.graph
    else:
        cfg = config or GraphConfig()
    store.ensure_schema()
    store.clear_intelligence_artifacts()

    _t = time.perf_counter() if _PROFILE_INDEX else 0.0
    exporter = GraphExporter(store)
    graph = exporter.to_igraph(database_key, config=cfg)
    if _PROFILE_INDEX:
        print(
            f"[profile] intelligence.to_igraph: {(time.perf_counter() - _t) * 1000:.0f}ms (vcount={graph.vcount()}, ecount={graph.ecount()})",
            flush=True,
        )
    if graph.vcount() == 0:
        logger.info(
            "Intelligence layer skipped: no tables for database %s", database_key
        )
        return

    ctx = PipelineContext(
        **{
            _CTX_STORE: store,
            _CTX_DATABASE_KEY: database_key,
            _CTX_CONFIG: cfg,
            _CTX_GRAPH: graph,
        }
    )
    runner = build_oss_pipeline()
    await runner.run(ctx)


def run_intelligence_layer_sync(
    store: KuzuStore,
    database_key: str,
    *,
    config: GraphConfig | PretensorConfig | None = None,
) -> None:
    """Sync wrapper for :func:`run_intelligence_layer` (CLI / builder)."""
    asyncio.run(run_intelligence_layer(store, database_key, config=config))
