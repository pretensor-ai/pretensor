"""Run clustering, labeling, and join-path precomputation after a graph build.

The pipeline is expressed as a sequence of named :class:`PipelineStep` objects
executed by :class:`PipelineRunner`.  Plugins can inject additional steps (e.g.
``llm_refine``, ``feedback_score``, ``semantic_propose``) between the built-in
OSS steps by calling :func:`build_oss_pipeline` and registering extra steps
before calling ``runner.run(ctx)``.

OSS step order (resolved from dependencies):
    embedding_index → classify → cluster → label → join_paths
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from pretensor.config import GraphConfig
from pretensor.core.store import KuzuStore
from pretensor.intelligence.cluster_labeler import ClusterLabeler
from pretensor.intelligence.clustering import Cluster, ClusteringEngine
from pretensor.intelligence.embeddings import embeddings_disabled_via_env
from pretensor.intelligence.graph_export import GraphExporter
from pretensor.intelligence.join_paths import JoinPathEngine
from pretensor.intelligence.schema_classification import (
    classify_database_tables_async,
    compute_cluster_schema_patterns,
    load_fk_reference_pairs,
)
from pretensor.intelligence.steps import PipelineContext, PipelineRunner, PipelineStep
from pretensor.intelligence.steps_embedding import EmbeddingIndexStep

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
# Canonical context keys live in ``steps.py`` so step modules can import
# the same constants instead of defining sibling string literals (which
# would silently no-op on drift). Re-exported here for backward compat
# with any tests / cloud extensions that imported them from this module.
from pretensor.intelligence.steps import (  # noqa: E402  (post-imports to avoid cycle re-order churn)
    _CTX_CLUSTERS,
    _CTX_CONFIG,
    _CTX_DATABASE_KEY,
    _CTX_EMBEDDINGS_CONFIG,
    _CTX_EMBEDDINGS_PRECOMPUTED,
    _CTX_GRAPH,
    _CTX_PATTERNS,
    _CTX_ROLE_BY_TABLE,
    _CTX_STORE,
)

# ---------------------------------------------------------------------------
# Built-in OSS steps
# ---------------------------------------------------------------------------


class _ClassifyStep:
    """Classify each table in the database by role (fact/dimension/bridge/lookup)."""

    name = "classify"
    # Depends on embedding_index so the optional role vote (role_weight > 0)
    # reads vectors computed in THIS run; without the edge the vote would
    # silently see an empty store on the first index and degrade to the
    # heuristic-only path until the next reindex.
    dependencies: list[str] = ["embedding_index"]

    async def execute(self, ctx: PipelineContext) -> None:
        store: KuzuStore = ctx.get(_CTX_STORE)
        database_key: str = ctx.get(_CTX_DATABASE_KEY)
        emb_cfg = ctx.get(_CTX_EMBEDDINGS_CONFIG)

        # When the user opts into the role-classification embedding vote,
        # instantiate a real client so the classifier can compute per-role
        # centroids + per-table votes.  Default keeps role_weight=0 and
        # embedding_client=None → heuristic-only path.
        embedding_client = None
        role_weight = 0.0
        if emb_cfg is not None and emb_cfg.role_weight > 0.0:
            from pretensor.intelligence.embeddings import (
                get_default_embedding_client,
            )

            embedding_client = get_default_embedding_client()
            role_weight = emb_cfg.role_weight

        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        role_by_table = await classify_database_tables_async(
            store,
            database_key,
            role_weight=role_weight,
            embedding_client=embedding_client,
        )
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
    """Assign domain labels to clusters and persist them to Kuzu.

    Declares ``embedding_index`` as a dependency so the topological
    runner orders the embedding writes before the labeler reads them.
    Without this, correctness would rely on insertion-order tie-breaking
    inside the runner — fragile to a future scheduling change.
    """

    name = "label"
    dependencies: list[str] = ["cluster", "embedding_index"]

    async def execute(self, ctx: PipelineContext) -> None:
        store: KuzuStore = ctx.get(_CTX_STORE)
        database_key: str = ctx.get(_CTX_DATABASE_KEY)
        clusters: list[Cluster] = ctx.get(_CTX_CLUSTERS)
        patterns = ctx.get(_CTX_PATTERNS)
        emb_cfg = ctx.get(_CTX_EMBEDDINGS_CONFIG)

        # When the user opts into clustering blend, thread an embedding
        # resolver into the labeler so per-cluster centroids can act as a
        # tiebreaker after role-weighted degree + row count.  Default is
        # ``None`` → labeler behavior is byte-identical to the pre-embedding
        # tiebreaker chain.
        embedding_resolver = None
        if emb_cfg is not None and emb_cfg.cluster_blend > 0.0:
            embedding_resolver = _build_embedding_resolver(store, database_key)

        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        labeler = ClusterLabeler(store, embedding_resolver=embedding_resolver)
        await labeler.label_and_persist(
            clusters, database_key, cluster_schema_patterns=patterns
        )
        if _PROFILE_INDEX:
            print(
                f"[profile] intelligence.cluster_labeling: {(time.perf_counter() - _t) * 1000:.0f}ms",
                flush=True,
            )


def _build_embedding_resolver(
    store: KuzuStore, database_key: str
) -> Callable[[str], list[float] | None]:
    """One-shot fetch of every embedded table for the database, return dict lookup.

    Issuing one bulk query and resolving via ``dict.get`` is materially cheaper
    than per-table Kuzu round-trips inside ``label_and_persist`` (which the
    labeler can call many times per cluster).

    Reuses :meth:`KuzuStore.iter_table_embeddings` so the embedding-fetch
    Cypher lives in exactly one place — if the ``SchemaTable.embedding``
    storage shape changes (column type, optional-match shape, etc.) the
    iterator is the single source of truth that needs updating.
    """
    cache: dict[str, list[float]] = {}
    for row in store.iter_table_embeddings(database=database_key):
        if row.embedding is None:
            continue
        # ``iter_table_embeddings`` may yield the same node_id multiple
        # times (one row per cluster the table belongs to via OPTIONAL
        # MATCH); the embedding is identical across those duplicates so
        # last-write-wins is safe and the dict naturally dedups.
        cache[row.node_id] = row.embedding

    def resolver(nid: str) -> list[float] | None:
        return cache.get(nid)

    return resolver


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
    EmbeddingIndexStep(),
    _LabelStep(),
    _JoinPathsStep(),
]


def build_oss_pipeline() -> PipelineRunner:
    """Return a :class:`PipelineRunner` pre-loaded with the five OSS steps.

    Plugins can append additional steps via :meth:`PipelineRunner.register`
    before calling ``runner.run(ctx)``.

    Returns:
        A :class:`PipelineRunner` with steps: embedding_index → classify →
        cluster → label → join_paths.
    """
    return PipelineRunner(list(_OSS_STEPS))


async def run_intelligence_layer(
    store: KuzuStore,
    database_key: str,
    *,
    config: GraphConfig | PretensorConfig | None = None,
    embeddings_precomputed: bool = False,
) -> None:
    """Clear prior intelligence rows, cluster tables, label, precompute join paths.

    Delegates to :func:`build_oss_pipeline` so external callers get identical
    semantics while plugins can extend the pipeline via :func:`build_oss_pipeline`
    directly.

    Args:
        store: The Kuzu graph store.
        database_key: Logical database name (``SchemaTable.database``).
        config: Optional tuning overrides; accepts :class:`GraphConfig` or
            :class:`PretensorConfig`. When a :class:`PretensorConfig` is given,
            its ``graph`` sub-field is used for clustering/join-path tuning.
            Defaults to :class:`GraphConfig` with OSS defaults.
        embeddings_precomputed: Set True by callers (index/reindex) that
            already ran :func:`compute_table_embeddings` before relationship
            discovery; the ``embedding_index`` pipeline step then skips its
            redundant recompute.
    """
    from pretensor.config import EmbeddingsConfig, PretensorConfig

    if isinstance(config, PretensorConfig):
        cfg = config.graph
        emb_cfg = config.embeddings
    else:
        cfg = config or GraphConfig()
        emb_cfg = EmbeddingsConfig()
    # Single choke point for the env kill switch: every embedding consumer
    # below (classify role vote, cluster blend, label resolver, index step)
    # reads this config, so swapping in the all-off default forces the null
    # path even when the caller's toggles are on.
    if embeddings_disabled_via_env() and emb_cfg != EmbeddingsConfig():
        logger.info(
            "intelligence: PRETENSOR_EMBEDDINGS_DISABLED set; "
            "forcing the null embeddings config"
        )
        emb_cfg = EmbeddingsConfig()
    store.ensure_schema()
    store.clear_intelligence_artifacts()

    _t = time.perf_counter() if _PROFILE_INDEX else 0.0
    exporter = GraphExporter(store)
    graph = exporter.to_igraph(
        database_key, config=cfg, cluster_blend=emb_cfg.cluster_blend
    )
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
            _CTX_EMBEDDINGS_CONFIG: emb_cfg,
            _CTX_EMBEDDINGS_PRECOMPUTED: embeddings_precomputed,
        }
    )
    runner = build_oss_pipeline()
    await runner.run(ctx)


def run_intelligence_layer_sync(
    store: KuzuStore,
    database_key: str,
    *,
    config: GraphConfig | PretensorConfig | None = None,
    embeddings_precomputed: bool = False,
) -> None:
    """Sync wrapper for :func:`run_intelligence_layer` (CLI / builder)."""
    asyncio.run(
        run_intelligence_layer(
            store,
            database_key,
            config=config,
            embeddings_precomputed=embeddings_precomputed,
        )
    )
