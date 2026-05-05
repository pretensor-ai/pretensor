"""Per-call graph construction for the L2 benchmark.

Each invocation of :func:`run_l2` indexes the fixture's schema YAML into
a fresh, throwaway Kuzu store + ``registry.json`` under a temp dir,
then hands that directory to the MCP tool payload functions
(``query_payload``, ``traverse_payload``, ``compile_metric_payload``).
This guarantees determinism — no leftover state from previous runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import SchemaSnapshot
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore

__all__ = ["build_l2_graph_dir"]


# Pinned so the persisted ``last_indexed_at`` doesn't churn the registry
# bytes between runs. The benchmark only consults graph contents — the
# timestamp is metadata that has no semantic effect on tool output, but
# we still want byte-identical state across invocations for any future
# code that hashes the registry.
_DETERMINISTIC_INDEXED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


def build_l2_graph_dir(snapshot: SchemaSnapshot, *, work_dir: Path) -> Path:
    """Build a Kuzu graph + registry under ``work_dir`` and return the dir.

    The returned path is suitable as the ``graph_dir`` argument to every
    MCP tool payload function. ``work_dir`` is created if it doesn't
    exist; the caller (typically ``tempfile.TemporaryDirectory``) is
    responsible for cleanup.

    Notes:
        ``run_relationship_discovery=False`` matches the determinism
        policy — L2 metrics measure the tools' behaviour over the
        declared FK / structural shape, not over heuristic discovery
        output. (L1 covers the discovery quality leg via its own
        :func:`pretensor.benchmark.l1.pipeline.discover_inferred_joins_blind`.)

        After build, this helper strips :class:`Cluster` nodes (and the
        ``IN_CLUSTER`` edges that link them to tables). Reason: the
        intelligence layer that ``GraphBuilder.build`` runs unconditionally
        produces cluster labels via Louvain/Leiden community detection,
        and even with a seeded RNG the labels can differ across
        environments (different igraph builds, different platform RNG
        wiring). Those labels feed into the FTS5 keyword index's
        ``cluster_context`` column and shift BM25 scores across machines.
        L2 measures ``query`` / ``semantic_search`` / ``traverse`` /
        ``compile_metric`` quality — none of those tools depend on
        clustering being present, so removing it from the graph keeps
        the benchmark cross-environment-deterministic without changing
        what's being measured.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    graph_path = work_dir / "graphs" / f"{snapshot.connection_name}.kuzu"
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    store = KuzuStore(graph_path)
    try:
        GraphBuilder().build(snapshot, store, run_relationship_discovery=False)
        # See docstring: clusters introduce cross-environment nondeterminism
        # via FTS5's cluster_context column. L2 doesn't measure clustering
        # quality (L1 does), so strip them before the runner queries the graph.
        _strip_clusters(store)
    finally:
        store.close()

    reg = GraphRegistry(work_dir / "registry.json").load()
    reg.upsert(
        connection_name=snapshot.connection_name,
        database=snapshot.database,
        dsn="",
        graph_path=graph_path,
        indexed_at=_DETERMINISTIC_INDEXED_AT,
    )
    reg.save()

    return work_dir


def _strip_clusters(store: KuzuStore) -> None:
    """Remove all ``Cluster`` nodes (and ``IN_CLUSTER`` edges) from the graph.

    Kuzu doesn't expose ``DETACH DELETE`` semantics on every type, so
    we wipe the relationship rows first and then the cluster nodes.
    """
    store.execute_write("MATCH ()-[r:IN_CLUSTER]->() DELETE r")
    store.execute_write("MATCH (c:Cluster) DELETE c")
