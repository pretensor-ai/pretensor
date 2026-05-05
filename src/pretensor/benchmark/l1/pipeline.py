"""Pipeline orchestration for L1 metric collection.

Glues the existing :class:`GraphBuilder` + :class:`KuzuStore` +
:class:`ClusteringEngine` + :func:`classify_database_tables` plumbing
into a single helper that returns the artifacts the four L1 metrics
need. Lives separately from :mod:`pretensor.benchmark.l1.metrics` so
the pure metric helpers stay testable without spinning up Kuzu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pretensor.benchmark.l1.metrics import JoinKey
from pretensor.connectors.models import SchemaSnapshot
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.intelligence.clustering import ClusteringEngine
from pretensor.intelligence.discovery import RelationshipDiscovery
from pretensor.intelligence.graph_export import GraphExporter
from pretensor.intelligence.schema_classification import classify_database_tables

__all__ = ["L1Artifacts", "build_l1_artifacts", "discover_inferred_joins_blind"]


@dataclass(frozen=True, slots=True)
class L1Artifacts:
    """Per-run outputs collected from the intelligence pipeline."""

    clusters: list[frozenset[str]] = field(default_factory=list)
    """One frozenset of table node IDs per cluster."""

    roles: dict[str, str] = field(default_factory=dict)
    """Map of bare table name → role literal (matches ``TableRole``)."""


def build_l1_artifacts(
    snapshot: SchemaSnapshot,
    *,
    work_dir: Path,
) -> L1Artifacts:
    """Run the production pipeline once and collect L1-relevant outputs.

    The pipeline runs over the snapshot as authored, with declared FKs
    in place — clustering and role classification both take advantage
    of that signal in production, and the L1 metric must reflect that.

    The blind-discovery leg (FKs masked, used for inferred-join P/R)
    runs separately via :func:`discover_inferred_joins_blind` so the
    two legs don't share state.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(work_dir / "graph.kuzu")
    try:
        GraphBuilder().build(snapshot, store)

        graph = GraphExporter(store).to_igraph(snapshot.database)
        clusters_raw = ClusteringEngine().cluster(graph)
        clusters = sorted(
            (frozenset(c.table_ids) for c in clusters_raw),
            key=lambda fs: tuple(sorted(fs)),
        )

        classifications = classify_database_tables(store, snapshot.database)
        roles = {
            _bare_table_name(node_id, snapshot.connection_name): cls.role
            for node_id, cls in classifications.items()
        }
    finally:
        store.close()

    return L1Artifacts(clusters=list(clusters), roles=roles)


def discover_inferred_joins_blind(
    snapshot: SchemaSnapshot,
    *,
    work_dir: Path,
) -> list[JoinKey]:
    """Run heuristic discovery against an FK-masked snapshot.

    Production discovery filters out candidates that match declared FKs,
    so a metric that compares the inferred set against declared FKs only
    makes sense when those FKs are hidden from the heuristic. This
    helper masks the snapshot, runs the heuristic, and returns the
    candidate joins as canonical (table-name, column) pairs ready for
    :func:`pretensor.benchmark.l1.metrics.inferred_join_pr`.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    masked = _mask_foreign_keys(snapshot)
    store = KuzuStore(work_dir / "graph.kuzu")
    try:
        # Build the FK-masked graph so RelationshipDiscovery has node
        # context (column metadata, neighbours) to score candidates.
        GraphBuilder().build(masked, store, run_relationship_discovery=False)
        candidates = RelationshipDiscovery(store).discover(masked)
    finally:
        store.close()

    keys: list[JoinKey] = []
    for cand in candidates:
        src = _bare_table_name(cand.source_node_id, snapshot.connection_name)
        dst = _bare_table_name(cand.target_node_id, snapshot.connection_name)
        keys.append((src, cand.source_column, dst, cand.target_column))
    keys.sort()
    return keys


def collect_declared_fks(snapshot: SchemaSnapshot) -> list[JoinKey]:
    """Return the canonical declared-FK ground-truth set for the snapshot."""
    keys: list[JoinKey] = []
    for table in snapshot.tables:
        for fk in table.foreign_keys:
            src = f"{fk.source_schema}.{fk.source_table}"
            dst = f"{fk.target_schema}.{fk.target_table}"
            keys.append((src, fk.source_column, dst, fk.target_column))
    keys.sort()
    return keys


def _mask_foreign_keys(snapshot: SchemaSnapshot) -> SchemaSnapshot:
    """Return a copy of ``snapshot`` with every table's ``foreign_keys`` cleared.

    Used so :class:`RelationshipDiscovery` has to rediscover the joins
    from column-level signals rather than reading them off the FK list.
    """
    masked_tables = []
    for table in snapshot.tables:
        masked_tables.append(table.model_copy(update={"foreign_keys": []}))
    return snapshot.model_copy(update={"tables": masked_tables})


def _bare_table_name(node_id: str, connection_name: str) -> str:
    """Turn a Kuzu ``node_id`` (``conn::schema::table``) into ``schema.table``."""
    prefix = f"{connection_name}::"
    rest = node_id[len(prefix) :] if node_id.startswith(prefix) else node_id
    return rest.replace("::", ".", 1)
