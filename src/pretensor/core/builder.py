"""Build a Kuzu graph from a pipeline :class:`SchemaSnapshot`."""

from __future__ import annotations

import json
import logging

from pretensor.config import GraphConfig, PretensorConfig
from pretensor.connectors.models import Column, SchemaSnapshot, ViewDependency
from pretensor.core.ids import (
    column_node_id,
    fk_edge_id,
    lineage_edge_id,
    table_node_id,
)
from pretensor.core.store import KuzuStore
from pretensor.graph_models.edge import GraphEdge, LineageEdge
from pretensor.graph_models.node import GraphNode
from pretensor.intelligence.discovery import RelationshipDiscovery
from pretensor.intelligence.pipeline import run_intelligence_layer_sync
from pretensor.visibility.filter import VisibilityFilter

__all__ = ["GraphBuilder", "table_node_id", "fk_edge_id"]

logger = logging.getLogger(__name__)


def _filtered_snapshot_for_visibility(
    snapshot: SchemaSnapshot,
    visibility_filter: VisibilityFilter,
) -> SchemaSnapshot:
    """Drop hidden tables and view dependencies that touch hidden tables."""
    visible_tables = [
        t
        for t in snapshot.tables
        if visibility_filter.is_table_visible(
            snapshot.connection_name, t.schema_name, t.name
        )
    ]
    keys = {(t.schema_name, t.name) for t in visible_tables}
    deps: list[ViewDependency] = []
    for dep in snapshot.view_dependencies:
        if (dep.source_schema, dep.source_table) not in keys:
            continue
        if (dep.target_schema, dep.target_table) not in keys:
            continue
        deps.append(dep)
    return snapshot.model_copy(
        update={"tables": visible_tables, "view_dependencies": deps}
    )


class GraphBuilder:
    """Writes table nodes and foreign-key edges from a snapshot."""

    def build(
        self,
        snapshot: SchemaSnapshot,
        store: KuzuStore,
        *,
        run_relationship_discovery: bool = True,
        replace_mode: str = "full",
        graph_config: GraphConfig | None = None,
        config: PretensorConfig | None = None,
        visibility_filter: VisibilityFilter | None = None,
    ) -> None:
        """Populate ``store`` from ``snapshot`` (replaces existing graph content).

        When ``run_relationship_discovery`` is True, runs heuristic discovery and
        writes ``INFERRED_JOIN`` edges that are not already explicit FKs.

        Args:
            replace_mode: ``full`` clears the entire graph (default per-connection index).
                ``connection`` removes only nodes/edges for ``snapshot.connection_name``
                (unified multi-DB graph file).
            graph_config: Deprecated. Use ``config.graph`` instead. If both are given,
                ``config.graph`` takes precedence.
            config: Full pluggable configuration.  When provided, its ``scorer_registry``
                and ``combiner`` are forwarded to :class:`RelationshipDiscovery` and
                its ``graph`` field is forwarded to the intelligence pipeline.
        """
        store.ensure_schema()
        if replace_mode == "connection":
            store.clear_connection_subgraph(snapshot.connection_name)
        else:
            store.clear_graph()

        snap = (
            _filtered_snapshot_for_visibility(snapshot, visibility_filter)
            if visibility_filter is not None
            else snapshot
        )

        for table in snap.tables:
            grants_json = (
                json.dumps(table.grants, ensure_ascii=False) if table.grants else None
            )
            node = GraphNode(
                node_id=table_node_id(
                    snap.connection_name, table.schema_name, table.name
                ),
                connection_name=snap.connection_name,
                database=snap.database,
                schema_name=table.schema_name,
                table_name=table.name,
                row_count=table.row_count,
                row_count_source=table.row_count_source,
                comment=table.comment,
                table_type=table.table_type,
                seq_scan_count=table.seq_scan_count,
                idx_scan_count=table.idx_scan_count,
                insert_count=table.insert_count,
                update_count=table.update_count,
                delete_count=table.delete_count,
                is_partitioned=table.is_partitioned,
                partition_key=table.partition_key,
                grants_json=grants_json,
                access_read_count=table.access_read_count,
                access_write_count=table.access_write_count,
                days_since_last_access=table.days_since_last_access,
                potentially_unused=table.potentially_unused,
                table_bytes=table.table_bytes,
                clustering_key=table.clustering_key,
            )
            store.upsert_table(node)
            col_names = [c.name for c in table.columns]
            if visibility_filter is not None:
                allowed_cols = visibility_filter.visible_columns(
                    snap.connection_name,
                    table.schema_name,
                    table.name,
                    col_names,
                )
                allowed_set = set(allowed_cols)
                pending = [c for c in table.columns if c.name in allowed_set]
            else:
                pending = list(table.columns)
            written: set[str] = set()
            while pending:
                progressed = False
                next_pass: list[Column] = []
                for col in pending:
                    parent_id: str | None = None
                    if col.parent_column is not None:
                        if col.parent_column not in written:
                            next_pass.append(col)
                            continue
                        parent_id = column_node_id(
                            snap.connection_name,
                            table.schema_name,
                            table.name,
                            col.parent_column,
                        )
                    cid = column_node_id(
                        snap.connection_name, table.schema_name, table.name, col.name
                    )
                    mcv_json = (
                        json.dumps(col.most_common_values, ensure_ascii=False)
                        if col.most_common_values
                        else None
                    )
                    hb_json = (
                        json.dumps(col.histogram_bounds, ensure_ascii=False)
                        if col.histogram_bounds
                        else None
                    )
                    store.upsert_column_for_table(
                        column_node_id=cid,
                        connection_name=snap.connection_name,
                        database=snap.database,
                        schema_name=table.schema_name,
                        table_name=table.name,
                        column_name=col.name,
                        data_type=col.data_type,
                        nullable=col.nullable,
                        is_primary_key=col.is_primary_key,
                        is_foreign_key=col.is_foreign_key,
                        table_node_id=node.node_id,
                        comment=col.comment,
                        default_value=col.default_value,
                        is_indexed=col.is_indexed,
                        check_constraints=col.check_constraints,
                        ordinal_position=col.ordinal_position,
                        most_common_values_json=mcv_json,
                        histogram_bounds_json=hb_json,
                        stats_correlation=col.stats_correlation,
                        column_cardinality=col.column_cardinality,
                        index_type=col.index_type,
                        index_is_unique=col.index_is_unique,
                        parent_column_id=parent_id,
                        is_array=col.is_array,
                    )
                    written.add(col.name)
                    progressed = True
                if not progressed and next_pass:
                    logger.warning(
                        "build: %d column(s) for table %s.%s skipped — "
                        "unresolvable parent_column reference(s): %s",
                        len(next_pass),
                        table.schema_name,
                        table.name,
                        ", ".join(col.name for col in next_pass),
                    )
                    break
                pending = next_pass

        known = {
            table_node_id(snap.connection_name, t.schema_name, t.name) for t in snap.tables
        }

        for table in snap.tables:
            for fk in table.foreign_keys:
                src_id = table_node_id(
                    snap.connection_name, fk.source_schema, fk.source_table
                )
                dst_id = table_node_id(
                    snap.connection_name, fk.target_schema, fk.target_table
                )
                if src_id not in known or dst_id not in known:
                    continue
                edge = GraphEdge(
                    edge_id=fk_edge_id(fk),
                    source_node_id=src_id,
                    target_node_id=dst_id,
                    source_column=fk.source_column,
                    target_column=fk.target_column,
                    constraint_name=fk.constraint_name,
                )
                store.merge_fk_edge(edge)

        lineage_by_id: dict[str, LineageEdge] = {}
        for dep in snap.view_dependencies:
            src_id = table_node_id(
                snap.connection_name, dep.source_schema, dep.source_table
            )
            dst_id = table_node_id(
                snap.connection_name, dep.target_schema, dep.target_table
            )
            if src_id not in known or dst_id not in known:
                continue
            eid = lineage_edge_id(snap.connection_name, dep)
            src_label = f"{snap.connection_name}:{dep.object_name}"
            cand = LineageEdge(
                edge_id=eid,
                source_node_id=src_id,
                target_node_id=dst_id,
                source=src_label,
                lineage_type=dep.lineage_type,
                confidence=float(dep.confidence),
            )
            existing = lineage_by_id.get(eid)
            if existing is None or cand.confidence > existing.confidence:
                lineage_by_id[eid] = cand
        for ledge in lineage_by_id.values():
            store.upsert_lineage_edge(ledge)

        effective_config = config or PretensorConfig()
        effective_graph_cfg = effective_config.graph if config else (graph_config or GraphConfig())

        if run_relationship_discovery:
            RelationshipDiscovery(
                store,
                scorers=effective_config.scorer_registry,
                combiner=effective_config.combiner,
                graph_config=effective_graph_cfg,
            ).discover(snap)

        run_intelligence_layer_sync(store, snap.database, config=effective_graph_cfg)
