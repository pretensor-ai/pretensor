"""Apply :class:`SchemaChange` list to the Kuzu graph (incremental reindex)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.connectors.snapshot import ChangeTarget, ChangeType, SchemaChange
from pretensor.core.ids import column_node_id, fk_edge_id, table_node_id
from pretensor.core.store import KuzuStore
from pretensor.graph_models.edge import GraphEdge
from pretensor.graph_models.node import GraphNode

__all__ = ["GraphPatcher", "PatchResult"]


@dataclass
class PatchResult:
    """Counts of mutations applied (or that would be applied in dry-run)."""

    tables_added: int = 0
    tables_removed: int = 0
    columns_added: int = 0
    columns_removed: int = 0
    columns_updated: int = 0
    fk_edges_removed: int = 0
    inferred_joins_removed: int = 0
    join_paths_marked_stale: int = 0
    cluster_memberships_removed: int = 0
    clusters_marked_stale: int = 0
    foreign_keys_synced: int = 0
    notes: list[str] = field(default_factory=list)


class GraphPatcher:
    """Translate pipeline :class:`SchemaChange` rows into Kuzu writes."""

    def __init__(self, store: KuzuStore) -> None:
        self._store = store

    def apply(
        self,
        changes: list[SchemaChange],
        new_snapshot: SchemaSnapshot,
        *,
        dry_run: bool = False,
    ) -> PatchResult:
        """Apply changes using ``new_snapshot`` for full row data.

        Args:
            changes: Output of :func:`pretensor.connectors.snapshot.diff_snapshots`.
            new_snapshot: Fresh introspection result (source of truth for adds/updates).
            dry_run: If True, compute counts only (no graph writes).

        Returns:
            Summary of operations.
        """
        self._store.ensure_schema()
        result = PatchResult()
        tables_by_key = {(t.schema_name, t.name): t for t in new_snapshot.tables}
        connection_name = new_snapshot.connection_name
        database_key = new_snapshot.database

        def invalidate_table(schema_name: str, table_name: str) -> None:
            tid = table_node_id(connection_name, schema_name, table_name)
            if dry_run:
                return
            self._store.mark_join_paths_stale_for_table(tid, database_key)
            self._store.remove_table_from_clusters(tid)
            self._store.mark_clusters_stale_for_table(tid, database_key)

        for ch in changes:
            key = (ch.schema_name, ch.table_name)
            if ch.change_type == ChangeType.ADDED and ch.target == ChangeTarget.TABLE:
                tbl = tables_by_key.get(key)
                if tbl is None:
                    result.notes.append(f"skip ADDED table missing in snapshot: {key}")
                    continue
                if dry_run:
                    result.tables_added += 1
                    result.columns_added += len(tbl.columns)
                    continue
                invalidate_table(ch.schema_name, ch.table_name)
                grants_json = (
                    json.dumps(tbl.grants, ensure_ascii=False) if tbl.grants else None
                )
                node = GraphNode(
                    node_id=table_node_id(connection_name, tbl.schema_name, tbl.name),
                    connection_name=connection_name,
                    database=new_snapshot.database,
                    schema_name=tbl.schema_name,
                    table_name=tbl.name,
                    row_count=tbl.row_count,
                    comment=tbl.comment,
                    table_type=tbl.table_type,
                    seq_scan_count=tbl.seq_scan_count,
                    idx_scan_count=tbl.idx_scan_count,
                    insert_count=tbl.insert_count,
                    update_count=tbl.update_count,
                    delete_count=tbl.delete_count,
                    is_partitioned=tbl.is_partitioned,
                    partition_key=tbl.partition_key,
                    grants_json=grants_json,
                    access_read_count=tbl.access_read_count,
                    access_write_count=tbl.access_write_count,
                    days_since_last_access=tbl.days_since_last_access,
                    potentially_unused=tbl.potentially_unused,
                    table_bytes=tbl.table_bytes,
                    clustering_key=tbl.clustering_key,
                )
                self._store.upsert_table(node)
                for col in tbl.columns:
                    self._upsert_column(
                        connection_name, new_snapshot.database, tbl, col
                    )
                result.tables_added += 1
                result.columns_added += len(tbl.columns)

            elif (
                ch.change_type == ChangeType.REMOVED and ch.target == ChangeTarget.TABLE
            ):
                tid = table_node_id(connection_name, ch.schema_name, ch.table_name)
                if dry_run:
                    result.tables_removed += 1
                    continue
                self._store.delete_join_paths_for_table(tid, database_key)
                self._store.remove_table_from_clusters(tid)
                self._store.mark_clusters_stale_for_table(tid, database_key)
                self._store.delete_table_node_cascade(tid)
                result.tables_removed += 1

            elif (
                ch.change_type == ChangeType.ADDED and ch.target == ChangeTarget.COLUMN
            ):
                tbl = tables_by_key.get(key)
                if tbl is None or not ch.column_name:
                    continue
                col = next((c for c in tbl.columns if c.name == ch.column_name), None)
                if col is None:
                    continue
                if dry_run:
                    result.columns_added += 1
                    continue
                invalidate_table(ch.schema_name, ch.table_name)
                self._upsert_column(connection_name, new_snapshot.database, tbl, col)
                result.columns_added += 1

            elif (
                ch.change_type == ChangeType.REMOVED
                and ch.target == ChangeTarget.COLUMN
            ):
                if not ch.column_name:
                    continue
                tid = table_node_id(connection_name, ch.schema_name, ch.table_name)
                cid = column_node_id(
                    connection_name, ch.schema_name, ch.table_name, ch.column_name
                )
                if dry_run:
                    result.columns_removed += 1
                    result.fk_edges_removed += self._count_fk_touching(
                        tid, ch.column_name
                    )
                    result.inferred_joins_removed += self._count_inferred_touching(
                        tid, ch.column_name
                    )
                    continue
                invalidate_table(ch.schema_name, ch.table_name)
                result.fk_edges_removed += self._store.delete_fk_edges_touching_column(
                    tid, ch.column_name
                )
                result.inferred_joins_removed += (
                    self._store.delete_inferred_joins_touching_column(
                        tid, ch.column_name
                    )
                )
                self._store.delete_column_node(cid)
                result.columns_removed += 1

            elif (
                ch.change_type == ChangeType.MODIFIED
                and ch.target == ChangeTarget.COLUMN
            ):
                tbl = tables_by_key.get(key)
                if tbl is None or not ch.column_name:
                    continue
                col = next((c for c in tbl.columns if c.name == ch.column_name), None)
                if col is None:
                    continue
                if dry_run:
                    result.columns_updated += 1
                    continue
                invalidate_table(ch.schema_name, ch.table_name)
                self._upsert_column(connection_name, new_snapshot.database, tbl, col)
                result.columns_updated += 1

            elif (
                ch.change_type == ChangeType.MODIFIED
                and ch.target == ChangeTarget.TABLE
            ):
                tbl = tables_by_key.get(key)
                if tbl is None:
                    continue
                if dry_run:
                    result.columns_updated += len(tbl.columns)
                    continue
                invalidate_table(ch.schema_name, ch.table_name)
                grants_json = (
                    json.dumps(tbl.grants, ensure_ascii=False) if tbl.grants else None
                )
                node = GraphNode(
                    node_id=table_node_id(connection_name, tbl.schema_name, tbl.name),
                    connection_name=connection_name,
                    database=new_snapshot.database,
                    schema_name=tbl.schema_name,
                    table_name=tbl.name,
                    row_count=tbl.row_count,
                    comment=tbl.comment,
                    table_type=tbl.table_type,
                    seq_scan_count=tbl.seq_scan_count,
                    idx_scan_count=tbl.idx_scan_count,
                    insert_count=tbl.insert_count,
                    update_count=tbl.update_count,
                    delete_count=tbl.delete_count,
                    is_partitioned=tbl.is_partitioned,
                    partition_key=tbl.partition_key,
                    grants_json=grants_json,
                    access_read_count=tbl.access_read_count,
                    access_write_count=tbl.access_write_count,
                    days_since_last_access=tbl.days_since_last_access,
                    potentially_unused=tbl.potentially_unused,
                    table_bytes=tbl.table_bytes,
                    clustering_key=tbl.clustering_key,
                )
                self._store.upsert_table(node)
                for col in tbl.columns:
                    self._upsert_column(
                        connection_name, new_snapshot.database, tbl, col
                    )
                result.columns_updated += len(tbl.columns)

        if not dry_run:
            result.foreign_keys_synced = self._sync_foreign_keys(new_snapshot)

        return result

    def _upsert_column(
        self,
        connection_name: str,
        database: str,
        table: Table,
        col: Column,
    ) -> None:
        tid = table_node_id(connection_name, table.schema_name, table.name)
        cid = column_node_id(connection_name, table.schema_name, table.name, col.name)
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
        self._store.upsert_column_for_table(
            column_node_id=cid,
            connection_name=connection_name,
            database=database,
            schema_name=table.schema_name,
            table_name=table.name,
            column_name=col.name,
            data_type=col.data_type,
            nullable=col.nullable,
            is_primary_key=col.is_primary_key,
            is_foreign_key=col.is_foreign_key,
            table_node_id=tid,
            comment=col.comment,
            default_value=col.default_value,
            is_indexed=col.is_indexed,
            check_constraints=col.check_constraints,
            ordinal_position=col.ordinal_position,
            most_common_values_json=mcv_json,
            histogram_bounds_json=hb_json,
            stats_correlation=col.stats_correlation,
        )

    def _sync_foreign_keys(self, snapshot: SchemaSnapshot) -> int:
        """Replace FK edges with the set implied by ``snapshot``."""
        known = {
            table_node_id(snapshot.connection_name, t.schema_name, t.name)
            for t in snapshot.tables
        }
        new_edge_ids: set[str] = set()
        edges: list[GraphEdge] = []
        for t in snapshot.tables:
            src_id = table_node_id(snapshot.connection_name, t.schema_name, t.name)
            for fk in t.foreign_keys:
                dst_id = table_node_id(
                    snapshot.connection_name, fk.target_schema, fk.target_table
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
                new_edge_ids.add(edge.edge_id)
                edges.append(edge)

        rows = self._store.query_all_rows(
            """
            MATCH (src:SchemaTable)-[r:FK_REFERENCES]->(dst:SchemaTable)
            WHERE src.connection_name = $cn
            RETURN r.edge_id
            """,
            {"cn": snapshot.connection_name},
        )
        old_ids = {str(r[0]) for r in rows if r[0] is not None}
        for eid in old_ids - new_edge_ids:
            self._store.execute(
                "MATCH ()-[r:FK_REFERENCES {edge_id: $eid}]->() DELETE r",
                {"eid": eid},
            )
        for edge in edges:
            self._store.merge_fk_edge(edge)
        return len(edges)

    def _count_fk_touching(self, table_node_id: str, column_name: str) -> int:
        rows = self._store.query_all_rows(
            """
            MATCH (src:SchemaTable)-[r:FK_REFERENCES]->(dst:SchemaTable)
            WHERE (src.node_id = $tid AND r.source_column = $col)
               OR (dst.node_id = $tid AND r.target_column = $col)
            RETURN count(*)
            """,
            {"tid": table_node_id, "col": column_name},
        )
        return int(rows[0][0]) if rows and rows[0][0] is not None else 0

    def _count_inferred_touching(self, table_node_id: str, column_name: str) -> int:
        rows = self._store.query_all_rows(
            """
            MATCH (src:SchemaTable)-[r:INFERRED_JOIN]->(dst:SchemaTable)
            WHERE (src.node_id = $tid AND r.source_column = $col)
               OR (dst.node_id = $tid AND r.target_column = $col)
            RETURN count(*)
            """,
            {"tid": table_node_id, "col": column_name},
        )
        return int(rows[0][0]) if rows and rows[0][0] is not None else 0
