"""Kuzu LINEAGE rel table and store helpers."""

from __future__ import annotations

from pathlib import Path

from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.graph_models.edge import LineageEdge
from pretensor.graph_models.node import GraphNode


def _table_node(
    connection: str, schema: str, name: str, *, table_type: str = "table"
) -> GraphNode:
    return GraphNode(
        node_id=table_node_id(connection, schema, name),
        connection_name=connection,
        database="db",
        schema_name=schema,
        table_name=name,
        row_count=None,
        comment=None,
        entity_type=None,
        table_type=table_type,
        seq_scan_count=None,
        idx_scan_count=None,
        insert_count=None,
        update_count=None,
        delete_count=None,
        is_partitioned=None,
        partition_key=None,
        grants_json=None,
        access_read_count=None,
        access_write_count=None,
        days_since_last_access=None,
        potentially_unused=None,
        table_bytes=None,
        clustering_key=None,
    )


def test_upsert_lineage_edge_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "g.kuzu"
    store = KuzuStore(db)
    try:
        store.ensure_schema()
        store.upsert_table(_table_node("c", "public", "a"))
        store.upsert_table(_table_node("c", "public", "b", table_type="view"))
        edge = LineageEdge(
            edge_id="lineage::test1",
            source_node_id=table_node_id("c", "public", "a"),
            target_node_id=table_node_id("c", "public", "b"),
            source="c:public.b",
            lineage_type="VIEW",
            confidence=1.0,
        )
        store.upsert_lineage_edge(edge)
        store.upsert_lineage_edge(
            LineageEdge(
                edge_id="lineage::test1",
                source_node_id=table_node_id("c", "public", "a"),
                target_node_id=table_node_id("c", "public", "b"),
                source="c:public.b",
                lineage_type="VIEW",
                confidence=0.5,
            )
        )
        rows = store.query_all_rows(
            """
            MATCH ()-[r:LINEAGE {edge_id: $eid}]->()
            RETURN r.confidence
            """,
            {"eid": "lineage::test1"},
        )
        assert len(rows) == 1
        assert float(rows[0][0]) == 0.5
    finally:
        store.close()


def test_clear_lineage_edges_by_connection(tmp_path: Path) -> None:
    db = tmp_path / "g2.kuzu"
    store = KuzuStore(db)
    try:
        store.ensure_schema()
        store.upsert_table(_table_node("c1", "public", "x"))
        store.upsert_table(_table_node("c1", "public", "y"))
        store.upsert_lineage_edge(
            LineageEdge(
                edge_id="e1",
                source_node_id=table_node_id("c1", "public", "x"),
                target_node_id=table_node_id("c1", "public", "y"),
                source="t",
                lineage_type="VIEW",
                confidence=1.0,
            )
        )
        store.clear_lineage_edges("c1")
        n = store.query_all_rows("MATCH ()-[r:LINEAGE]->() RETURN count(*) AS n")
        assert int(n[0][0]) == 0
    finally:
        store.close()
