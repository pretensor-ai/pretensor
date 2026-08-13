"""Round-trip ExternalConsumer / CONSUMES upserts against an in-memory Kuzu graph."""

from __future__ import annotations

from pathlib import Path

from pretensor.core.ids import (
    consumes_edge_id,
    external_consumer_node_id,
    table_node_id,
)
from pretensor.core.store import KuzuStore
from pretensor.graph_models.consumer import ConsumesEdge, ExternalConsumerNode
from pretensor.graph_models.node import GraphNode


def _table_node(connection: str, schema: str, name: str) -> GraphNode:
    return GraphNode(
        node_id=table_node_id(connection, schema, name),
        connection_name=connection,
        database="db",
        schema_name=schema,
        table_name=name,
        row_count=None,
        comment=None,
        entity_type=None,
        table_type="table",
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


def _consumer(
    service: str, file_path: str, fingerprint: str, run_id: str
) -> ExternalConsumerNode:
    return ExternalConsumerNode(
        node_id=external_consumer_node_id("c", service, file_path, 10, fingerprint),
        connection_name="c",
        service_name=service,
        file_path=file_path,
        language="python",
        symbol="query",
        kind="assignment",
        line_start=10,
        line_end=12,
        sql_fingerprint=fingerprint,
        confidence=0.9,
        dialect_used="postgres",
        scan_run_id=run_id,
    )


def test_ensure_schema_creates_consumer_tables(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        # No rows yet, but the tables must exist (queryable without error).
        assert (
            store.query_all_rows("MATCH (c:ExternalConsumer) RETURN count(*)")[0][0]
            == 0
        )
        assert (
            store.query_all_rows("MATCH ()-[r:CONSUMES]->() RETURN count(*)")[0][0] == 0
        )
    finally:
        store.close()


def test_ensure_schema_idempotent(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        store.ensure_schema()  # must not raise on an existing graph
    finally:
        store.close()


def test_consumer_and_edge_round_trip(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        tnid = table_node_id("c", "public", "users")
        store.upsert_table(_table_node("c", "public", "users"))

        node = _consumer("svc", "src/app.py", "abc123def456", "run1")
        store.upsert_external_consumer(node)
        eid = consumes_edge_id(node.node_id, tnid, "read")
        store.upsert_consumes_edge(
            ConsumesEdge(
                edge_id=eid,
                source_node_id=node.node_id,
                target_node_id=tnid,
                op="read",
                source="analyze",
                confidence=0.9,
                scan_run_id="run1",
            )
        )

        rows = store.query_all_rows(
            """
            MATCH (c:ExternalConsumer)-[r:CONSUMES]->(t:SchemaTable)
            RETURN c.service_name, c.file_path, c.kind, c.sql_fingerprint,
                   r.op, r.source, t.table_name
            """
        )
        assert len(rows) == 1
        svc, fpath, kind, fp, op, src, tname = rows[0]
        assert (svc, fpath, kind, op, src, tname) == (
            "svc",
            "src/app.py",
            "assignment",
            "read",
            "analyze",
            "users",
        )
        assert fp == "abc123def456"
    finally:
        store.close()


def test_upsert_is_idempotent_by_id(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        tnid = table_node_id("c", "public", "users")
        store.upsert_table(_table_node("c", "public", "users"))
        node = _consumer("svc", "src/app.py", "fp", "run1")
        for conf in (0.9, 0.5):
            store.upsert_external_consumer(node.model_copy(update={"confidence": conf}))
            store.upsert_consumes_edge(
                ConsumesEdge(
                    edge_id=consumes_edge_id(node.node_id, tnid, "read"),
                    source_node_id=node.node_id,
                    target_node_id=tnid,
                    op="read",
                    confidence=conf,
                    scan_run_id="run1",
                )
            )
        n = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN count(*)")
        e = store.query_all_rows("MATCH ()-[r:CONSUMES]->() RETURN count(*)")
        assert int(n[0][0]) == 1
        assert int(e[0][0]) == 1
        conf = store.query_all_rows("MATCH ()-[r:CONSUMES]->() RETURN r.confidence")
        assert float(conf[0][0]) == 0.5  # last write wins
    finally:
        store.close()


def test_mark_tables_have_external_consumers(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        tnid = table_node_id("c", "public", "users")
        store.upsert_table(_table_node("c", "public", "users"))
        store.mark_tables_have_external_consumers([tnid])
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {node_id: $n}) RETURN t.has_external_consumers",
            {"n": tnid},
        )
        assert rows[0][0] is True
    finally:
        store.close()


def test_sweep_stale_consumers(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        tnid = table_node_id("c", "public", "users")
        store.upsert_table(_table_node("c", "public", "users"))

        # Prior run.
        old = _consumer("svc", "src/old.py", "fpold", "run1")
        store.upsert_external_consumer(old)
        store.upsert_consumes_edge(
            ConsumesEdge(
                edge_id=consumes_edge_id(old.node_id, tnid, "read"),
                source_node_id=old.node_id,
                target_node_id=tnid,
                op="read",
                scan_run_id="run1",
            )
        )
        # Current run.
        new = _consumer("svc", "src/new.py", "fpnew", "run2")
        store.upsert_external_consumer(new)

        store.sweep_stale_consumers("svc", "c", "run2")

        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN c.scan_run_id")
        assert [r[0] for r in rows] == ["run2"]
        edges = store.query_all_rows("MATCH ()-[r:CONSUMES]->() RETURN count(*)")
        assert int(edges[0][0]) == 0  # stale run1 edge removed with its node
    finally:
        store.close()


def test_sweep_removes_stale_edge_on_refreshed_node(tmp_path: Path) -> None:
    """A consumer re-upserted under the current run sheds edges from prior runs."""
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        users_nid = table_node_id("c", "public", "users")
        orders_nid = table_node_id("c", "public", "orders")
        store.upsert_table(_table_node("c", "public", "users"))
        store.upsert_table(_table_node("c", "public", "orders"))

        node = _consumer("svc", "src/app.py", "fp", "run2")  # refreshed to run2
        store.upsert_external_consumer(node)
        # Orphan edge left over from run1 (its table no longer resolves).
        store.upsert_consumes_edge(
            ConsumesEdge(
                edge_id=consumes_edge_id(node.node_id, users_nid, "read"),
                source_node_id=node.node_id,
                target_node_id=users_nid,
                op="read",
                scan_run_id="run1",
            )
        )
        # Current edge written by run2.
        store.upsert_consumes_edge(
            ConsumesEdge(
                edge_id=consumes_edge_id(node.node_id, orders_nid, "read"),
                source_node_id=node.node_id,
                target_node_id=orders_nid,
                op="read",
                scan_run_id="run2",
            )
        )

        store.sweep_stale_consumers("svc", "c", "run2")

        edges = store.query_all_rows(
            "MATCH ()-[r:CONSUMES]->(t:SchemaTable) RETURN t.table_name, r.scan_run_id"
        )
        assert edges == [("orders", "run2")]
        nodes = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN count(*)")
        assert int(nodes[0][0]) == 1  # the refreshed node itself survives
    finally:
        store.close()


def test_sweep_clears_flag_when_last_consumer_removed(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        tnid = table_node_id("c", "public", "users")
        store.upsert_table(_table_node("c", "public", "users"))

        node = _consumer("svc", "src/app.py", "fp", "run1")
        store.upsert_external_consumer(node)
        store.upsert_consumes_edge(
            ConsumesEdge(
                edge_id=consumes_edge_id(node.node_id, tnid, "read"),
                source_node_id=node.node_id,
                target_node_id=tnid,
                op="read",
                scan_run_id="run1",
            )
        )
        store.mark_tables_have_external_consumers([tnid])

        # run2 found nothing for this service: the swept table loses its flag.
        store.sweep_stale_consumers("svc", "c", "run2")

        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {node_id: $n}) RETURN t.has_external_consumers",
            {"n": tnid},
        )
        assert rows[0][0] is False
    finally:
        store.close()


def test_sweep_preserves_flag_still_backed_by_other_service(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        tnid = table_node_id("c", "public", "users")
        store.upsert_table(_table_node("c", "public", "users"))

        for service, run in (("svc_a", "run1"), ("svc_b", "runX")):
            node = _consumer(service, "src/app.py", "fp", run)
            store.upsert_external_consumer(node)
            store.upsert_consumes_edge(
                ConsumesEdge(
                    edge_id=consumes_edge_id(node.node_id, tnid, "read"),
                    source_node_id=node.node_id,
                    target_node_id=tnid,
                    op="read",
                    scan_run_id=run,
                )
            )
        store.mark_tables_have_external_consumers([tnid])

        # svc_a rescans and drops the table; svc_b's edge still backs the flag.
        store.sweep_stale_consumers("svc_a", "c", "run2")

        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {node_id: $n}) RETURN t.has_external_consumers",
            {"n": tnid},
        )
        assert rows[0][0] is True
    finally:
        store.close()


def test_sweep_never_touches_database_side_flag(tmp_path: Path) -> None:
    """A table flagged by the dbt-exposure signal (no CONSUMES edges) is untouched."""
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        tnid = table_node_id("c", "public", "exposed")
        store.upsert_table(_table_node("c", "public", "exposed"))
        store.mark_tables_have_external_consumers([tnid])  # dbt-style flag, no edges

        store.sweep_stale_consumers("svc", "c", "run1")

        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {node_id: $n}) RETURN t.has_external_consumers",
            {"n": tnid},
        )
        assert rows[0][0] is True
    finally:
        store.close()
