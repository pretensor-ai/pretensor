"""Direct tests for the ``consumers`` MCP tool payload."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import (
    consumes_edge_id,
    external_consumer_node_id,
    table_node_id,
)
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.graph_models.consumer import ConsumesEdge, ExternalConsumerNode
from pretensor.mcp.tools.consumers import consumers_payload


def _add_consumer(
    store: KuzuStore,
    table_nid: str,
    *,
    service: str,
    op: str,
    confidence: float,
    fingerprint: str,
) -> None:
    c_nid = external_consumer_node_id(
        "demo", service, f"src/{service}.py", 1, fingerprint
    )
    store.upsert_external_consumer(
        ExternalConsumerNode(
            node_id=c_nid,
            connection_name="demo",
            service_name=service,
            file_path=f"src/{service}.py",
            language="python",
            symbol="query",
            kind="call",
            line_start=1,
            line_end=1,
            sql_fingerprint=fingerprint,
            confidence=confidence,
            dialect_used="postgres",
            scan_run_id="run1",
        )
    )
    store.upsert_consumes_edge(
        ConsumesEdge(
            edge_id=consumes_edge_id(c_nid, table_nid, op),
            source_node_id=c_nid,
            target_node_id=table_nid,
            op=op,
            confidence=confidence,
            scan_run_id="run1",
        )
    )


def _seed(tmp_path: Path) -> None:
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[
            Table(
                name="orders",
                schema_name="public",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            )
        ],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        nid = table_node_id("demo", "public", "orders")
        _add_consumer(
            store,
            nid,
            service="api",
            op="read",
            confidence=0.9,
            fingerprint="fp_aaaaaaaaaaaa",
        )
        _add_consumer(
            store,
            nid,
            service="etl",
            op="write",
            confidence=0.9,
            fingerprint="fp_bbbbbbbbbbbb",
        )
        _add_consumer(
            store,
            nid,
            service="adhoc",
            op="read",
            confidence=0.4,
            fingerprint="fp_cccccccccccc",
        )
    finally:
        store.close()
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://localhost/demo",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()


def test_consumers_returns_entries_and_counts(tmp_path: Path) -> None:
    _seed(tmp_path)
    res = consumers_payload(tmp_path, table="public.orders", database="demo")
    assert "error" not in res
    # adhoc (0.4) is below the default 0.6 cutoff → excluded.
    assert res["counts"] == {"read": 1, "write": 1, "total": 2}
    services = {c["service"] for c in res["consumers"]}
    assert services == {"api", "etl"}
    assert "sql_fingerprint" in res["consumers"][0]


def test_consumers_op_filter(tmp_path: Path) -> None:
    _seed(tmp_path)
    res = consumers_payload(
        tmp_path, table="public.orders", database="demo", op="write"
    )
    assert [c["service"] for c in res["consumers"]] == ["etl"]
    assert res["counts"]["total"] == 1


def test_consumers_min_confidence_cutoff(tmp_path: Path) -> None:
    _seed(tmp_path)
    res = consumers_payload(
        tmp_path, table="public.orders", database="demo", min_confidence=0.0
    )
    # Lowering the cutoff surfaces the 0.4 adhoc consumer too.
    assert res["counts"]["total"] == 3


def test_consumers_unknown_table_errors(tmp_path: Path) -> None:
    _seed(tmp_path)
    res = consumers_payload(tmp_path, table="public.nope", database="demo")
    assert "error" in res
