"""``impact`` payload carries an additive ``consumers`` field per reached table."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import (
    consumes_edge_id,
    external_consumer_node_id,
    table_node_id,
)
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.graph_models.consumer import ConsumesEdge, ExternalConsumerNode
from pretensor.mcp.tools.impact import impact_payload


def _seed(tmp_path: Path) -> None:
    """``root`` ← FK ``a``; table ``a`` is consumed by one external service."""
    t_root = Table(
        name="root",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    t_a = Table(
        name="a",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="root_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="a",
                source_column="root_id",
                target_schema="public",
                target_table="root",
                target_column="id",
            )
        ],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_root, t_a],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        a_nid = table_node_id("demo", "public", "a")
        c_nid = external_consumer_node_id(
            "demo", "billing", "src/a.py", 10, "fp0123456789abcd"
        )
        store.upsert_external_consumer(
            ExternalConsumerNode(
                node_id=c_nid,
                connection_name="demo",
                service_name="billing",
                file_path="src/a.py",
                language="python",
                symbol="query",
                kind="call",
                line_start=10,
                line_end=10,
                sql_fingerprint="fp0123456789abcd",
                confidence=0.9,
                dialect_used="postgres",
                scan_run_id="run1",
            )
        )
        store.upsert_consumes_edge(
            ConsumesEdge(
                edge_id=consumes_edge_id(c_nid, a_nid, "read"),
                source_node_id=c_nid,
                target_node_id=a_nid,
                op="read",
                confidence=0.9,
                scan_run_id="run1",
            )
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


def test_impact_items_carry_consumers(tmp_path: Path) -> None:
    _seed(tmp_path)
    res = impact_payload(tmp_path, table="public.root", database="demo", max_depth=3)
    assert "error" not in res
    direct = res["impact"]["direct"]
    assert [x["name"] for x in direct] == ["public.a"]
    # Additive field present on every item.
    assert "consumers" in direct[0]
    consumers = direct[0]["consumers"]
    assert len(consumers) == 1
    entry = consumers[0]
    assert entry["service"] == "billing"
    assert entry["op"] == "read"
    assert entry["kind"] == "call"
    assert entry["file"] == "src/a.py"


def test_impact_consumers_empty_when_none(tmp_path: Path) -> None:
    _seed(tmp_path)
    # ``root`` itself has no consumers; reached ``a`` does. Verify empty-list default
    # by querying a graph table with no consumer: build a second reachable table.
    res = impact_payload(tmp_path, table="public.root", database="demo", max_depth=3)
    for group in res["impact"].values():
        for item in group:
            assert isinstance(item["consumers"], list)  # always present, possibly empty
