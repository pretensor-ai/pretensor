"""MCP context: LINEAGE edges and deprecation signal."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.graph_models.edge import LineageEdge
from pretensor.mcp.tools.context import context_payload


def test_context_lineage_in_out_and_deprecation(tmp_path: Path) -> None:
    tables = [
        Table(
            name="orders",
            schema_name="public",
            columns=[Column(name="id", data_type="int")],
            potentially_unused=True,
            days_since_last_access=100,
        ),
        Table(
            name="v_orders",
            schema_name="public",
            columns=[Column(name="id", data_type="int")],
            table_type="view",
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        store.upsert_lineage_edge(
            LineageEdge(
                edge_id="manual1",
                source_node_id=table_node_id("demo", "public", "orders"),
                target_node_id=table_node_id("demo", "public", "v_orders"),
                source="demo:public.v_orders",
                lineage_type="VIEW",
                confidence=1.0,
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

    ctx_o = context_payload(tmp_path, table="public.orders", db="demo", detail="standard")
    assert ctx_o.get("deprecation_signal")
    assert "deprecated" in str(ctx_o["deprecation_signal"]).lower()
    assert len(ctx_o.get("lineage_in", [])) == 0
    out = ctx_o.get("lineage_out", [])
    assert any("v_orders" in str(x.get("table", "")) for x in out)

    ctx_v = context_payload(tmp_path, table="v_orders", db="demo", detail="standard")
    inn = ctx_v.get("lineage_in", [])
    assert any("orders" in str(x.get("table", "")) for x in inn)
    assert "Lineage" in ctx_v.get("lineage_markdown", "")


def test_context_lineage_fields_omitted_when_empty(tmp_path: Path) -> None:
    """Tables with no lineage edges on either side drop the lineage fields
    entirely rather than returning empty arrays (which read as broken)."""
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[
            Table(
                name="standalone",
                schema_name="public",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
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

    ctx = context_payload(tmp_path, table="standalone", db="demo")
    assert "error" not in ctx
    assert "lineage_in" not in ctx
    assert "lineage_out" not in ctx
    assert "lineage_markdown" not in ctx
