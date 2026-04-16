"""MCP context: dbt enrichment fields and LINEAGE from enrichment."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.manifest import DbtManifest, DbtModel, DbtTest
from pretensor.enrichment.dbt.metadata import write_dbt_metadata
from pretensor.enrichment.dbt.signals import write_dbt_signals
from pretensor.graph_models.edge import LineageEdge
from pretensor.mcp.tools.context import context_payload


def test_context_surfaces_dbt_fields_and_lineage(tmp_path: Path) -> None:
    stg = Table(
        name="stg_orders",
        schema_name="public",
        columns=[Column(name="id", data_type="int")],
    )
    fct = Table(
        name="fct_orders",
        schema_name="public",
        columns=[Column(name="id", data_type="int")],
        comment="Legacy warehouse comment",
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[stg, fct],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)

        # Use the real dbt enrichment writers to populate description/tags,
        # test_count, and has_external_consumers — matching how the CLI path
        # mutates SchemaTable rows after indexing.
        manifest = DbtManifest(
            schema_version=None,
            nodes={
                "model.pkg.fct_orders": DbtModel(
                    unique_id="model.pkg.fct_orders",
                    name="fct_orders",
                    description="Orders fact from dbt — authoritative doc",
                    tags=("finance", "core"),
                    database="demo",
                    schema_name="public",
                    alias="fct_orders",
                ),
            },
            sources={},
            exposures={},
            tests={
                "test.pkg.t1": DbtTest(
                    unique_id="test.pkg.t1",
                    name="not_null_fct_orders_id",
                    attached_node="model.pkg.fct_orders",
                ),
                "test.pkg.t2": DbtTest(
                    unique_id="test.pkg.t2",
                    name="unique_fct_orders_id",
                    attached_node="model.pkg.fct_orders",
                ),
                "test.pkg.t3": DbtTest(
                    unique_id="test.pkg.t3",
                    name="relationships_fct_orders_id",
                    attached_node="model.pkg.fct_orders",
                ),
                "test.pkg.t4": DbtTest(
                    unique_id="test.pkg.t4",
                    name="accepted_values_fct_orders_id",
                    attached_node="model.pkg.fct_orders",
                ),
            },
            parent_map={},
        )
        write_dbt_metadata(manifest, store, "demo")
        write_dbt_signals(manifest, store, "demo")

        # has_external_consumers is computed from exposures; manifest has no
        # exposure in this test, so set it directly via Cypher to cover the
        # payload surface.
        store.execute_write(
            """
            MATCH (t:SchemaTable {connection_name: 'demo', schema_name: 'public', table_name: 'fct_orders'})
            SET t.has_external_consumers = true
            """,
            {},
        )

        store.upsert_lineage_edge(
            LineageEdge(
                edge_id="dbt1",
                source_node_id=table_node_id("demo", "public", "stg_orders"),
                target_node_id=table_node_id("demo", "public", "fct_orders"),
                source="dbt:manifest",
                lineage_type="MODEL",
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

    ctx = context_payload(tmp_path, table="public.fct_orders", db="demo", detail="standard")
    assert ctx.get("error") is None
    assert ctx["description"] == "Orders fact from dbt — authoritative doc"
    assert set(ctx["tags"]) == {"finance", "core"}
    assert ctx["has_external_consumers"] is True
    assert ctx["test_count"] == 4
    inn = ctx.get("lineage_in", [])
    assert any("stg_orders" in str(x.get("table", "")) for x in inn)

    ctx_stg = context_payload(tmp_path, table="stg_orders", db="demo", detail="standard")
    assert "tags" not in ctx_stg
    assert "has_external_consumers" not in ctx_stg
    assert "test_count" not in ctx_stg
