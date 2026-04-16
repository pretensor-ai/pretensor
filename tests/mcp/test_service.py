"""Tests for MCP service helpers (list, query, context, resources)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import (
    dbt_lineage_edge_id,
    entity_node_id,
    metric_template_node_id,
    table_node_id,
)
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.graph_models.edge import LineageEdge
from pretensor.graph_models.entity import EntityNode
from pretensor.mcp.service import (
    context_payload,
    cypher_payload,
    databases_resource_markdown,
    list_databases_payload,
    metrics_resource_markdown,
    query_payload,
)
from pretensor.mcp.tools.cypher import assert_read_only_cypher


def _write_minimal_registry(tmp_path: Path, *, stale: bool = False) -> None:
    t_orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="user_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="orders",
                source_column="user_id",
                target_schema="public",
                target_table="users",
                target_column="id",
            )
        ],
        comment="Customer orders",
    )
    t_users = Table(
        name="users",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_orders, t_users],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()

    when = datetime.now(timezone.utc)
    if stale:
        when = when - timedelta(days=10)
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://localhost/demo",
        graph_path=graph,
        indexed_at=when,
    )
    reg.save()


def test_assert_read_only_cypher_allows_match() -> None:
    assert_read_only_cypher("MATCH (t:SchemaTable) RETURN t.table_name LIMIT 1")


def test_assert_read_only_cypher_rejects_create() -> None:
    with pytest.raises(ValueError, match="read queries"):
        assert_read_only_cypher("CREATE (n:Foo)")


def test_assert_read_only_cypher_ignores_strings() -> None:
    assert_read_only_cypher(
        "MATCH (t:SchemaTable) WHERE t.comment CONTAINS 'DELETE old rows' RETURN t"
    )


def test_cypher_tool_returns_rows(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = cypher_payload(
        tmp_path,
        query="MATCH (t:SchemaTable) RETURN t.table_name AS name ORDER BY name",
        database="demo",
    )
    assert "error" not in out
    assert out["rows"] == [{"name": "orders"}, {"name": "users"}]


def test_cypher_tool_rejects_delete(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = cypher_payload(
        tmp_path,
        query="MATCH (t:SchemaTable) DELETE t",
        database="demo",
    )
    assert "error" in out
    assert "read" in out["error"].lower()


def test_cypher_tool_unknown_database(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = cypher_payload(
        tmp_path,
        query="RETURN 1",
        database="nope",
    )
    assert out.get("error", "").startswith("Unknown database")


def _write_second_graph(tmp_path: Path, connection: str) -> None:
    t = Table(
        name="t",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name=connection,
        database=connection,
        schemas=["public"],
        tables=[t],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / f"{connection}.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name=connection,
        database=connection,
        dsn=f"postgresql://localhost/{connection}",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()


def test_cypher_auto_picks_sole_indexed_graph(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = cypher_payload(
        tmp_path,
        query="MATCH (t:SchemaTable) RETURN t.table_name AS name ORDER BY name",
    )
    assert "error" not in out
    assert out["rows"] == [{"name": "orders"}, {"name": "users"}]
    assert "warning" in out
    assert "auto-selected" in out["warning"]


def test_cypher_auto_pick_emits_deprecation_warning_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_minimal_registry(tmp_path)
    with caplog.at_level("WARNING", logger="pretensor.mcp.tools.cypher"):
        cypher_payload(tmp_path, query="RETURN 1 AS n")
    assert any("deprecation shim" in r.message for r in caplog.records)


def test_cypher_hard_errors_when_multiple_graphs_indexed(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    _write_second_graph(tmp_path, "second")
    out = cypher_payload(tmp_path, query="RETURN 1 AS n")
    assert "error" in out
    err = out["error"]
    assert "multiple graphs" in err
    assert "demo" in err and "second" in err


def test_cypher_hard_errors_when_no_graphs_indexed(tmp_path: Path) -> None:
    (tmp_path / "registry.json").write_text("{}")
    out = cypher_payload(tmp_path, query="RETURN 1 AS n")
    assert "error" in out
    assert "no graphs" in out["error"].lower()


def test_list_databases_payload(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    data = list_databases_payload(tmp_path)
    assert len(data["databases"]) == 1
    row = data["databases"][0]
    assert row["name"] == "demo"
    assert row["db_type"] == "postgresql"
    assert row["table_count"] == 2
    assert row["column_count"] == 3
    assert row["row_count"] == 0
    assert row["is_stale"] is False
    # When logical database == connection name, the redundant "database" field
    # is omitted (see docs/specs/mcp/spec.md Breaking changes).
    assert "database" not in row
    # Capability flags: empty graph has one schema and no enrichment.
    assert row["schemas"] == ["public"]
    # Trinary enum: entry has no dbt_manifest_path / llm_enrichment_ran=False
    # and no external-consumer flag, so all three surface as "not_attempted"
    # (unified default; callers don't have to special-case external_consumers).
    assert row["has_dbt_manifest"] == "not_attempted"
    assert row["has_llm_enrichment"] == "not_attempted"
    assert row["has_external_consumers"] == "not_attempted"
    # Baseline capabilities always present; no enrichment extensions.
    assert row["capabilities"] == [
        "schema",
        "fk_edges",
        "inferred_joins",
        "clustering",
    ]


def test_list_databases_payload_surfaces_capability_flags(tmp_path: Path) -> None:
    """A graph with a second schema, dbt lineage, an Entity, and a table flagged
    ``has_external_consumers`` surfaces all four capability fields as populated."""
    t_orders = Table(
        name="orders",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    t_metrics = Table(
        name="daily_sales",
        schema_name="analytics",
        columns=[Column(name="day", data_type="date", is_primary_key=True)],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public", "analytics"],
        tables=[t_orders, t_metrics],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        store.upsert_entity(
            EntityNode(
                node_id=entity_node_id("demo", "Order"),
                connection_name="demo",
                database="demo",
                name="Order",
                description="Customer orders entity",
            )
        )
        # dbt parent_map → LINEAGE edge tagged model_dependency (analytics depends on public).
        store.upsert_lineage_edge(
            LineageEdge(
                edge_id=dbt_lineage_edge_id(
                    "demo",
                    "model.demo.orders",
                    "model.demo.daily_sales",
                ),
                source_node_id=table_node_id("demo", "public", "orders"),
                target_node_id=table_node_id("demo", "analytics", "daily_sales"),
                source="dbt",
                lineage_type="model_dependency",
                confidence=1.0,
            )
        )
        # Mark one table as feeding external consumers (e.g. dbt exposure).
        store._conn.execute(
            "MATCH (t:SchemaTable {node_id: $nid}) SET t.has_external_consumers = true",
            {"nid": table_node_id("demo", "analytics", "daily_sales")},
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
        dbt_manifest_path="/tmp/manifest.json",
        llm_enrichment_ran=True,
    )
    reg.save()

    data = list_databases_payload(tmp_path)
    row = data["databases"][0]
    assert row["schemas"] == ["analytics", "public"]
    assert row["has_dbt_manifest"] == "present"
    assert row["has_llm_enrichment"] == "present"
    assert row["has_external_consumers"] == "present"
    # Every "present" flag contributes a capability; dbt contributes two.
    caps = row["capabilities"]
    assert caps[:4] == ["schema", "fk_edges", "inferred_joins", "clustering"]
    assert "dbt_lineage" in caps
    assert "dbt_metrics" in caps
    assert "llm_enrichment" in caps
    assert "external_consumers" in caps


def test_databases_resource_markdown_renders_capability_flags(tmp_path: Path) -> None:
    """The ``pretensor://databases`` markdown resource mirrors the new fields."""
    _write_minimal_registry(tmp_path)
    md = databases_resource_markdown(tmp_path)
    assert "**Schemas:** public" in md
    assert "**dbt manifest:** not_attempted" in md
    assert "**LLM enrichment:** not_attempted" in md
    assert "**External consumers:** not_attempted" in md


def test_list_databases_snowflake_db_type(tmp_path: Path) -> None:
    """Snowflake connections report db_type='snowflake', not 'postgresql'."""
    t = Table(
        name="events",
        schema_name="raw",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="sf_warehouse",
        database="analytics",
        schemas=["raw"],
        tables=[t],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "sf_warehouse.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="sf_warehouse",
        database="analytics",
        dsn="snowflake://account/analytics",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
        dialect="snowflake",
    )
    reg.save()

    data = list_databases_payload(tmp_path)
    row = data["databases"][0]
    assert row["name"] == "sf_warehouse"
    assert row["db_type"] == "snowflake"
    assert row["column_count"] == 1
    # connection_name differs from logical database — both surface.
    assert row["database"] == "analytics"


def test_context_composite_fk_grouped(tmp_path: Path) -> None:
    """Composite FK edges should be grouped into a single relationship row."""
    tables = [
        Table(
            name="lineitem",
            schema_name="public",
            columns=[
                Column(name="l_partkey", data_type="int"),
                Column(name="l_suppkey", data_type="int"),
            ],
            foreign_keys=[
                ForeignKey(
                    source_schema="public",
                    source_table="lineitem",
                    source_column="l_partkey",
                    target_schema="public",
                    target_table="partsupp",
                    target_column="ps_partkey",
                    constraint_name="fk_lineitem_partsupp",
                ),
                ForeignKey(
                    source_schema="public",
                    source_table="lineitem",
                    source_column="l_suppkey",
                    target_schema="public",
                    target_table="partsupp",
                    target_column="ps_suppkey",
                    constraint_name="fk_lineitem_partsupp",
                ),
            ],
        ),
        Table(
            name="partsupp",
            schema_name="public",
            columns=[
                Column(name="ps_partkey", data_type="int"),
                Column(name="ps_suppkey", data_type="int"),
            ],
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="tpch",
        database="tpch",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "tpch.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="tpch",
        database="tpch",
        dsn="postgresql://localhost/tpch",
        graph_path=graph,
    )
    reg.save()
    ctx = context_payload(tmp_path, table="public.lineitem", db="tpch")
    fk_rels = [r for r in ctx["relationships"] if r["rel_type"] == "FK_REFERENCES"]
    # Composite FK should be collapsed into one row
    assert len(fk_rels) == 1
    rel = fk_rels[0]
    assert rel["constraint_name"] == "fk_lineitem_partsupp"
    # Deterministic ordering — sorted by (source_column, target_column) — with pairing preserved.
    assert rel["source_columns"] == ["l_partkey", "l_suppkey"]
    assert rel["target_columns"] == ["ps_partkey", "ps_suppkey"]
    # i-th source pairs with i-th target per the FK definition.
    pairs = list(zip(rel["source_columns"], rel["target_columns"]))
    assert pairs == [("l_partkey", "ps_partkey"), ("l_suppkey", "ps_suppkey")]


def test_query_tool_finds_table_comment(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = query_payload(tmp_path, q="orders", limit=5)
    assert out["results"]
    first = out["results"][0]
    assert "orders" in first["name"].lower() or "order" in first["name"].lower()


def test_query_tool_returns_schema_table_hits_only(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = query_payload(tmp_path, q="customer", limit=10)
    assert out["results"]
    assert all(r["node_type"] == "SchemaTable" for r in out["results"])


def test_query_payload_emits_timing_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """query_payload emits a structured timing event for MCP observability."""
    _write_minimal_registry(tmp_path)
    with caplog.at_level("INFO"):
        out = query_payload(tmp_path, q="orders", limit=5)
    assert out["results"]
    matched = [r for r in caplog.records if getattr(r, "event", "") == "mcp.query_payload"]
    assert matched
    rec = matched[-1]
    assert getattr(rec, "status", None) == "ok"
    assert getattr(rec, "limit", None) == 5


def test_context_film_pagila_shape(tmp_path: Path) -> None:
    """Pagila-style names: film + rental with FK-style names (no real FK in snapshot)."""
    tables = [
        Table(
            name="film",
            schema_name="public",
            columns=[Column(name="film_id", data_type="int")],
            comment="Films in inventory",
        ),
        Table(
            name="rental",
            schema_name="public",
            columns=[
                Column(name="rental_id", data_type="int"),
                Column(name="inventory_id", data_type="int"),
            ],
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="pagila",
        database="pagila",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "pagila.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="pagila",
        database="pagila",
        dsn="postgresql://x",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()

    ctx = context_payload(tmp_path, table="film", db="pagila", detail="standard")
    assert "error" not in ctx
    assert ctx["table_name"] == "film"
    assert ctx["schema_name"] == "public"
    assert "Films" in ctx["description"] or "inventory" in ctx["description"].lower()
    assert len(ctx["columns"]) == 1
    col0 = ctx["columns"][0]
    assert col0["column_name"] == "film_id"
    assert col0["name"] == "film_id"
    assert "nullable" in col0
    assert "is_primary_key" in col0
    assert "is_foreign_key" in col0


def test_context_staleness_warning(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path, stale=True)
    ctx = context_payload(tmp_path, table="public.orders", db="demo")
    assert ctx.get("staleness_warning")


def test_metrics_resource_markdown_lists_templates(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    graph = tmp_path / "graphs" / "demo.kuzu"
    store = KuzuStore(graph)
    try:
        store.ensure_schema()
        store.upsert_metric_template(
            node_id=metric_template_node_id("demo", "demo", "sum_orders"),
            connection_name="demo",
            database="demo",
            name="sum_orders",
            display_name="Order count proxy",
            description="total revenue metric for testing",
            sql_template="SELECT COUNT(*) AS n FROM public.orders",
            tables_used=["public.orders"],
            validated=True,
            validation_errors=[],
            generated_at_iso="2026-01-01T00:00:00+00:00",
            stale=False,
            depends_on_table_node_ids=["demo::public::orders"],
        )
    finally:
        store.close()
    md = metrics_resource_markdown(tmp_path, "demo")
    assert "sum_orders" in md
    assert "SELECT COUNT" in md
    assert "public.orders" in md


def test_databases_resource_markdown(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    md = databases_resource_markdown(tmp_path)
    assert "demo" in md
    assert "Indexed databases" in md


def test_metrics_resource_markdown_shows_dialect(tmp_path: Path) -> None:
    """The metrics resource markdown should include the SQL dialect for each template."""
    _write_minimal_registry(tmp_path)
    graph = tmp_path / "graphs" / "demo.kuzu"
    store = KuzuStore(graph)
    try:
        store.ensure_schema()
        store.upsert_metric_template(
            node_id=metric_template_node_id("demo", "demo", "dialect_test"),
            connection_name="demo",
            database="demo",
            dialect="postgresql",
            name="dialect_test",
            display_name="Dialect check",
            description="Verify dialect is shown",
            sql_template="SELECT COUNT(*) FROM public.orders",
            tables_used=["public.orders"],
            validated=True,
            validation_errors=[],
            generated_at_iso="2026-01-01T00:00:00+00:00",
            stale=False,
            depends_on_table_node_ids=["demo::public::orders"],
        )
    finally:
        store.close()

    md = metrics_resource_markdown(tmp_path, "demo")
    assert "postgresql" in md


# ---------------------------------------------------------------------------
# A1 — context INFERRED edges carry source_columns / target_columns
# ---------------------------------------------------------------------------


def test_context_inferred_edge_has_column_lists(tmp_path: Path) -> None:
    """INFERRED_JOIN relationship rows should populate source_columns / target_columns
    list fields (not only the singular scalars)."""
    # Two tables sharing column 'customer_id' with no explicit FK → heuristic
    # discovery will emit an INFERRED_JOIN on the shared name.
    t_orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="customer_id", data_type="int"),
        ],
        foreign_keys=[],
    )
    t_shipments = Table(
        name="shipments",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="customer_id", data_type="int"),
        ],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_orders, t_shipments],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=True)
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

    out = context_payload(tmp_path, table="public.orders", db="demo")
    inferred = [
        r for r in out.get("relationships", []) if r.get("rel_type") == "INFERRED_JOIN"
    ]
    assert inferred, "expected at least one INFERRED_JOIN on shared 'customer_id'"
    row = inferred[0]
    # Legacy scalar fields still populated for back-compat.
    assert row["source_column"] == "customer_id"
    assert row["target_column"] == "customer_id"
    # New list fields (A1).
    assert row["source_columns"] == ["customer_id"]
    assert row["target_columns"] == ["customer_id"]


# ---------------------------------------------------------------------------
# A2 — weak-column block-list drops rating / length / active
# ---------------------------------------------------------------------------


def test_context_drops_inferred_edges_on_weak_shared_columns(tmp_path: Path) -> None:
    """Shared columns 'rating', 'length', 'active' are in _BORING_SHARED_NAMES
    and should NOT produce INFERRED_JOIN rows (pagila-style noise)."""
    t_film = Table(
        name="film",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="rating", data_type="text"),
            Column(name="length", data_type="int"),
        ],
        foreign_keys=[],
    )
    t_category = Table(
        name="category",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="rating", data_type="text"),
        ],
        foreign_keys=[],
    )
    t_store = Table(
        name="store",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="active", data_type="bool"),
        ],
        foreign_keys=[],
    )
    t_customer = Table(
        name="customer",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="active", data_type="bool"),
            Column(name="length", data_type="int"),
        ],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_film, t_category, t_store, t_customer],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=True)
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

    out = context_payload(tmp_path, table="public.film", db="demo")
    inferred_cols = {
        r.get("source_column")
        for r in out.get("relationships", [])
        if r.get("rel_type") == "INFERRED_JOIN"
    }
    assert "rating" not in inferred_cols
    assert "length" not in inferred_cols
    assert "active" not in inferred_cols


# ---------------------------------------------------------------------------
# A3 — Snowflake case-insensitive table resolution
# ---------------------------------------------------------------------------


def test_context_resolves_lowercase_request_against_uppercase_snowflake(
    tmp_path: Path,
) -> None:
    """Requesting 'lineitem' should resolve to 'TPCH.LINEITEM' on Snowflake-style
    graphs where identifiers are stored uppercase."""
    t = Table(
        name="LINEITEM",
        schema_name="TPCH",
        columns=[Column(name="L_ORDERKEY", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="sf",
        database="analytics",
        schemas=["TPCH"],
        tables=[t],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "sf.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="sf",
        database="analytics",
        dsn="snowflake://account/analytics",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
        dialect="snowflake",
    )
    reg.save()

    # Bare lowercase name — should resolve via case-insensitive fallback.
    out = context_payload(tmp_path, table="lineitem", db="sf")
    assert "error" not in out, out
    assert out.get("qualified_name") == "TPCH.LINEITEM"

    # Qualified lowercase name — same.
    out2 = context_payload(tmp_path, table="tpch.lineitem", db="sf")
    assert "error" not in out2, out2
    assert out2.get("qualified_name") == "TPCH.LINEITEM"
