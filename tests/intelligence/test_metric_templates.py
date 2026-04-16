"""Tests for heuristic MetricTemplate generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.intelligence.metric_templates import MetricTemplateBuilder


def _fact_store(tmp_path: Path) -> KuzuStore:
    """Build a minimal shop graph with a fact-classified orders table."""
    tables = [
        Table(
            name="orders",
            schema_name="public",
            columns=[
                Column(name="order_id", data_type="int", is_primary_key=True),
                Column(name="total_amount", data_type="numeric"),
                Column(name="tax_amount", data_type="numeric"),
                Column(name="qty", data_type="int"),
            ],
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="shop",
        database="shop",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "shop.kuzu"
    store = KuzuStore(graph)
    GraphBuilder().build(snap, store, run_relationship_discovery=False)
    store.set_table_classification(
        table_node_id="shop::public::orders",
        role="fact",
        role_confidence=0.9,
        classification_signals_json='["test"]',
    )
    return store


def test_metric_template_builder_creates_three_for_fact_metrics(tmp_path: Path) -> None:
    store = _fact_store(tmp_path)
    try:
        n = MetricTemplateBuilder(store).build("shop")
        assert n >= 3
        rows = store.query_all_rows(
            """
            MATCH (m:MetricTemplate {database: $db})
            RETURN m.validated
            """,
            {"db": "shop"},
        )
        assert len(rows) >= 3
        assert all(bool(r[0]) for r in rows)
    finally:
        store.close()


def test_metric_template_builder_stores_dialect(tmp_path: Path) -> None:
    store = _fact_store(tmp_path)
    try:
        MetricTemplateBuilder(store).build("shop")
        rows = store.query_all_rows(
            """
            MATCH (m:MetricTemplate {database: $db})
            RETURN m.dialect
            """,
            {"db": "shop"},
        )
        assert rows, "expected at least one MetricTemplate"
        assert all(str(r[0]) == "postgresql" for r in rows)
    finally:
        store.close()


def test_metric_template_builder_skips_non_fact_tables(tmp_path: Path) -> None:
    """Templates should only be generated for fact-classified tables."""
    tables = [
        Table(
            name="users",
            schema_name="public",
            columns=[
                Column(name="user_id", data_type="int", is_primary_key=True),
                Column(name="total_logins", data_type="int"),
            ],
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="app",
        database="app",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    store = KuzuStore(tmp_path / "app.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        # No classification set — role remains NULL, not "fact"
        n = MetricTemplateBuilder(store).build("app")
        assert n == 0
    finally:
        store.close()


def test_mark_stale_for_database_sets_stale_flag(tmp_path: Path) -> None:
    store = _fact_store(tmp_path)
    try:
        MetricTemplateBuilder(store).build("shop")
        # All templates start as not stale
        rows_before = store.query_all_rows(
            "MATCH (m:MetricTemplate {database: $db}) RETURN m.stale",
            {"db": "shop"},
        )
        assert rows_before and not any(bool(r[0]) for r in rows_before)

        MetricTemplateBuilder.mark_stale_for_database(store, "shop")

        rows_after = store.query_all_rows(
            "MATCH (m:MetricTemplate {database: $db}) RETURN m.stale",
            {"db": "shop"},
        )
        assert rows_after and all(bool(r[0]) for r in rows_after)
    finally:
        store.close()


def test_metric_template_builder_depends_on_edges(tmp_path: Path) -> None:
    """Each MetricTemplate should have a METRIC_DEPENDS edge to its source table."""
    store = _fact_store(tmp_path)
    try:
        MetricTemplateBuilder(store).build("shop")
        rows = store.query_all_rows(
            """
            MATCH (m:MetricTemplate {database: $db})-[:METRIC_DEPENDS]->(t:SchemaTable)
            RETURN m.name, t.table_name
            """,
            {"db": "shop"},
        )
        assert rows, "expected METRIC_DEPENDS edges"
        assert all(str(r[1]) == "orders" for r in rows)
    finally:
        store.close()
