"""Tests for QueryValidator (sqlglot + Kuzu semantic checks)."""

from __future__ import annotations

from datetime import datetime, timezone

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.validation.query_validator import QueryValidator


def _demo_snapshot(*, extra_tables: list[Table] | None = None) -> SchemaSnapshot:
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
    )
    t_users = Table(
        name="users",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    tables = [t_orders, t_users]
    if extra_tables:
        tables.extend(extra_tables)
    return SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )


def test_validate_valid_sql(tmp_path) -> None:
    snap = _demo_snapshot()
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        v = QueryValidator(store, connection_name="demo", database_key="demo")
        sql = """
        SELECT o.id
        FROM public.orders AS o
        JOIN public.users AS u ON o.user_id = u.id
        """
        r = v.validate(sql)
        assert r.valid
        assert r.syntax_errors == []
        assert r.missing_tables == []
        assert r.missing_columns == []
        assert r.invalid_joins == []
    finally:
        store.close()


def test_syntax_error(tmp_path) -> None:
    snap = _demo_snapshot()
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        v = QueryValidator(store, connection_name="demo", database_key="demo")
        r = v.validate("SELECT FROM")
        assert not r.valid
        assert r.syntax_errors
        assert r.missing_tables == []
    finally:
        store.close()


def test_missing_table(tmp_path) -> None:
    snap = _demo_snapshot()
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        v = QueryValidator(store, connection_name="demo", database_key="demo")
        r = v.validate("SELECT * FROM public.nonexistent_table")
        assert not r.valid
        assert "public.nonexistent_table" in r.missing_tables
    finally:
        store.close()


def test_missing_column_with_suggestion(tmp_path) -> None:
    snap = _demo_snapshot()
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        v = QueryValidator(store, connection_name="demo", database_key="demo")
        r = v.validate("SELECT o.usre_id FROM public.orders o")
        assert not r.valid
        assert any("usre_id" in c for c in r.missing_columns)
        assert any("user_id" in s for s in r.suggestions)
    finally:
        store.close()


def test_invalid_join_no_graph_edge(tmp_path) -> None:
    t_products = Table(
        name="products",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    snap = _demo_snapshot(extra_tables=[t_products])
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        v = QueryValidator(store, connection_name="demo", database_key="demo")
        sql = """
        SELECT u.id, p.id
        FROM public.users u
        JOIN public.products p ON u.id = p.id
        """
        r = v.validate(sql)
        assert not r.valid
        assert r.invalid_joins
        assert "public.users" in {
            r.invalid_joins[0].left_table,
            r.invalid_joins[0].right_table,
        }
    finally:
        store.close()
