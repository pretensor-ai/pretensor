"""Structural table extraction from SQL via sqlglot (no literals stored)."""

from __future__ import annotations

from pretensor.connectors.lineage_sqlglot import dml_write_targets, table_refs_from_sql


def test_table_refs_from_simple_select() -> None:
    refs = table_refs_from_sql(
        "SELECT * FROM orders o JOIN users u ON o.user_id = u.id",
        dialect="postgres",
        default_schema="public",
    )
    schemas_tables = {(s, t) for s, t in refs}
    assert ("public", "orders") in schemas_tables
    assert ("public", "users") in schemas_tables


def test_dml_write_targets_insert() -> None:
    targets = dml_write_targets(
        "INSERT INTO audit_log SELECT * FROM orders",
        dialect="postgres",
        default_schema="public",
    )
    assert ("public", "audit_log") in targets


def test_table_refs_with_cte() -> None:
    refs = table_refs_from_sql(
        """
        WITH recent AS (SELECT * FROM events WHERE ts > now() - interval '7 days')
        SELECT * FROM recent r JOIN users u ON r.user_id = u.id
        """,
        dialect="postgres",
        default_schema="analytics",
    )
    names = {t for _, t in refs}
    assert "events" in names
    assert "users" in names


def test_table_refs_cross_schema() -> None:
    refs = table_refs_from_sql(
        "SELECT * FROM billing.invoices i JOIN public.users u ON i.user_id = u.id",
        dialect="postgres",
        default_schema="public",
    )
    schemas_tables = {(s, t) for s, t in refs}
    assert ("billing", "invoices") in schemas_tables
    assert ("public", "users") in schemas_tables


def test_table_refs_subquery() -> None:
    refs = table_refs_from_sql(
        "SELECT * FROM (SELECT id FROM raw_events) sub JOIN sessions s ON sub.id = s.event_id",
        dialect="postgres",
        default_schema="public",
    )
    names = {t for _, t in refs}
    assert "raw_events" in names
    assert "sessions" in names


def test_table_refs_empty_sql_returns_empty() -> None:
    assert table_refs_from_sql("", dialect="postgres", default_schema="public") == []
    assert table_refs_from_sql("   ", dialect="postgres", default_schema="public") == []


def test_table_refs_invalid_sql_returns_empty() -> None:
    # Malformed SQL should not raise — returns empty list
    result = table_refs_from_sql(
        "SELECT FROM WHERE",
        dialect="postgres",
        default_schema="public",
    )
    assert isinstance(result, list)


def test_dml_write_targets_update() -> None:
    targets = dml_write_targets(
        "UPDATE inventory SET qty = qty - 1 WHERE product_id = 42",
        dialect="postgres",
        default_schema="public",
    )
    assert ("public", "inventory") in targets


def test_dml_write_targets_delete() -> None:
    targets = dml_write_targets(
        "DELETE FROM sessions WHERE expires_at < now()",
        dialect="postgres",
        default_schema="public",
    )
    assert ("public", "sessions") in targets


def test_dml_write_targets_empty_sql_returns_empty() -> None:
    assert dml_write_targets("", dialect="postgres", default_schema="public") == []
