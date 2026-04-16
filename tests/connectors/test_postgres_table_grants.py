"""Unit tests for PostgresConnector.get_table_grants (SELECT + role inheritance)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import ProgrammingError

from pretensor.connectors.base import TableGrant
from pretensor.connectors.postgres import PostgresConnector, _pg_transitive_members
from pretensor.introspection.models.config import (
    ConnectionConfig,
    DatabaseType,
    SchemaFilter,
)


def _config() -> ConnectionConfig:
    return ConnectionConfig(
        name="t",
        type=DatabaseType.POSTGRES,
        host="localhost",
        port=5432,
        user="u",
        password="p",
        database="db",
        schema_filter=SchemaFilter(),
    )


def _make_engine_mock(
    priv_rows: list[dict[str, Any]],
    member_rows: list[dict[str, Any]] | None,
    member_error: Exception | None = None,
) -> MagicMock:
    """Return a mock Engine whose connect() runs SELECT then pg_auth_members (or error)."""

    def execute_side_effect(stmt: Any, *_a: Any, **_kw: Any) -> MagicMock:
        sql = str(stmt)
        result = MagicMock()
        if "information_schema.table_privileges" in sql:
            result.mappings.return_value.all.return_value = priv_rows
            return result
        if "pg_auth_members" in sql:
            if member_error is not None:
                raise member_error
            result.mappings.return_value.all.return_value = member_rows or []
            return result
        raise AssertionError(f"unexpected SQL: {sql[:200]}")

    conn = MagicMock()
    conn.execute.side_effect = execute_side_effect

    engine = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = None
    engine.connect.return_value = cm
    return engine


def test_pg_transitive_members_chain() -> None:
    children = {"analyst": {"bob"}, "bob": {"charlie"}}
    assert _pg_transitive_members(children, "analyst") == {"analyst", "bob", "charlie"}


def test_get_table_grants_direct_only() -> None:
    connector = PostgresConnector(_config())
    priv = [
        {"grantee": "app_reader", "table_schema": "public", "table_name": "orders"},
    ]
    member = []  # no inheritance edges
    engine = _make_engine_mock(priv, member)
    connector._engine = engine
    grants = connector.get_table_grants()
    assert grants == [
        TableGrant(grantee="app_reader", schema_name="public", table_name="orders"),
    ]


def test_get_table_grants_inherited_via_membership() -> None:
    connector = PostgresConnector(_config())
    priv = [
        {"grantee": "parent_role", "table_schema": "public", "table_name": "items"},
    ]
    member = [
        {"member_name": "child_user", "role_name": "parent_role"},
    ]
    engine = _make_engine_mock(priv, member)
    connector._engine = engine
    grants = connector.get_table_grants()
    assert len(grants) == 2
    assert TableGrant(
        grantee="parent_role", schema_name="public", table_name="items"
    ) in grants
    assert TableGrant(
        grantee="child_user", schema_name="public", table_name="items"
    ) in grants


def test_get_table_grants_transitive_membership() -> None:
    connector = PostgresConnector(_config())
    priv = [
        {"grantee": "r1", "table_schema": "app", "table_name": "t1"},
    ]
    member = [
        {"member_name": "r2", "role_name": "r1"},
        {"member_name": "enduser", "role_name": "r2"},
    ]
    engine = _make_engine_mock(priv, member)
    connector._engine = engine
    grants = connector.get_table_grants()
    grantees = {g.grantee for g in grants}
    assert grantees == {"r1", "r2", "enduser"}


def test_get_table_grants_skips_public_and_system_schema() -> None:
    connector = PostgresConnector(_config())
    priv = [
        {"grantee": "PUBLIC", "table_schema": "public", "table_name": "x"},
        {"grantee": "u1", "table_schema": "pg_catalog", "table_name": "y"},
        {"grantee": "u2", "table_schema": "public", "table_name": "z"},
    ]
    engine = _make_engine_mock(priv, [])
    connector._engine = engine
    grants = connector.get_table_grants()
    assert [g.table_name for g in grants] == ["z"]


def test_get_table_grants_schema_filter_include() -> None:
    cfg = _config()
    cfg.schema_filter = SchemaFilter(include=["app"])
    connector = PostgresConnector(cfg)
    priv = [
        {"grantee": "a", "table_schema": "public", "table_name": "t"},
        {"grantee": "b", "table_schema": "app", "table_name": "t"},
    ]
    engine = _make_engine_mock(priv, [])
    connector._engine = engine
    grants = connector.get_table_grants()
    assert len(grants) == 1
    assert grants[0].schema_name == "app"


def test_get_table_grants_pg_auth_members_permission_falls_back_to_direct(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """F2: a membership-query failure must not discard already-fetched direct grants.

    Previously, the Postgres connector returned ``[]`` if the ``pg_auth_members``
    query failed (e.g. an admin role without ``pg_catalog`` access). That
    silently wiped out all direct grants it had already read successfully. The
    connector now degrades gracefully: log a warning and yield each direct
    grantee one-to-one (no inherited-role expansion).
    """
    caplog.set_level("WARNING")
    connector = PostgresConnector(_config())
    priv = [
        {"grantee": "direct_reader", "table_schema": "public", "table_name": "orders"},
        {"grantee": "other_reader", "table_schema": "app", "table_name": "items"},
    ]
    engine = _make_engine_mock(
        priv,
        member_rows=None,
        member_error=ProgrammingError("stmt", {}, orig=Exception("permission denied")),
    )
    connector._engine = engine
    grants = connector.get_table_grants()
    assert grants == [
        TableGrant(grantee="other_reader", schema_name="app", table_name="items"),
        TableGrant(
            grantee="direct_reader", schema_name="public", table_name="orders"
        ),
    ]
    assert "pg_auth_members" in caplog.text
    assert "direct grants only" in caplog.text


def test_get_table_grants_privileges_query_fails_returns_empty_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")
    connector = PostgresConnector(_config())
    engine = MagicMock()
    conn = MagicMock()
    conn.execute.side_effect = ProgrammingError("stmt", {}, orig=Exception("denied"))
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = None
    engine.connect.return_value = cm

    connector._engine = engine
    grants = connector.get_table_grants()
    assert grants == []
    assert "table_privileges" in caplog.text
