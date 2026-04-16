"""Unit tests for SnowflakeConnector.get_table_grants (OBJECT_PRIVILEGES SELECT)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import ProgrammingError

from pretensor.connectors.base import TableGrant
from pretensor.connectors.snowflake import SnowflakeConnector
from pretensor.introspection.models.config import (
    ConnectionConfig,
    DatabaseType,
    SchemaFilter,
)


def _config() -> ConnectionConfig:
    return ConnectionConfig(
        name="t",
        type=DatabaseType.SNOWFLAKE,
        host="acct",
        user="u",
        password="p",
        database="MYDB",
        schema_filter=SchemaFilter(),
    )


def _make_engine_mock(
    priv_rows: list[dict[str, Any]],
    execute_error: Exception | None = None,
) -> MagicMock:
    """Return a mock Engine whose connect() runs OBJECT_PRIVILEGES SELECT."""

    def execute_side_effect(stmt: Any, *_a: Any, **_kw: Any) -> MagicMock:
        if execute_error is not None:
            raise execute_error
        sql = str(stmt)
        result = MagicMock()
        if "OBJECT_PRIVILEGES" in sql.upper():
            result.mappings.return_value.all.return_value = priv_rows
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


def test_get_table_grants_direct_select_rows() -> None:
    connector = SnowflakeConnector(_config())
    priv = [
        {
            "grantee_name": "ANALYST_ROLE",
            "table_schema": "SALES",
            "table_name": "ORDERS",
        },
    ]
    connector._engine = _make_engine_mock(priv)
    grants = connector.get_table_grants()
    assert grants == [
        TableGrant(
            grantee="ANALYST_ROLE",
            schema_name="SALES",
            table_name="ORDERS",
        ),
    ]


def test_get_table_grants_deduplicates_and_sorts() -> None:
    connector = SnowflakeConnector(_config())
    priv = [
        {
            "grantee_name": "R1",
            "table_schema": "A",
            "table_name": "T1",
        },
        {
            "grantee_name": "R1",
            "table_schema": "A",
            "table_name": "T1",
        },
        {
            "grantee_name": "R2",
            "table_schema": "A",
            "table_name": "T1",
        },
    ]
    connector._engine = _make_engine_mock(priv)
    grants = connector.get_table_grants()
    assert grants == [
        TableGrant(grantee="R1", schema_name="A", table_name="T1"),
        TableGrant(grantee="R2", schema_name="A", table_name="T1"),
    ]


def test_get_table_grants_skips_information_schema_and_system() -> None:
    connector = SnowflakeConnector(_config())
    priv = [
        {
            "grantee_name": "R1",
            "table_schema": "INFORMATION_SCHEMA",
            "table_name": "X",
        },
        {
            "grantee_name": "R2",
            "table_schema": "PUBLIC",
            "table_name": "Y",
        },
    ]
    connector._engine = _make_engine_mock(priv)
    grants = connector.get_table_grants()
    assert grants == [
        TableGrant(grantee="R2", schema_name="PUBLIC", table_name="Y"),
    ]


def test_get_table_grants_schema_filter_include() -> None:
    cfg = _config()
    cfg.schema_filter = SchemaFilter(include=["APP"])
    connector = SnowflakeConnector(cfg)
    priv = [
        {
            "grantee_name": "A",
            "table_schema": "PUBLIC",
            "table_name": "T",
        },
        {
            "grantee_name": "B",
            "table_schema": "APP",
            "table_name": "T",
        },
    ]
    connector._engine = _make_engine_mock(priv)
    grants = connector.get_table_grants()
    assert len(grants) == 1
    assert grants[0].schema_name == "APP"


def test_get_table_grants_permission_error_returns_empty_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")
    connector = SnowflakeConnector(_config())
    priv = [
        {
            "grantee_name": "R",
            "table_schema": "S",
            "table_name": "T",
        },
    ]
    connector._engine = _make_engine_mock(
        priv,
        execute_error=ProgrammingError("stmt", {}, orig=Exception("permission denied")),
    )
    grants = connector.get_table_grants()
    assert grants == []
    assert "object_privileges" in caplog.text.lower()


def test_get_table_grants_skips_null_cells() -> None:
    connector = SnowflakeConnector(_config())
    priv = [
        {
            "grantee_name": None,
            "table_schema": "S",
            "table_name": "T",
        },
        {
            "grantee_name": "OK",
            "table_schema": "S",
            "table_name": "T2",
        },
    ]
    connector._engine = _make_engine_mock(priv)
    grants = connector.get_table_grants()
    assert len(grants) == 1
    assert grants[0].table_name == "T2"
