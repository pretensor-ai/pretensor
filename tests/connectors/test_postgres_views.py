"""PostgresConnector view-row-count behavior with statement_timeout."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from sqlalchemy.exc import OperationalError

from pretensor.connectors.postgres import (
    VIEW_COUNT_TIMEOUT_MS,
    PostgresConnector,
)
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


def _make_get_tables_engine(
    list_rows: list[dict[str, Any]],
    *,
    view_count: int | None = None,
    view_count_error: Exception | None = None,
) -> tuple[MagicMock, list[str]]:
    """Mock engine that returns ``list_rows`` from get_tables() and supports
    a follow-up ``SELECT COUNT(*) FROM <view>`` (or raises ``view_count_error``).

    Returns ``(engine_mock, captured_sql)`` so tests can assert on issued SQL.
    """
    captured: list[str] = []

    def execute_side_effect(stmt: Any, *_a: Any, **_kw: Any) -> MagicMock:
        sql = str(stmt)
        captured.append(sql)
        result = MagicMock()
        if "information_schema.tables" in sql:
            result.mappings.return_value.all.return_value = list_rows
            return result
        if "SET LOCAL statement_timeout" in sql:
            return result
        if "SELECT COUNT(*)" in sql:
            if view_count_error is not None:
                raise view_count_error
            row = {"n": view_count}
            result.mappings.return_value.first.return_value = row
            return result
        result.mappings.return_value.all.return_value = []
        return result

    conn = MagicMock()
    conn.execute.side_effect = execute_side_effect
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    engine = MagicMock()
    engine.connect.return_value = conn
    engine.begin.return_value = conn
    return engine, captured


def _view_row(name: str = "vmovies") -> dict[str, Any]:
    return {
        "table_schema": "public",
        "table_name": name,
        "info_table_type": "VIEW",
        "relkind": "v",
        "table_comment": None,
        "approx_row_count": None,  # views always NULL in pg_stat_user_tables
    }


def _table_row(name: str = "orders", count: int | None = 100) -> dict[str, Any]:
    return {
        "table_schema": "public",
        "table_name": name,
        "info_table_type": "BASE TABLE",
        "relkind": "r",
        "table_comment": None,
        "approx_row_count": count,
    }


def test_view_count_succeeds_sets_view_count_source() -> None:
    engine, captured = _make_get_tables_engine(
        [_view_row()], view_count=42
    )
    conn = PostgresConnector(_config())
    conn._engine = engine
    tables = conn.get_tables()
    assert len(tables) == 1
    assert tables[0].row_count == 42
    assert tables[0].row_count_source == "view_count"
    assert tables[0].table_type == "view"
    assert any(f"SET LOCAL statement_timeout = {VIEW_COUNT_TIMEOUT_MS}" in s for s in captured)


def test_view_count_timeout_returns_minus_one() -> None:
    timeout_exc = OperationalError(
        "stmt", {}, Exception("canceling statement due to statement timeout")
    )
    engine, _ = _make_get_tables_engine(
        [_view_row()], view_count_error=timeout_exc
    )
    conn = PostgresConnector(_config())
    conn._engine = engine
    tables = conn.get_tables()
    assert len(tables) == 1
    assert tables[0].row_count == -1
    assert tables[0].row_count_source == "view_timeout"


def test_base_table_with_stat_count_keeps_stat_source() -> None:
    engine, _ = _make_get_tables_engine([_table_row(count=1234)])
    conn = PostgresConnector(_config())
    conn._engine = engine
    tables = conn.get_tables()
    assert tables[0].row_count == 1234
    assert tables[0].row_count_source == "stat"


def test_view_count_quotes_identifier_with_quotes() -> None:
    engine, captured = _make_get_tables_engine(
        [_view_row(name='odd"name')], view_count=5
    )
    conn = PostgresConnector(_config())
    conn._engine = engine
    tables = conn.get_tables()
    assert tables[0].row_count == 5
    # Embedded quote must be doubled in the issued SQL.
    assert any('"odd""name"' in s for s in captured)
