"""Tests for BaseConnector optional hooks and shared models."""

from __future__ import annotations

from typing import Any

from pretensor.connectors.base import (
    BaseConnector,
    ColumnInfo,
    ColumnStats,
    ForeignKeyInfo,
    TableGrant,
    TableInfo,
)
from pretensor.introspection.models.config import (
    ConnectionConfig,
    DatabaseType,
    SchemaFilter,
)


class _MinimalConnector(BaseConnector):
    """Concrete connector for exercising BaseConnector defaults."""

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def get_tables(
        self, schema_filter: SchemaFilter | None = None
    ) -> list[TableInfo]:
        _ = schema_filter
        return []

    def get_columns(self, table_name: str, schema_name: str) -> list[ColumnInfo]:
        _ = table_name, schema_name
        return []

    def get_foreign_keys(self) -> list[ForeignKeyInfo]:
        return []

    def get_table_row_count(self, table_name: str, schema_name: str) -> int:
        _ = table_name, schema_name
        return 0

    def get_column_stats(
        self, table_name: str, column_name: str, schema_name: str
    ) -> ColumnStats:
        _ = table_name, column_name, schema_name
        return ColumnStats()

    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        _ = sql
        return []


def test_table_grant_fields() -> None:
    tg = TableGrant(
        grantee="reader",
        schema_name="public",
        table_name="orders",
    )
    assert tg.grantee == "reader"
    assert tg.schema_name == "public"
    assert tg.table_name == "orders"


def test_get_table_grants_default_empty() -> None:
    cfg = ConnectionConfig(name="t", type=DatabaseType.POSTGRES)
    conn = _MinimalConnector(cfg)
    assert conn.get_table_grants() == []
    assert conn.get_table_grants(SchemaFilter(include=["public"])) == []
