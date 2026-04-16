"""Tests for catalog enrichment fields on ColumnInfo, Column, and inspect merge."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from pretensor.connectors.base import ColumnInfo, ColumnStats, TableInfo
from pretensor.connectors.inspect import inspect
from pretensor.connectors.models import Column
from pretensor.introspection.models.config import ConnectionConfig, DatabaseType


def test_column_info_new_fields_default_none() -> None:
    col = ColumnInfo(name="x", data_type="int")
    assert col.column_cardinality is None
    assert col.index_type is None
    assert col.index_is_unique is None


def test_column_info_new_fields_set() -> None:
    col = ColumnInfo(
        name="id",
        data_type="int",
        column_cardinality=500,
        index_type="btree",
        index_is_unique=True,
    )
    assert col.column_cardinality == 500
    assert col.index_type == "btree"
    assert col.index_is_unique is True


def test_column_model_new_fields_default_none() -> None:
    col = Column(name="x", data_type="varchar")
    assert col.column_cardinality is None
    assert col.index_type is None
    assert col.index_is_unique is None


def test_column_model_new_fields_set() -> None:
    col = Column(
        name="category",
        data_type="varchar",
        column_cardinality=12,
        index_type="gin",
        index_is_unique=False,
    )
    assert col.column_cardinality == 12
    assert col.index_type == "gin"
    assert col.index_is_unique is False


def test_column_cardinality_negative_fraction_semantics() -> None:
    """Negative n_distinct values from pg_stats encode fraction semantics."""
    col = Column(name="id", data_type="int", column_cardinality=-1)
    assert col.column_cardinality == -1


def _make_config() -> ConnectionConfig:
    return ConnectionConfig(
        name="test",
        type=DatabaseType.POSTGRES,
        host="localhost",
        port=5432,
        user="u",
        password="p",
        database="db",
    )


def _mock_connector(
    columns: list[ColumnInfo],
    column_extra: dict[tuple[str, str, str], dict[str, Any]],
) -> MagicMock:
    connector = MagicMock()
    connector.__enter__ = MagicMock(return_value=connector)
    connector.__exit__ = MagicMock(return_value=False)
    connector.get_tables.return_value = [
        TableInfo(name="items", schema_name="public", row_count=0)
    ]
    connector.get_columns.return_value = columns
    connector.get_foreign_keys.return_value = []
    connector.get_table_row_count.return_value = 0
    connector.load_deep_catalog.return_value = ({}, column_extra)
    return connector


def test_inspect_merges_catalog_enrichment_into_column() -> None:
    """column_catalog_extra from load_deep_catalog is merged into Column objects."""
    col_infos = [ColumnInfo(name="id", data_type="int")]
    col_extra: dict[tuple[str, str, str], dict[str, Any]] = {
        ("public", "items", "id"): {
            "column_cardinality": 250,
            "index_type": "btree",
            "index_is_unique": True,
        }
    }
    mock_conn = _mock_connector(col_infos, col_extra)

    with patch("pretensor.connectors.inspect.get_connector", return_value=mock_conn):
        snap = inspect(_make_config())

    col = snap.tables[0].columns[0]
    assert col.column_cardinality == 250
    assert col.index_type == "btree"
    assert col.index_is_unique is True


def test_inspect_column_without_catalog_enrichment_has_none_fields() -> None:
    """Columns not present in column_catalog_extra have None for enrichment fields."""
    col_infos = [ColumnInfo(name="description", data_type="text")]
    mock_conn = _mock_connector(col_infos, {})

    with patch("pretensor.connectors.inspect.get_connector", return_value=mock_conn):
        snap = inspect(_make_config())

    col = snap.tables[0].columns[0]
    assert col.column_cardinality is None
    assert col.index_type is None
    assert col.index_is_unique is None


def _mock_connector_with_rows(
    columns: list[ColumnInfo],
    row_count: int,
) -> MagicMock:
    """Like _mock_connector but with a non-zero row_count so stats collection runs."""
    connector = MagicMock()
    connector.__enter__ = MagicMock(return_value=connector)
    connector.__exit__ = MagicMock(return_value=False)
    connector.get_tables.return_value = [
        TableInfo(name="items", schema_name="public", row_count=row_count)
    ]
    connector.get_columns.return_value = columns
    connector.get_foreign_keys.return_value = []
    connector.get_table_row_count.return_value = row_count
    connector.load_deep_catalog.return_value = ({}, {})
    connector.get_column_stats.return_value = ColumnStats(
        distinct_count=2,
        min_value="0",
        max_value="1",
        null_percentage=0.0,
        sample_distinct_values=["0", "1"],
    )
    return connector


def test_inspect_skips_stats_for_unsupported_column_types() -> None:
    """Stats collection is pre-filtered for boolean/tsvector/bytea/etc."""
    col_infos = [
        ColumnInfo(name="activebool", data_type="boolean"),
        ColumnInfo(name="fulltext", data_type="tsvector"),
        ColumnInfo(name="picture", data_type="bytea"),
    ]
    mock_conn = _mock_connector_with_rows(col_infos, row_count=100)

    with patch("pretensor.connectors.inspect.get_connector", return_value=mock_conn):
        inspect(_make_config())

    assert mock_conn.get_column_stats.call_count == 0


def test_inspect_collects_stats_for_supported_column_types() -> None:
    """Supported types (integer, text) still trigger get_column_stats."""
    col_infos = [
        ColumnInfo(name="id", data_type="integer"),
        ColumnInfo(name="title", data_type="text"),
    ]
    mock_conn = _mock_connector_with_rows(col_infos, row_count=100)

    with patch("pretensor.connectors.inspect.get_connector", return_value=mock_conn):
        inspect(_make_config())

    assert mock_conn.get_column_stats.call_count == 2


def test_inspect_stats_filter_is_case_insensitive() -> None:
    """Type matching is case-insensitive (e.g. 'BOOLEAN' is also skipped)."""
    col_infos = [
        ColumnInfo(name="flag", data_type="BOOLEAN"),
        ColumnInfo(name="payload", data_type="JSON"),
    ]
    mock_conn = _mock_connector_with_rows(col_infos, row_count=100)

    with patch("pretensor.connectors.inspect.get_connector", return_value=mock_conn):
        inspect(_make_config())

    assert mock_conn.get_column_stats.call_count == 0
