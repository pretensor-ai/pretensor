"""Tests for the ``collect_stats`` switch on :func:`inspect`.

Per-column data stats cost at least one full-table-scan query per column, so
structure-only callers (drift detection) must be able to skip them.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pretensor.connectors.base import ColumnInfo, ColumnStats, TableInfo
from pretensor.connectors.inspect import inspect
from pretensor.introspection.models.config import ConnectionConfig, DatabaseType


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


def _mock_connector() -> MagicMock:
    connector = MagicMock()
    connector.__enter__ = MagicMock(return_value=connector)
    connector.__exit__ = MagicMock(return_value=False)
    connector.get_tables.return_value = [
        TableInfo(name="items", schema_name="public", row_count=100)
    ]
    connector.get_columns.return_value = [
        ColumnInfo(name="id", data_type="int"),
        ColumnInfo(name="label", data_type="text"),
    ]
    connector.get_foreign_keys.return_value = []
    connector.load_deep_catalog.return_value = ({}, {})
    connector.get_column_stats.return_value = ColumnStats(
        distinct_count=10,
        min_value="1",
        max_value="100",
        null_percentage=0.0,
    )
    return connector


def test_inspect_collects_stats_by_default() -> None:
    mock_conn = _mock_connector()
    with patch("pretensor.connectors.inspect.get_connector", return_value=mock_conn):
        snap = inspect(_make_config())
    assert mock_conn.get_column_stats.call_count == 2
    assert snap.tables[0].columns[0].distinct_count == 10


def test_inspect_collect_stats_false_skips_column_stats() -> None:
    mock_conn = _mock_connector()
    with patch("pretensor.connectors.inspect.get_connector", return_value=mock_conn):
        snap = inspect(_make_config(), collect_stats=False)
    mock_conn.get_column_stats.assert_not_called()
    # Structure is still fully introspected.
    tbl = snap.tables[0]
    assert [c.name for c in tbl.columns] == ["id", "label"]
    assert tbl.row_count == 100
    assert tbl.columns[0].distinct_count is None
