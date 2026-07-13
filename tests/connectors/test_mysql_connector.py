"""Unit tests for MySQLConnector — all SQL is mocked via MagicMock."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pretensor.connectors.base import ColumnStats
from pretensor.introspection.models.config import (
    ConnectionConfig,
    DatabaseType,
    SchemaFilter,
)


def _make_config(
    database: str = "sakila",
    schema_filter: SchemaFilter | None = None,
) -> ConnectionConfig:
    return ConnectionConfig(
        name="test",
        type=DatabaseType.MYSQL,
        host="localhost",
        port=3306,
        database=database,
        user="root",
        password="test",
        schema_filter=schema_filter or SchemaFilter(include=[database]),
    )


def _rows(*dicts: dict[str, Any]) -> list[MagicMock]:
    """Return a list of mapping-like mocks from plain dicts."""
    out = []
    for d in dicts:
        m = MagicMock()
        m.__getitem__ = lambda self, k, _d=d: _d[k]
        m.get = lambda k, default=None, _d=d: _d.get(k, default)
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# connect / disconnect / engine property
# ---------------------------------------------------------------------------


class TestConnect:
    def test_connect_success(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pretensor.connectors.mysql.create_engine", return_value=mock_engine):
            connector.connect()

        assert connector._engine is mock_engine

    def test_connect_raises_connector_error_on_failure(self) -> None:
        from pretensor.connectors.mysql import ConnectorError, MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)

        with patch(
            "pretensor.connectors.mysql.create_engine",
            side_effect=Exception("connection refused"),
        ):
            with pytest.raises(ConnectorError, match="Failed to connect"):
                connector.connect()

    def test_engine_property_raises_when_not_connected(self) -> None:
        from pretensor.connectors.mysql import ConnectorError, MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        with pytest.raises(ConnectorError, match="Not connected"):
            _ = connector.engine

    def test_disconnect_disposes_engine(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        connector.disconnect()

        mock_engine.dispose.assert_called_once()
        assert connector._engine is None

    def test_context_manager(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)

        with (
            patch.object(connector, "connect") as mock_connect,
            patch.object(connector, "disconnect") as mock_disconnect,
        ):
            with connector:
                pass

        mock_connect.assert_called_once()
        mock_disconnect.assert_called_once()

    def test_build_url(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        url = connector._build_url()
        assert url.startswith("mysql+pymysql://")
        assert "localhost" in url
        assert "sakila" in url


# ---------------------------------------------------------------------------
# get_tables
# ---------------------------------------------------------------------------


class TestGetTables:
    def _setup_connector(self, cfg: ConnectionConfig) -> tuple[Any, MagicMock]:
        from pretensor.connectors.mysql import MySQLConnector

        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine
        return connector, mock_engine

    def _make_exec(self, mock_engine: MagicMock, table_rows: list) -> None:
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = table_rows
        mock_conn.execute.return_value = mock_result

    def test_basic_table_listing(self) -> None:
        cfg = _make_config("sakila", SchemaFilter(include=["sakila"]))
        connector, mock_engine = self._setup_connector(cfg)
        self._make_exec(
            mock_engine,
            _rows(
                {
                    "TABLE_SCHEMA": "sakila",
                    "TABLE_NAME": "actor",
                    "TABLE_TYPE": "BASE TABLE",
                    "TABLE_COMMENT": "Actor catalog",
                    "TABLE_ROWS": 200,
                },
                {
                    "TABLE_SCHEMA": "other_db",
                    "TABLE_NAME": "ignored",
                    "TABLE_TYPE": "BASE TABLE",
                    "TABLE_COMMENT": None,
                    "TABLE_ROWS": 10,
                },
            ),
        )

        tables = connector.get_tables()

        assert len(tables) == 1
        assert tables[0].name == "actor"
        assert tables[0].schema_name == "sakila"
        assert tables[0].comment == "Actor catalog"
        assert tables[0].row_count == 200
        assert tables[0].row_count_source == "stat"
        assert tables[0].table_type == "table"

    def test_view_type_mapped(self) -> None:
        cfg = _make_config("sakila", SchemaFilter(include=["sakila"]))
        connector, mock_engine = self._setup_connector(cfg)

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = _rows(
            {
                "TABLE_SCHEMA": "sakila",
                "TABLE_NAME": "actor_info",
                "TABLE_TYPE": "VIEW",
                "TABLE_COMMENT": "",
                "TABLE_ROWS": None,
            }
        )
        mock_conn.execute.return_value = mock_result

        with patch.object(connector, "_count_view_rows", return_value=(42, "view_count")):
            tables = connector.get_tables()

        assert len(tables) == 1
        assert tables[0].table_type == "view"
        assert tables[0].row_count == 42
        assert tables[0].row_count_source == "view_count"

    def test_null_table_rows_becomes_none(self) -> None:
        cfg = _make_config("sakila", SchemaFilter(include=["sakila"]))
        connector, mock_engine = self._setup_connector(cfg)
        self._make_exec(
            mock_engine,
            _rows(
                {
                    "TABLE_SCHEMA": "sakila",
                    "TABLE_NAME": "film",
                    "TABLE_TYPE": "BASE TABLE",
                    "TABLE_COMMENT": None,
                    "TABLE_ROWS": None,
                }
            ),
        )

        tables = connector.get_tables()

        assert tables[0].row_count is None
        assert tables[0].row_count_source is None

    def test_schema_exclude_filter(self) -> None:
        cfg = _make_config("sakila", SchemaFilter(exclude=["unwanted"]))
        connector, mock_engine = self._setup_connector(cfg)
        self._make_exec(
            mock_engine,
            _rows(
                {"TABLE_SCHEMA": "sakila", "TABLE_NAME": "actor", "TABLE_TYPE": "BASE TABLE", "TABLE_COMMENT": None, "TABLE_ROWS": 10},
                {"TABLE_SCHEMA": "unwanted", "TABLE_NAME": "junk", "TABLE_TYPE": "BASE TABLE", "TABLE_COMMENT": None, "TABLE_ROWS": 1},
            ),
        )

        tables = connector.get_tables()

        names = {t.name for t in tables}
        assert "actor" in names
        assert "junk" not in names


# ---------------------------------------------------------------------------
# _count_view_rows
# ---------------------------------------------------------------------------


class TestCountViewRows:
    def test_success(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        row_mock = MagicMock()
        row_mock.__getitem__ = lambda s, k: 99
        mock_conn.execute.return_value.mappings.return_value.first.return_value = row_mock

        count, source = connector._count_view_rows("sakila", "actor_info")

        assert count == 99
        assert source == "view_count"

    def test_sqlalchemy_error_returns_timeout(self) -> None:
        from sqlalchemy.exc import SQLAlchemyError

        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = SQLAlchemyError("timeout")

        count, source = connector._count_view_rows("sakila", "actor_info")

        assert count == -1
        assert source == "view_timeout"


# ---------------------------------------------------------------------------
# get_columns
# ---------------------------------------------------------------------------


class TestGetColumns:
    def _setup(self, rows_data: list[dict]) -> Any:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = _rows(*rows_data)
        mock_conn.execute.return_value = mock_result

        return connector

    def test_primary_key_detection(self) -> None:
        connector = self._setup(
            [
                {
                    "COLUMN_NAME": "actor_id",
                    "ORDINAL_POSITION": 1,
                    "DATA_TYPE": "smallint",
                    "COLUMN_TYPE": "smallint unsigned",
                    "IS_NULLABLE": "NO",
                    "COLUMN_DEFAULT": None,
                    "COLUMN_COMMENT": "",
                    "COLUMN_KEY": "PRI",
                    "EXTRA": "auto_increment",
                }
            ]
        )

        cols = connector.get_columns("actor", "sakila")

        assert len(cols) == 1
        assert cols[0].is_primary_key is True
        assert cols[0].is_indexed is True
        assert cols[0].name == "actor_id"

    def test_mul_key_sets_indexed(self) -> None:
        connector = self._setup(
            [
                {
                    "COLUMN_NAME": "film_id",
                    "ORDINAL_POSITION": 2,
                    "DATA_TYPE": "smallint",
                    "COLUMN_TYPE": "smallint unsigned",
                    "IS_NULLABLE": "NO",
                    "COLUMN_DEFAULT": None,
                    "COLUMN_COMMENT": "",
                    "COLUMN_KEY": "MUL",
                    "EXTRA": "",
                }
            ]
        )

        cols = connector.get_columns("film_actor", "sakila")

        assert cols[0].is_primary_key is False
        assert cols[0].is_indexed is True

    def test_uni_key_sets_indexed(self) -> None:
        connector = self._setup(
            [
                {
                    "COLUMN_NAME": "email",
                    "ORDINAL_POSITION": 4,
                    "DATA_TYPE": "varchar",
                    "COLUMN_TYPE": "varchar(50)",
                    "IS_NULLABLE": "YES",
                    "COLUMN_DEFAULT": None,
                    "COLUMN_COMMENT": "Customer email",
                    "COLUMN_KEY": "UNI",
                    "EXTRA": "",
                }
            ]
        )

        cols = connector.get_columns("customer", "sakila")

        assert cols[0].is_indexed is True
        assert cols[0].nullable is True
        assert cols[0].comment == "Customer email"

    def test_plain_column_not_indexed(self) -> None:
        connector = self._setup(
            [
                {
                    "COLUMN_NAME": "first_name",
                    "ORDINAL_POSITION": 2,
                    "DATA_TYPE": "varchar",
                    "COLUMN_TYPE": "varchar(45)",
                    "IS_NULLABLE": "NO",
                    "COLUMN_DEFAULT": None,
                    "COLUMN_COMMENT": "",
                    "COLUMN_KEY": "",
                    "EXTRA": "",
                }
            ]
        )

        cols = connector.get_columns("actor", "sakila")

        assert cols[0].is_indexed is False
        assert cols[0].is_primary_key is False

    def test_nullable_flag(self) -> None:
        connector = self._setup(
            [
                {
                    "COLUMN_NAME": "return_date",
                    "ORDINAL_POSITION": 3,
                    "DATA_TYPE": "datetime",
                    "COLUMN_TYPE": "datetime",
                    "IS_NULLABLE": "YES",
                    "COLUMN_DEFAULT": None,
                    "COLUMN_COMMENT": "",
                    "COLUMN_KEY": "",
                    "EXTRA": "",
                }
            ]
        )

        cols = connector.get_columns("rental", "sakila")

        assert cols[0].nullable is True

    def test_empty_comment_becomes_none(self) -> None:
        connector = self._setup(
            [
                {
                    "COLUMN_NAME": "title",
                    "ORDINAL_POSITION": 2,
                    "DATA_TYPE": "varchar",
                    "COLUMN_TYPE": "varchar(128)",
                    "IS_NULLABLE": "NO",
                    "COLUMN_DEFAULT": None,
                    "COLUMN_COMMENT": "",
                    "COLUMN_KEY": "",
                    "EXTRA": "",
                }
            ]
        )

        cols = connector.get_columns("film", "sakila")

        assert cols[0].comment is None


# ---------------------------------------------------------------------------
# get_foreign_keys
# ---------------------------------------------------------------------------


class TestGetForeignKeys:
    def _setup(self, rows_data: list[dict]) -> Any:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = _rows(*rows_data)
        mock_conn.execute.return_value = mock_result

        return connector

    def test_basic_fk(self) -> None:
        connector = self._setup(
            [
                {
                    "CONSTRAINT_NAME": "fk_film_actor_actor",
                    "source_schema": "sakila",
                    "source_table": "film_actor",
                    "source_column": "actor_id",
                    "target_schema": "sakila",
                    "target_table": "actor",
                    "target_column": "actor_id",
                }
            ]
        )

        fks = connector.get_foreign_keys()

        assert len(fks) == 1
        assert fks[0].constraint_name == "fk_film_actor_actor"
        assert fks[0].source_table == "film_actor"
        assert fks[0].source_column == "actor_id"
        assert fks[0].target_table == "actor"
        assert fks[0].target_column == "actor_id"

    def test_fk_from_excluded_schema_dropped(self) -> None:
        cfg = _make_config("sakila", SchemaFilter(include=["sakila"]))
        from pretensor.connectors.mysql import MySQLConnector

        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = _rows(
            {
                "CONSTRAINT_NAME": "fk_good",
                "source_schema": "sakila",
                "source_table": "film_actor",
                "source_column": "actor_id",
                "target_schema": "sakila",
                "target_table": "actor",
                "target_column": "actor_id",
            },
            {
                "CONSTRAINT_NAME": "fk_bad",
                "source_schema": "other_db",
                "source_table": "some_table",
                "source_column": "ref_id",
                "target_schema": "sakila",
                "target_table": "actor",
                "target_column": "actor_id",
            },
        )
        mock_conn.execute.return_value = mock_result

        fks = connector.get_foreign_keys()

        assert len(fks) == 1
        assert fks[0].constraint_name == "fk_good"

    def test_composite_fk_multiple_rows(self) -> None:
        connector = self._setup(
            [
                {
                    "CONSTRAINT_NAME": "fk_composite",
                    "source_schema": "sakila",
                    "source_table": "order_item",
                    "source_column": "order_id",
                    "target_schema": "sakila",
                    "target_table": "orders",
                    "target_column": "order_id",
                },
                {
                    "CONSTRAINT_NAME": "fk_composite",
                    "source_schema": "sakila",
                    "source_table": "order_item",
                    "source_column": "line_no",
                    "target_schema": "sakila",
                    "target_table": "orders",
                    "target_column": "line_no",
                },
            ]
        )

        fks = connector.get_foreign_keys()

        # Both columns returned as separate ForeignKeyInfo records
        assert len(fks) == 2
        col_names = {fk.source_column for fk in fks}
        assert col_names == {"order_id", "line_no"}


# ---------------------------------------------------------------------------
# get_table_row_count
# ---------------------------------------------------------------------------


class TestGetTableRowCount:
    def test_returns_table_rows(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        row = MagicMock()
        row.__getitem__ = lambda s, k: 500
        mock_conn.execute.return_value.mappings.return_value.first.return_value = row

        count = connector.get_table_row_count("actor", "sakila")

        assert count == 500

    def test_returns_zero_when_no_row(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.mappings.return_value.first.return_value = None

        count = connector.get_table_row_count("actor", "sakila")

        assert count == 0


# ---------------------------------------------------------------------------
# get_column_stats
# ---------------------------------------------------------------------------


class TestGetColumnStats:
    def _setup_connector_with_execute(self, *side_effects: Any) -> Any:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = list(side_effects)

        return connector

    def test_low_cardinality_produces_samples(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        stats_row = MagicMock()
        stats_row.__getitem__ = lambda s, k: {
            "distinct_count": 3,
            "min_val": "G",
            "max_val": "R",
            "null_pct": 0.0,
        }[k]

        sample_row1 = MagicMock()
        sample_row1.__getitem__ = lambda s, k: "G"
        sample_row2 = MagicMock()
        sample_row2.__getitem__ = lambda s, k: "PG"
        sample_row3 = MagicMock()
        sample_row3.__getitem__ = lambda s, k: "R"

        stats_result = MagicMock()
        stats_result.mappings.return_value.first.return_value = stats_row

        sample_result = MagicMock()
        sample_result.mappings.return_value.all.return_value = [
            sample_row1,
            sample_row2,
            sample_row3,
        ]

        mock_conn.execute.side_effect = [stats_result, sample_result]

        result = connector.get_column_stats("film", "rating", "sakila")

        assert result.distinct_count == 3
        assert result.min_value == "G"
        assert result.max_value == "R"
        assert result.null_percentage == 0.0
        assert result.sample_distinct_values == ["G", "PG", "R"]

    def test_high_cardinality_no_sample(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        stats_row = MagicMock()
        stats_row.__getitem__ = lambda s, k: {
            "distinct_count": 100,
            "min_val": "1",
            "max_val": "999",
            "null_pct": 5.0,
        }[k]

        stats_result = MagicMock()
        stats_result.mappings.return_value.first.return_value = stats_row

        mock_conn.execute.return_value = stats_result

        result = connector.get_column_stats("film", "film_id", "sakila")

        assert result.distinct_count == 100
        assert result.sample_distinct_values is None
        # Only one execute call (no sample query)
        assert mock_conn.execute.call_count == 1

    def test_no_row_returns_empty_stats(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.mappings.return_value.first.return_value = None

        result = connector.get_column_stats("film", "rating", "sakila")

        assert isinstance(result, ColumnStats)
        assert result.distinct_count is None


# ---------------------------------------------------------------------------
# load_view_dependencies
# ---------------------------------------------------------------------------


class TestLoadViewDependencies:
    def test_view_lineage_parsed(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        view_row = MagicMock()
        view_row.__getitem__ = lambda s, k: {
            "TABLE_SCHEMA": "sakila",
            "TABLE_NAME": "actor_info",
            "VIEW_DEFINITION": "SELECT a.actor_id FROM actor a JOIN film_actor fa ON a.actor_id = fa.actor_id",
        }[k]

        mock_conn.execute.return_value.mappings.return_value.all.return_value = [view_row]

        sf = SchemaFilter(include=["sakila"])
        deps = connector.load_view_dependencies(sf)

        source_tables = {(d.source_schema, d.source_table) for d in deps}
        assert ("sakila", "actor") in source_tables
        assert ("sakila", "film_actor") in source_tables
        for d in deps:
            assert d.target_table == "actor_info"
            assert d.lineage_type == "VIEW"

    def test_none_definition_skipped(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        view_row = MagicMock()
        view_row.__getitem__ = lambda s, k: {
            "TABLE_SCHEMA": "sakila",
            "TABLE_NAME": "broken_view",
            "VIEW_DEFINITION": None,
        }[k]

        mock_conn.execute.return_value.mappings.return_value.all.return_value = [view_row]

        deps = connector.load_view_dependencies(SchemaFilter(include=["sakila"]))

        assert deps == []


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


class TestExecuteQuery:
    def test_execute_query_returns_dicts(self) -> None:
        from pretensor.connectors.mysql import MySQLConnector

        cfg = _make_config()
        connector = MySQLConnector(cfg)
        mock_engine = MagicMock()
        connector._engine = mock_engine

        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        row_mock = MagicMock()
        row_mock.__iter__ = lambda s: iter(["k"])
        row_mock.items = lambda: [("col", "val")]
        mock_conn.execute.return_value.mappings.return_value.all.return_value = [row_mock]

        # We just verify it doesn't raise; exact dict shape depends on mock
        result = connector.execute_query("SELECT 1")
        assert isinstance(result, list)
