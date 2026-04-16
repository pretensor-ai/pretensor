"""Unit tests for SnowflakeConnector using TPC-H sample schema as fixture data.

Snowflake's SNOWFLAKE_SAMPLE_DATA.TPCH_SF1 is used as the model because it is
publicly accessible on every Snowflake account and has realistic FKs, check
constraints, views, and row counts — making it a good ground-truth fixture.

No live Snowflake connection is required: SQLAlchemy engine calls are mocked
so these run fully offline.

TPC-H schema used:
  REGION   (R_REGIONKEY PK, R_NAME, R_COMMENT)
  NATION   (N_NATIONKEY PK, N_NAME, N_REGIONKEY FK→REGION, N_COMMENT)
  SUPPLIER (S_SUPPKEY PK, S_NAME, S_ADDRESS, S_NATIONKEY FK→NATION, ...)
  CUSTOMER (C_CUSTKEY PK, C_NAME, C_ADDRESS, C_NATIONKEY FK→NATION, ...)
  PART     (P_PARTKEY PK, P_NAME, P_MFGR, P_BRAND, P_TYPE, ...)
  PARTSUPP (PS_PARTKEY PK+FK→PART, PS_SUPPKEY PK+FK→SUPPLIER, ...)
  ORDERS   (O_ORDERKEY PK, O_CUSTKEY FK→CUSTOMER, O_ORDERSTATUS CHECK IN ('F','O','P'), ...)
  LINEITEM (L_ORDERKEY PK+FK→ORDERS, L_LINENUMBER PK, L_RETURNFLAG CHECK IN ('A','N','R'), ...)
  REVENUE0 (view — typical TPC-H Q15 helper view)
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from pretensor.connectors.snowflake import SnowflakeConnector, SnowflakeConnectorError
from pretensor.introspection.models.config import ConnectionConfig, DatabaseType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DSN = "snowflake://bob:pw@xy12345.us-east-1.aws/SNOWFLAKE_SAMPLE_DATA/TPCH_SF1?warehouse=COMPUTE_WH"


def _cfg(schema: str | None = "TPCH_SF1") -> ConnectionConfig:
    from pretensor.introspection.models.dsn import connection_config_from_url

    return connection_config_from_url(_DSN, "tpch")


def _make_connector(schema: str | None = "TPCH_SF1") -> SnowflakeConnector:
    """Return a connector with a mock engine already injected (no real connect)."""
    connector = SnowflakeConnector(_cfg(schema))
    connector._engine = _mock_engine()  # type: ignore[assignment]
    return connector


def _mock_engine() -> MagicMock:
    """Return a MagicMock that behaves like a SQLAlchemy engine context manager."""
    engine = MagicMock()
    conn = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    return engine


def _mock_conn(connector: SnowflakeConnector) -> MagicMock:
    """Return the mock connection object for a connector built by _make_connector."""
    return cast(
        MagicMock, connector._engine
    ).connect.return_value.__enter__.return_value


def _mapping_rows(rows: list[dict[str, Any]]) -> MagicMock:
    """Return a mock execute result whose .mappings().all() yields dicts."""
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows
    return result


def _tuple_rows(rows: list[tuple]) -> MagicMock:
    """Return a mock execute result whose .all() yields tuples."""
    result = MagicMock()
    result.all.return_value = rows
    return result


# ---------------------------------------------------------------------------
# TPC-H fixture data
# ---------------------------------------------------------------------------

TPCH_TABLES_ROWS = [
    {
        "table_schema": "TPCH_SF1",
        "table_name": "CUSTOMER",
        "table_type": "BASE TABLE",
        "comment": "Customer master data",
        "row_count": 150000,
    },
    {
        "table_schema": "TPCH_SF1",
        "table_name": "LINEITEM",
        "table_type": "BASE TABLE",
        "comment": None,
        "row_count": 6001215,
    },
    {
        "table_schema": "TPCH_SF1",
        "table_name": "NATION",
        "table_type": "BASE TABLE",
        "comment": "Nations reference table",
        "row_count": 25,
    },
    {
        "table_schema": "TPCH_SF1",
        "table_name": "ORDERS",
        "table_type": "BASE TABLE",
        "comment": None,
        "row_count": 1500000,
    },
    {
        "table_schema": "TPCH_SF1",
        "table_name": "PART",
        "table_type": "BASE TABLE",
        "comment": None,
        "row_count": 200000,
    },
    {
        "table_schema": "TPCH_SF1",
        "table_name": "PARTSUPP",
        "table_type": "BASE TABLE",
        "comment": None,
        "row_count": 800000,
    },
    {
        "table_schema": "TPCH_SF1",
        "table_name": "REGION",
        "table_type": "BASE TABLE",
        "comment": "Geographic regions",
        "row_count": 5,
    },
    {
        "table_schema": "TPCH_SF1",
        "table_name": "SUPPLIER",
        "table_type": "BASE TABLE",
        "comment": None,
        "row_count": 10000,
    },
    {
        "table_schema": "TPCH_SF1",
        "table_name": "REVENUE0",
        "table_type": "VIEW",
        "comment": "Q15 revenue helper view",
        "row_count": None,
    },
]

ORDERS_COLUMNS_ROWS = [
    {
        "column_name": "O_ORDERKEY",
        "ordinal_position": 1,
        "data_type": "NUMBER",
        "is_nullable": "NO",
        "column_default": None,
        "comment": "Order primary key",
    },
    {
        "column_name": "O_CUSTKEY",
        "ordinal_position": 2,
        "data_type": "NUMBER",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "O_ORDERSTATUS",
        "ordinal_position": 3,
        "data_type": "TEXT",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "O_TOTALPRICE",
        "ordinal_position": 4,
        "data_type": "NUMBER",
        "is_nullable": "YES",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "O_ORDERDATE",
        "ordinal_position": 5,
        "data_type": "DATE",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "O_ORDERPRIORITY",
        "ordinal_position": 6,
        "data_type": "TEXT",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "O_CLERK",
        "ordinal_position": 7,
        "data_type": "TEXT",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "O_SHIPPRIORITY",
        "ordinal_position": 8,
        "data_type": "NUMBER",
        "is_nullable": "NO",
        "column_default": "0",
        "comment": None,
    },
    {
        "column_name": "O_COMMENT",
        "ordinal_position": 9,
        "data_type": "TEXT",
        "is_nullable": "YES",
        "column_default": None,
        "comment": None,
    },
]

LINEITEM_COLUMNS_ROWS = [
    {
        "column_name": "L_ORDERKEY",
        "ordinal_position": 1,
        "data_type": "NUMBER",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_PARTKEY",
        "ordinal_position": 2,
        "data_type": "NUMBER",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_SUPPKEY",
        "ordinal_position": 3,
        "data_type": "NUMBER",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_LINENUMBER",
        "ordinal_position": 4,
        "data_type": "NUMBER",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_QUANTITY",
        "ordinal_position": 5,
        "data_type": "NUMBER",
        "is_nullable": "YES",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_EXTENDEDPRICE",
        "ordinal_position": 6,
        "data_type": "NUMBER",
        "is_nullable": "YES",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_DISCOUNT",
        "ordinal_position": 7,
        "data_type": "NUMBER",
        "is_nullable": "YES",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_TAX",
        "ordinal_position": 8,
        "data_type": "NUMBER",
        "is_nullable": "YES",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_RETURNFLAG",
        "ordinal_position": 9,
        "data_type": "TEXT",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_LINESTATUS",
        "ordinal_position": 10,
        "data_type": "TEXT",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_SHIPDATE",
        "ordinal_position": 11,
        "data_type": "DATE",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_COMMITDATE",
        "ordinal_position": 12,
        "data_type": "DATE",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_RECEIPTDATE",
        "ordinal_position": 13,
        "data_type": "DATE",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_SHIPINSTRUCT",
        "ordinal_position": 14,
        "data_type": "TEXT",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_SHIPMODE",
        "ordinal_position": 15,
        "data_type": "TEXT",
        "is_nullable": "NO",
        "column_default": None,
        "comment": None,
    },
    {
        "column_name": "L_COMMENT",
        "ordinal_position": 16,
        "data_type": "TEXT",
        "is_nullable": "YES",
        "column_default": None,
        "comment": None,
    },
]

# SHOW IMPORTED KEYS IN DATABASE result columns (by index):
# 0=created_on, 1=pk_database_name, 2=pk_schema_name, 3=pk_table_name,
# 4=pk_column_name, 5=fk_database_name, 6=fk_schema_name, 7=fk_table_name,
# 8=fk_column_name, 9=key_sequence, 10=update_rule, 11=delete_rule, 12=fk_name
_DB = "SNOWFLAKE_SAMPLE_DATA"
_TS = "2024-01-01"  # created_on placeholder

TPCH_FOREIGN_KEYS_TUPLES: list[tuple] = [
    # FK_CUSTOMER_NATION
    (_TS, _DB, "TPCH_SF1", "NATION", "N_NATIONKEY", _DB, "TPCH_SF1", "CUSTOMER", "C_NATIONKEY", 1, "NO ACTION", "NO ACTION", "FK_CUSTOMER_NATION"),
    # FK_LINEITEM_ORDERS
    (_TS, _DB, "TPCH_SF1", "ORDERS", "O_ORDERKEY", _DB, "TPCH_SF1", "LINEITEM", "L_ORDERKEY", 1, "NO ACTION", "NO ACTION", "FK_LINEITEM_ORDERS"),
    # FK_LINEITEM_PART
    (_TS, _DB, "TPCH_SF1", "PART", "P_PARTKEY", _DB, "TPCH_SF1", "LINEITEM", "L_PARTKEY", 1, "NO ACTION", "NO ACTION", "FK_LINEITEM_PART"),
    # FK_LINEITEM_SUPPLIER
    (_TS, _DB, "TPCH_SF1", "SUPPLIER", "S_SUPPKEY", _DB, "TPCH_SF1", "LINEITEM", "L_SUPPKEY", 1, "NO ACTION", "NO ACTION", "FK_LINEITEM_SUPPLIER"),
    # FK_NATION_REGION
    (_TS, _DB, "TPCH_SF1", "REGION", "R_REGIONKEY", _DB, "TPCH_SF1", "NATION", "N_REGIONKEY", 1, "NO ACTION", "NO ACTION", "FK_NATION_REGION"),
    # FK_ORDERS_CUSTOMER
    (_TS, _DB, "TPCH_SF1", "CUSTOMER", "C_CUSTKEY", _DB, "TPCH_SF1", "ORDERS", "O_CUSTKEY", 1, "NO ACTION", "NO ACTION", "FK_ORDERS_CUSTOMER"),
    # FK_PARTSUPP_PART
    (_TS, _DB, "TPCH_SF1", "PART", "P_PARTKEY", _DB, "TPCH_SF1", "PARTSUPP", "PS_PARTKEY", 1, "NO ACTION", "NO ACTION", "FK_PARTSUPP_PART"),
    # FK_PARTSUPP_SUPPLIER
    (_TS, _DB, "TPCH_SF1", "SUPPLIER", "S_SUPPKEY", _DB, "TPCH_SF1", "PARTSUPP", "PS_SUPPKEY", 1, "NO ACTION", "NO ACTION", "FK_PARTSUPP_SUPPLIER"),
    # FK_SUPPLIER_NATION
    (_TS, _DB, "TPCH_SF1", "NATION", "N_NATIONKEY", _DB, "TPCH_SF1", "SUPPLIER", "S_NATIONKEY", 1, "NO ACTION", "NO ACTION", "FK_SUPPLIER_NATION"),
]


# ---------------------------------------------------------------------------
# Tests: get_tables
# ---------------------------------------------------------------------------


class TestGetTables:
    def test_returns_all_base_tables_and_views(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.execute.return_value = _mapping_rows(TPCH_TABLES_ROWS)

        tables = connector.get_tables()

        assert len(tables) == 9
        names = {t.name for t in tables}
        assert "ORDERS" in names
        assert "LINEITEM" in names
        assert "REVENUE0" in names

    def test_base_table_type(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.execute.return_value = _mapping_rows(TPCH_TABLES_ROWS)

        tables = connector.get_tables()
        by_name = {t.name: t for t in tables}

        assert by_name["ORDERS"].table_type == "table"
        assert by_name["CUSTOMER"].table_type == "table"

    def test_view_type(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.execute.return_value = _mapping_rows(TPCH_TABLES_ROWS)

        tables = connector.get_tables()
        by_name = {t.name: t for t in tables}

        assert by_name["REVENUE0"].table_type == "view"

    def test_row_counts_populated(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.execute.return_value = _mapping_rows(TPCH_TABLES_ROWS)

        tables = connector.get_tables()
        by_name = {t.name: t for t in tables}

        assert by_name["LINEITEM"].row_count == 6001215
        assert by_name["REGION"].row_count == 5
        assert by_name["REVENUE0"].row_count is None  # view

    def test_comment_populated(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.execute.return_value = _mapping_rows(TPCH_TABLES_ROWS)

        tables = connector.get_tables()
        by_name = {t.name: t for t in tables}

        assert by_name["CUSTOMER"].comment == "Customer master data"
        assert by_name["REGION"].comment == "Geographic regions"
        assert by_name["ORDERS"].comment is None

    def test_schema_filter_include(self) -> None:
        """Only tables in schema_filter.include should be returned."""
        connector = _make_connector()
        conn = _mock_conn(connector)
        # Mix two schemas in the result
        rows = TPCH_TABLES_ROWS + [
            {
                "table_schema": "OTHER_SCHEMA",
                "table_name": "SHOULD_BE_EXCLUDED",
                "table_type": "BASE TABLE",
                "comment": None,
                "row_count": 0,
            }
        ]
        conn.execute.return_value = _mapping_rows(rows)

        from pretensor.introspection.models.config import SchemaFilter

        tables = connector.get_tables(schema_filter=SchemaFilter(include=["TPCH_SF1"]))

        names = {t.name for t in tables}
        assert "SHOULD_BE_EXCLUDED" not in names
        assert "ORDERS" in names

    def test_schema_filter_exclude(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        rows = TPCH_TABLES_ROWS + [
            {
                "table_schema": "STAGING",
                "table_name": "RAW_EVENTS",
                "table_type": "BASE TABLE",
                "comment": None,
                "row_count": 0,
            }
        ]
        conn.execute.return_value = _mapping_rows(rows)

        from pretensor.introspection.models.config import SchemaFilter

        tables = connector.get_tables(schema_filter=SchemaFilter(exclude=["STAGING"]))

        names = {t.name for t in tables}
        assert "RAW_EVENTS" not in names

    def test_table_type_materialized_view(self) -> None:
        # Schema must match schema_filter.include ("TPCH_SF1") set by _cfg()
        rows = [
            {
                "table_schema": "TPCH_SF1",
                "table_name": "MV",
                "table_type": "MATERIALIZED VIEW",
                "comment": None,
                "row_count": 100,
            }
        ]
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.execute.return_value = _mapping_rows(rows)

        tables = connector.get_tables()
        assert tables[0].table_type == "materialized_view"

    def test_table_type_external(self) -> None:
        rows = [
            {
                "table_schema": "TPCH_SF1",
                "table_name": "EXT",
                "table_type": "EXTERNAL TABLE",
                "comment": None,
                "row_count": None,
            }
        ]
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.execute.return_value = _mapping_rows(rows)

        tables = connector.get_tables()
        assert tables[0].table_type == "foreign_table"


# ---------------------------------------------------------------------------
# Tests: get_columns
# ---------------------------------------------------------------------------


def _setup_get_columns(
    connector: SnowflakeConnector,
    *,
    pk_cols: list[str],
    column_rows: list[dict[str, Any]],
    table_name: str = "ORDERS",
    schema_name: str = "TPCH_SF1",
) -> None:
    """Wire the mock engine so that _primary_key_columns (via SHOW PRIMARY KEYS)
    and the main column query each return the right fixture data.

    PK data comes from ``exec_driver_sql`` (SHOW command, cached per-schema).
    Check constraints always return ``{}`` (no mock needed).
    The main COLUMNS query uses ``execute``.
    """
    conn = _mock_conn(connector)
    # SHOW PRIMARY KEYS IN SCHEMA returns tuples:
    # (created_on, db, schema, table, column, key_seq, constraint_name)
    pk_tuples = [
        ("2024-01-01", "SNOWFLAKE_SAMPLE_DATA", schema_name, table_name, col, i + 1, "PK")
        for i, col in enumerate(pk_cols)
    ]
    conn.exec_driver_sql.return_value = _tuple_rows(pk_tuples)
    conn.execute.return_value = _mapping_rows(column_rows)


class TestGetColumns:
    def test_orders_column_count(self) -> None:
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["O_ORDERKEY"],
            column_rows=ORDERS_COLUMNS_ROWS,
        )

        cols = connector.get_columns("ORDERS", "TPCH_SF1")
        assert len(cols) == 9

    def test_primary_key_detected(self) -> None:
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["O_ORDERKEY"],
            column_rows=ORDERS_COLUMNS_ROWS,
        )

        cols = connector.get_columns("ORDERS", "TPCH_SF1")
        by_name = {c.name: c for c in cols}

        assert by_name["O_ORDERKEY"].is_primary_key is True
        assert by_name["O_CUSTKEY"].is_primary_key is False

    def test_nullable_correctly_parsed(self) -> None:
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["O_ORDERKEY"],
            column_rows=ORDERS_COLUMNS_ROWS,
        )

        cols = connector.get_columns("ORDERS", "TPCH_SF1")
        by_name = {c.name: c for c in cols}

        assert by_name["O_ORDERKEY"].nullable is False
        assert by_name["O_TOTALPRICE"].nullable is True
        assert by_name["O_COMMENT"].nullable is True

    def test_ordinal_position_order_preserved(self) -> None:
        """Columns must come back in schema order (ordinal_position), not alphabetically."""
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["O_ORDERKEY"],
            column_rows=ORDERS_COLUMNS_ROWS,
        )

        cols = connector.get_columns("ORDERS", "TPCH_SF1")
        names = [c.name for c in cols]

        assert names[0] == "O_ORDERKEY"
        assert names[1] == "O_CUSTKEY"
        assert names[-1] == "O_COMMENT"
        assert names == sorted(
            names,
            key=lambda n: next(
                r["ordinal_position"]
                for r in ORDERS_COLUMNS_ROWS
                if r["column_name"] == n
            ),
        )

    def test_check_constraints_always_empty(self) -> None:
        """Snowflake has no SHOW equivalent for CHECK constraints; always empty."""
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["O_ORDERKEY"],
            column_rows=ORDERS_COLUMNS_ROWS,
        )

        cols = connector.get_columns("ORDERS", "TPCH_SF1")
        assert all(c.check_constraints == [] for c in cols)

    def test_composite_pk_lineitem(self) -> None:
        """LINEITEM has a composite PK (L_ORDERKEY, L_LINENUMBER)."""
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["L_ORDERKEY", "L_LINENUMBER"],
            column_rows=LINEITEM_COLUMNS_ROWS,
            table_name="LINEITEM",
        )

        cols = connector.get_columns("LINEITEM", "TPCH_SF1")
        by_name = {c.name: c for c in cols}

        assert by_name["L_ORDERKEY"].is_primary_key is True
        assert by_name["L_LINENUMBER"].is_primary_key is True
        assert by_name["L_PARTKEY"].is_primary_key is False

    def test_default_value_preserved(self) -> None:
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["O_ORDERKEY"],
            column_rows=ORDERS_COLUMNS_ROWS,
        )

        cols = connector.get_columns("ORDERS", "TPCH_SF1")
        by_name = {c.name: c for c in cols}

        assert by_name["O_SHIPPRIORITY"].default_value == "0"
        assert by_name["O_ORDERKEY"].default_value is None

    def test_column_comment_preserved(self) -> None:
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["O_ORDERKEY"],
            column_rows=ORDERS_COLUMNS_ROWS,
        )

        cols = connector.get_columns("ORDERS", "TPCH_SF1")
        by_name = {c.name: c for c in cols}

        assert by_name["O_ORDERKEY"].comment == "Order primary key"
        assert by_name["O_CUSTKEY"].comment is None

    def test_is_indexed_always_false_for_snowflake(self) -> None:
        """Snowflake has no secondary indexes; is_indexed must always be False."""
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["O_ORDERKEY"],
            column_rows=ORDERS_COLUMNS_ROWS,
        )

        cols = connector.get_columns("ORDERS", "TPCH_SF1")
        assert all(c.is_indexed is False for c in cols)

    def test_no_check_constraints_returns_empty_lists(self) -> None:
        connector = _make_connector()
        _setup_get_columns(
            connector,
            pk_cols=["R_REGIONKEY"],
            table_name="REGION",
            column_rows=[
                {
                    "column_name": "R_REGIONKEY",
                    "ordinal_position": 1,
                    "data_type": "NUMBER",
                    "is_nullable": "NO",
                    "column_default": None,
                    "comment": None,
                },
                {
                    "column_name": "R_NAME",
                    "ordinal_position": 2,
                    "data_type": "TEXT",
                    "is_nullable": "NO",
                    "column_default": None,
                    "comment": None,
                },
                {
                    "column_name": "R_COMMENT",
                    "ordinal_position": 3,
                    "data_type": "TEXT",
                    "is_nullable": "YES",
                    "column_default": None,
                    "comment": None,
                },
            ],
        )

        cols = connector.get_columns("REGION", "TPCH_SF1")
        assert all(c.check_constraints == [] for c in cols)


# ---------------------------------------------------------------------------
# Tests: get_foreign_keys
# ---------------------------------------------------------------------------


class TestGetForeignKeys:
    def test_all_tpch_fks_returned(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.exec_driver_sql.return_value = _tuple_rows(TPCH_FOREIGN_KEYS_TUPLES)

        fks = connector.get_foreign_keys()
        assert len(fks) == 9

    def test_fk_fields(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.exec_driver_sql.return_value = _tuple_rows(TPCH_FOREIGN_KEYS_TUPLES)

        fks = connector.get_foreign_keys()
        orders_fk = next(f for f in fks if f.source_table == "ORDERS")

        assert orders_fk.source_column == "O_CUSTKEY"
        assert orders_fk.target_table == "CUSTOMER"
        assert orders_fk.target_column == "C_CUSTKEY"
        assert orders_fk.constraint_name == "FK_ORDERS_CUSTOMER"

    def test_lineitem_has_three_fks(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.exec_driver_sql.return_value = _tuple_rows(TPCH_FOREIGN_KEYS_TUPLES)

        fks = connector.get_foreign_keys()
        lineitem_fks = [f for f in fks if f.source_table == "LINEITEM"]
        assert len(lineitem_fks) == 3
        targets = {f.target_table for f in lineitem_fks}
        assert targets == {"ORDERS", "PART", "SUPPLIER"}

    def test_no_foreign_keys(self) -> None:
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.exec_driver_sql.return_value = _tuple_rows([])

        fks = connector.get_foreign_keys()
        assert fks == []

    def test_composite_fk_ordered_by_key_sequence(self) -> None:
        """Composite FK columns are sorted by key_sequence within a constraint."""
        connector = _make_connector()
        conn = _mock_conn(connector)
        # Simulate a composite FK: LINEITEM(L_PARTKEY, L_SUPPKEY) → PARTSUPP(PS_PARTKEY, PS_SUPPKEY)
        # Deliberately return key_sequence=2 before key_sequence=1 to test sorting.
        composite_rows = [
            (_TS, _DB, "TPCH_SF1", "PARTSUPP", "PS_SUPPKEY", _DB, "TPCH_SF1", "LINEITEM", "L_SUPPKEY", 2, "NO ACTION", "NO ACTION", "FK_LINEITEM_PARTSUPP"),
            (_TS, _DB, "TPCH_SF1", "PARTSUPP", "PS_PARTKEY", _DB, "TPCH_SF1", "LINEITEM", "L_PARTKEY", 1, "NO ACTION", "NO ACTION", "FK_LINEITEM_PARTSUPP"),
        ]
        conn.exec_driver_sql.return_value = _tuple_rows(composite_rows)

        fks = connector.get_foreign_keys()
        assert len(fks) == 2
        assert fks[0].constraint_name == "FK_LINEITEM_PARTSUPP"
        assert fks[0].source_column == "L_PARTKEY"   # key_sequence=1 first
        assert fks[0].target_column == "PS_PARTKEY"
        assert fks[1].source_column == "L_SUPPKEY"   # key_sequence=2 second
        assert fks[1].target_column == "PS_SUPPKEY"

    def test_show_imported_keys_failure_falls_back_to_information_schema(self) -> None:
        """SHOW failure should trigger INFORMATION_SCHEMA FK fallback."""
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.exec_driver_sql.side_effect = Exception("Access denied")
        conn.execute.return_value = _mapping_rows(
            [
                {
                    "constraint_name": "FK_ORDERS_CUSTOMER",
                    "source_schema": "TPCH_SF1",
                    "source_table": "ORDERS",
                    "source_column": "O_CUSTKEY",
                    "target_schema": "TPCH_SF1",
                    "target_table": "CUSTOMER",
                    "target_column": "C_CUSTKEY",
                    "source_ordinal_position": 1,
                }
            ]
        )

        fks = connector.get_foreign_keys()
        assert len(fks) == 1
        assert fks[0].constraint_name == "FK_ORDERS_CUSTOMER"
        assert fks[0].source_table == "ORDERS"
        assert fks[0].target_table == "CUSTOMER"

    def test_show_imported_keys_empty_falls_back_to_information_schema(self) -> None:
        """SHOW returning no rows should still trigger INFORMATION_SCHEMA fallback."""
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.exec_driver_sql.return_value = _tuple_rows([])
        conn.execute.return_value = _mapping_rows(
            [
                {
                    "constraint_name": "FK_LINEITEM_ORDERS",
                    "source_schema": "TPCH_SF1",
                    "source_table": "LINEITEM",
                    "source_column": "L_ORDERKEY",
                    "target_schema": "TPCH_SF1",
                    "target_table": "ORDERS",
                    "target_column": "O_ORDERKEY",
                    "source_ordinal_position": 1,
                }
            ]
        )

        fks = connector.get_foreign_keys()
        assert len(fks) == 1
        assert fks[0].constraint_name == "FK_LINEITEM_ORDERS"
        assert fks[0].source_column == "L_ORDERKEY"
        assert fks[0].target_column == "O_ORDERKEY"

    def test_information_schema_fallback_preserves_composite_fk_order(self) -> None:
        """Fallback rows are sorted by ordinal position for composite FKs."""
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.exec_driver_sql.return_value = _tuple_rows([])
        conn.execute.return_value = _mapping_rows(
            [
                {
                    "constraint_name": "FK_LINEITEM_PARTSUPP",
                    "source_schema": "TPCH_SF1",
                    "source_table": "LINEITEM",
                    "source_column": "L_SUPPKEY",
                    "target_schema": "TPCH_SF1",
                    "target_table": "PARTSUPP",
                    "target_column": "PS_SUPPKEY",
                    "source_ordinal_position": 2,
                },
                {
                    "constraint_name": "FK_LINEITEM_PARTSUPP",
                    "source_schema": "TPCH_SF1",
                    "source_table": "LINEITEM",
                    "source_column": "L_PARTKEY",
                    "target_schema": "TPCH_SF1",
                    "target_table": "PARTSUPP",
                    "target_column": "PS_PARTKEY",
                    "source_ordinal_position": 1,
                },
            ]
        )

        fks = connector.get_foreign_keys()
        assert len(fks) == 2
        assert fks[0].source_column == "L_PARTKEY"
        assert fks[0].target_column == "PS_PARTKEY"
        assert fks[1].source_column == "L_SUPPKEY"
        assert fks[1].target_column == "PS_SUPPKEY"

    def test_fk_fallback_failure_returns_empty(self) -> None:
        """If SHOW and INFORMATION_SCHEMA both fail, return empty list."""
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.exec_driver_sql.side_effect = Exception("Access denied")
        conn.execute.side_effect = SQLAlchemyError("No privilege on INFORMATION_SCHEMA")

        fks = connector.get_foreign_keys()
        assert fks == []


# ---------------------------------------------------------------------------
# Tests: URL building
# ---------------------------------------------------------------------------


class TestSnowflakeUrl:
    def test_url_with_warehouse_and_role(self) -> None:
        connector = SnowflakeConnector(_cfg())
        url = connector._snowflake_url()

        assert "xy12345.us-east-1.aws" in url
        assert "SNOWFLAKE_SAMPLE_DATA" in url
        assert "warehouse=COMPUTE_WH" in url

    def test_url_without_warehouse(self) -> None:
        from pretensor.introspection.models.dsn import connection_config_from_url

        cfg = connection_config_from_url("snowflake://u:p@acct.us-east-1.aws/MYDB", "c")
        connector = SnowflakeConnector(cfg)
        url = connector._snowflake_url()

        assert "warehouse" not in url
        assert "MYDB" in url

    def test_missing_account_raises(self) -> None:
        from pretensor.introspection.models.config import ConnectionConfig

        cfg = ConnectionConfig(
            name="x", type=DatabaseType.SNOWFLAKE, host="", database="DB"
        )
        connector = SnowflakeConnector(cfg)
        with pytest.raises(SnowflakeConnectorError, match="account"):
            connector._snowflake_url()

    def test_missing_database_raises(self) -> None:
        from pretensor.introspection.models.config import ConnectionConfig

        cfg = ConnectionConfig(
            name="x", type=DatabaseType.SNOWFLAKE, host="acct", database=""
        )
        connector = SnowflakeConnector(cfg)
        with pytest.raises(SnowflakeConnectorError, match="database"):
            connector._snowflake_url()

    def test_schema_included_in_path(self) -> None:
        from pretensor.introspection.models.dsn import connection_config_from_url

        cfg = connection_config_from_url("snowflake://u:p@acct/DB/MY_SCHEMA", "c")
        connector = SnowflakeConnector(cfg)
        url = connector._snowflake_url()

        assert "/DB/MY_SCHEMA" in url


# ---------------------------------------------------------------------------
# Tests: engine not connected guard
# ---------------------------------------------------------------------------


class TestConnectionGuard:
    def test_engine_not_set_raises(self) -> None:
        connector = SnowflakeConnector(_cfg())
        # _engine is None by default (no connect() called)
        with pytest.raises(SnowflakeConnectorError, match="Not connected"):
            _ = connector.engine

    def test_context_manager_calls_connect_and_disconnect(self) -> None:
        connector = SnowflakeConnector(_cfg())
        # Replace connect/disconnect with no-ops that track calls
        connected: list[bool] = []
        connector.connect = lambda: connected.append(True)  # type: ignore[method-assign]
        connector.disconnect = lambda: connected.append(False)  # type: ignore[method-assign]

        with connector:
            pass

        assert connected == [True, False]


# ---------------------------------------------------------------------------
# Tests: qualified_information_schema
# ---------------------------------------------------------------------------


class TestQualifiedInformationSchema:
    def test_standard(self) -> None:
        connector = _make_connector()
        assert (
            connector._qualified_information_schema()
            == '"SNOWFLAKE_SAMPLE_DATA".INFORMATION_SCHEMA'
        )

    def test_database_with_double_quotes_escaped(self) -> None:
        from pretensor.introspection.models.config import ConnectionConfig

        cfg = ConnectionConfig(
            name="x", type=DatabaseType.SNOWFLAKE, host="acct", database='DB"NAME'
        )
        connector = SnowflakeConnector(cfg)
        ischema = connector._qualified_information_schema()
        assert '""' in ischema  # double-quote escaped


# ---------------------------------------------------------------------------
# Tests: PK cache (SHOW PRIMARY KEYS)
# ---------------------------------------------------------------------------


class TestPrimaryKeyCache:
    def test_cache_populated_on_first_call(self) -> None:
        """First _primary_key_columns call triggers SHOW PRIMARY KEYS."""
        connector = _make_connector()
        conn = _mock_conn(connector)
        pk_rows = [
            ("2024-01-01", "SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1", "ORDERS", "O_ORDERKEY", 1, "PK_ORDERS"),
        ]
        conn.exec_driver_sql.return_value = _tuple_rows(pk_rows)

        assert connector._pk_cache is None
        result = connector._primary_key_columns("TPCH_SF1", "ORDERS")

        assert result == {"O_ORDERKEY"}
        assert connector._pk_cache is not None
        conn.exec_driver_sql.assert_called_once()

    def test_cache_hit_no_requery(self) -> None:
        """Second call uses cache — no additional query."""
        connector = _make_connector()
        conn = _mock_conn(connector)
        pk_rows = [
            ("2024-01-01", "SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1", "ORDERS", "O_ORDERKEY", 1, "PK_ORDERS"),
            ("2024-01-01", "SNOWFLAKE_SAMPLE_DATA", "TPCH_SF1", "NATION", "N_NATIONKEY", 1, "PK_NATION"),
        ]
        conn.exec_driver_sql.return_value = _tuple_rows(pk_rows)

        connector._primary_key_columns("TPCH_SF1", "ORDERS")
        result = connector._primary_key_columns("TPCH_SF1", "NATION")

        assert result == {"N_NATIONKEY"}
        conn.exec_driver_sql.assert_called_once()  # only one SHOW call

    def test_show_primary_keys_failure_returns_empty(self) -> None:
        """When SHOW PRIMARY KEYS fails, PK info is empty but no exception raised."""
        connector = _make_connector()
        conn = _mock_conn(connector)
        conn.exec_driver_sql.side_effect = Exception("Access denied")

        result = connector._primary_key_columns("TPCH_SF1", "ORDERS")
        assert result == set()
        assert connector._pk_cache == {}

    def test_check_constraints_always_empty(self) -> None:
        """Snowflake has no SHOW CHECK CONSTRAINTS; always returns empty."""
        connector = _make_connector()
        result = connector._load_check_constraints_for_table("TPCH_SF1", "ORDERS")
        assert result == {}
