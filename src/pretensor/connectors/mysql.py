"""MySQL connector.

All MySQL-specific imports and SQL live here. Nothing outside this
package should reference pymysql or sqlalchemy.dialects.mysql.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from pretensor.connectors.base import (
    BaseConnector,
    ColumnInfo,
    ColumnStats,
    ForeignKeyInfo,
    TableInfo,
)
from pretensor.connectors.lineage_sqlglot import table_refs_from_sql
from pretensor.connectors.models import ViewDependency
from pretensor.errors import ConnectorError
from pretensor.introspection.models.config import ConnectionConfig, SchemaFilter

logger = logging.getLogger(__name__)

LOW_CARDINALITY_THRESHOLD = 50

# MySQL system databases excluded from all INFORMATION_SCHEMA queries.
_SYSTEM_SCHEMAS = frozenset(
    {"information_schema", "performance_schema", "mysql", "sys"}
)


def _map_mysql_table_type(table_type: str) -> str:
    if table_type == "VIEW":
        return "view"
    return "table"


def _quote_ident(identifier: str) -> str:
    """Backtick-quote a MySQL identifier; escape embedded backticks."""
    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"


class MySQLConnector(BaseConnector):
    """MySQL implementation of the database connector interface."""

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._engine: Engine | None = None

    def _build_url(self) -> str:
        cfg = self.config
        user = cfg.user or "root"
        password = cfg.password or ""
        host = cfg.host or "localhost"
        port = cfg.port or 3306
        database = cfg.database or ""
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

    def connect(self) -> None:
        url = self._build_url()
        try:
            self._engine = create_engine(
                url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 10},
            )
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(
                "Connected to %s:%s/%s",
                self.config.host,
                self.config.port,
                self.config.database,
            )
        except Exception as exc:
            raise ConnectorError(
                f"Failed to connect to MySQL at {self.config.host}:{self.config.port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("Disconnected from %s", self.config.database)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise ConnectorError("Not connected. Call connect() first.")
        return self._engine

    def _effective_schema_filter(self, sf: SchemaFilter) -> SchemaFilter:
        """When no include list is given, default to the connected database."""
        if not sf.include and self.config.database:
            return SchemaFilter(
                include=[self.config.database], exclude=list(sf.exclude)
            )
        return sf

    def _schema_visible(self, schema_name: str, sf: SchemaFilter) -> bool:
        if schema_name in _SYSTEM_SCHEMAS:
            return False
        if sf.include and schema_name not in sf.include:
            return False
        if sf.exclude and schema_name in sf.exclude:
            return False
        return True

    def get_tables(self, schema_filter: SchemaFilter | None = None) -> list[TableInfo]:
        sf = self._effective_schema_filter(schema_filter or self.config.schema_filter)

        query = text("""\
            SELECT
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_TYPE,
                TABLE_COMMENT,
                TABLE_ROWS
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA NOT IN ('information_schema', 'performance_schema', 'mysql', 'sys')
              AND TABLE_TYPE IN ('BASE TABLE', 'VIEW')
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()

        tables: list[TableInfo] = []
        for row in rows:
            schema_name = str(row["TABLE_SCHEMA"])
            if not self._schema_visible(schema_name, sf):
                continue

            table_type = _map_mysql_table_type(str(row["TABLE_TYPE"] or "BASE TABLE"))
            approx = row["TABLE_ROWS"]
            comment_raw = row["TABLE_COMMENT"]
            comment = str(comment_raw) if comment_raw else None

            if table_type == "view":
                row_count, row_count_source = self._count_view_rows(
                    schema_name, str(row["TABLE_NAME"])
                )
            elif approx is not None:
                row_count: int | None = int(approx)
                row_count_source: str | None = "stat"
            else:
                row_count = None
                row_count_source = None

            tables.append(
                TableInfo(
                    name=str(row["TABLE_NAME"]),
                    schema_name=schema_name,
                    row_count=row_count,
                    row_count_source=row_count_source,
                    comment=comment,
                    table_type=table_type,
                )
            )
        return tables

    def _count_view_rows(self, schema_name: str, view_name: str) -> tuple[int, str]:
        """Run SELECT COUNT(*) on a view; return (-1, 'view_timeout') on any error."""
        qualified = f"{_quote_ident(schema_name)}.{_quote_ident(view_name)}"
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) AS n FROM {qualified}"))
                row = result.mappings().first()
                if row is None or row["n"] is None:
                    return -1, "view_timeout"
                return int(row["n"]), "view_count"
        except SQLAlchemyError as exc:
            logger.debug(
                "View row-count failed for %s.%s: %s", schema_name, view_name, exc
            )
            return -1, "view_timeout"

    def get_columns(self, table_name: str, schema_name: str) -> list[ColumnInfo]:
        query = text("""\
            SELECT
                COLUMN_NAME,
                ORDINAL_POSITION,
                DATA_TYPE,
                COLUMN_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                COLUMN_COMMENT,
                COLUMN_KEY,
                EXTRA
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema_name
              AND TABLE_NAME = :table_name
            ORDER BY ORDINAL_POSITION
        """)

        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    query,
                    {"schema_name": schema_name, "table_name": table_name},
                )
                .mappings()
                .all()
            )

        columns: list[ColumnInfo] = []
        for row in rows:
            col_name = str(row["COLUMN_NAME"])
            data_type = str(row["DATA_TYPE"])
            col_key = str(row["COLUMN_KEY"] or "")
            is_pk = col_key == "PRI"
            is_indexed = col_key in ("PRI", "MUL", "UNI")
            comment_raw = row["COLUMN_COMMENT"]
            comment = str(comment_raw) if comment_raw else None
            ord_pos = row["ORDINAL_POSITION"]
            columns.append(
                ColumnInfo(
                    name=col_name,
                    data_type=data_type,
                    nullable=str(row["IS_NULLABLE"]) == "YES",
                    is_primary_key=is_pk,
                    default_value=row["COLUMN_DEFAULT"],
                    comment=comment,
                    is_indexed=is_indexed,
                    check_constraints=[],
                    ordinal_position=int(ord_pos) if ord_pos is not None else None,
                )
            )
        return columns

    def get_foreign_keys(self) -> list[ForeignKeyInfo]:
        """All FK relationships via INFORMATION_SCHEMA.KEY_COLUMN_USAGE."""
        sf = self._effective_schema_filter(self.config.schema_filter)

        query = text("""\
            SELECT
                kcu.CONSTRAINT_NAME,
                kcu.TABLE_SCHEMA      AS source_schema,
                kcu.TABLE_NAME        AS source_table,
                kcu.COLUMN_NAME       AS source_column,
                kcu.REFERENCED_TABLE_SCHEMA  AS target_schema,
                kcu.REFERENCED_TABLE_NAME    AS target_table,
                kcu.REFERENCED_COLUMN_NAME   AS target_column
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
            WHERE kcu.REFERENCED_TABLE_NAME IS NOT NULL
              AND kcu.TABLE_SCHEMA NOT IN (
                  'information_schema', 'performance_schema', 'mysql', 'sys'
              )
            ORDER BY kcu.TABLE_SCHEMA, kcu.TABLE_NAME,
                     kcu.CONSTRAINT_NAME, kcu.ORDINAL_POSITION
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()

        fks: list[ForeignKeyInfo] = []
        for row in rows:
            source_schema = str(row["source_schema"])
            if not self._schema_visible(source_schema, sf):
                continue
            fks.append(
                ForeignKeyInfo(
                    constraint_name=str(row["CONSTRAINT_NAME"]),
                    source_schema=source_schema,
                    source_table=str(row["source_table"]),
                    source_column=str(row["source_column"]),
                    target_schema=str(row["target_schema"]),
                    target_table=str(row["target_table"]),
                    target_column=str(row["target_column"]),
                )
            )
        return fks

    def get_table_row_count(self, table_name: str, schema_name: str) -> int:
        query = text("""\
            SELECT TABLE_ROWS AS approx_count
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = :schema_name
              AND TABLE_NAME = :table_name
        """)

        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    query,
                    {"schema_name": schema_name, "table_name": table_name},
                )
                .mappings()
                .first()
            )

        if row is None or row["approx_count"] is None:
            return 0
        return int(row["approx_count"])

    def get_column_stats(
        self, table_name: str, column_name: str, schema_name: str
    ) -> ColumnStats:
        fqn = f"{_quote_ident(schema_name)}.{_quote_ident(table_name)}"
        col = _quote_ident(column_name)

        stats_query = text(f"""\
            SELECT
                COUNT(DISTINCT {col})  AS distinct_count,
                MIN({col})             AS min_val,
                MAX({col})             AS max_val,
                ROUND(
                    100.0 * SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END)
                    / GREATEST(COUNT(*), 1),
                    2
                )                      AS null_pct
            FROM {fqn}
        """)

        with self.engine.connect() as conn:
            row = conn.execute(stats_query).mappings().first()

        if row is None:
            return ColumnStats()

        distinct_count = int(row["distinct_count"])
        sample_values: list[str] | None = None

        if distinct_count <= LOW_CARDINALITY_THRESHOLD:
            sample_query = text(
                f"SELECT DISTINCT CAST({col} AS CHAR) AS val"
                f" FROM {fqn}"
                f" WHERE {col} IS NOT NULL"
                f" ORDER BY val"
                f" LIMIT {LOW_CARDINALITY_THRESHOLD}"
            )
            with self.engine.connect() as conn:
                sample_rows = conn.execute(sample_query).mappings().all()
            sample_values = [str(r["val"]) for r in sample_rows if r["val"] is not None]

        return ColumnStats(
            distinct_count=distinct_count,
            min_value=str(row["min_val"]) if row["min_val"] is not None else None,
            max_value=str(row["max_val"]) if row["max_val"] is not None else None,
            null_percentage=float(row["null_pct"])
            if row["null_pct"] is not None
            else 0.0,
            sample_distinct_values=sample_values,
        )

    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        return [dict(row) for row in rows]

    def load_view_dependencies(
        self, schema_filter: SchemaFilter
    ) -> list[ViewDependency]:
        """Lineage from MySQL views via INFORMATION_SCHEMA.VIEWS."""
        sf = self._effective_schema_filter(schema_filter)
        views_sql = text("""\
            SELECT TABLE_SCHEMA, TABLE_NAME, VIEW_DEFINITION
            FROM INFORMATION_SCHEMA.VIEWS
            WHERE TABLE_SCHEMA NOT IN (
                'information_schema', 'performance_schema', 'mysql', 'sys'
            )
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(views_sql).mappings().all()

        deps: list[ViewDependency] = []
        for row in rows:
            view_schema = str(row["TABLE_SCHEMA"])
            if not self._schema_visible(view_schema, sf):
                continue
            view_name = str(row["TABLE_NAME"])
            definition = row["VIEW_DEFINITION"]
            if definition is None:
                continue
            sql_text = str(definition)
            for src_schema, src_table in table_refs_from_sql(
                sql_text, dialect="mysql", default_schema=view_schema
            ):
                if not self._schema_visible(src_schema, sf):
                    continue
                if src_schema == view_schema and src_table == view_name:
                    continue
                deps.append(
                    ViewDependency(
                        source_schema=src_schema,
                        source_table=src_table,
                        target_schema=view_schema,
                        target_table=view_name,
                        lineage_type="VIEW",
                        object_name=f"{view_schema}.{view_name}",
                        confidence=1.0,
                    )
                )
        return deps
