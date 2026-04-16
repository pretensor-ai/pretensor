"""Heuristic metric templates from classified fact tables.

Templates are generated deterministically as ``SELECT SUM(col) AS total FROM schema.table``
using ANSI double-quote identifier quoting.  LLM-based refinement (COUNT, AVG, multi-column
aggregations) is deferred to a follow-up task.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from pretensor.core.ids import metric_template_node_id
from pretensor.core.store import KuzuStore
from pretensor.validation.query_validator import QueryValidator

__all__ = ["MetricTemplateBuilder"]

logger = logging.getLogger(__name__)

_METRIC_NAME_PARTS = frozenset(
    {
        "amount",
        "total",
        "price",
        "cost",
        "revenue",
        "qty",
        "quantity",
        "count",
        "fee",
        "fees",
        "tax",
        "payment",
        "payments",
        "subtotal",
        "discount",
    }
)


_DIALECT_POSTGRESQL = "postgresql"


def _pg_ident(part: str) -> str:
    """ANSI / PostgreSQL double-quote identifier escaping."""
    return '"' + part.replace('"', '""') + '"'


def _is_numeric_type(data_type: str) -> bool:
    t = data_type.lower()
    return any(
        x in t
        for x in (
            "int",
            "numeric",
            "decimal",
            "float",
            "double",
            "real",
            "money",
            "smallserial",
            "serial",
            "bigserial",
        )
    )


def _column_signals_metric(column_name: str) -> bool:
    base = column_name.lower().strip()
    if not base:
        return False
    for part in _METRIC_NAME_PARTS:
        if part in base:
            return True
    return False


def _slugify_metric_name(schema: str, table: str, column: str) -> str:
    raw = f"sum_{schema}_{table}_{column}"
    s = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return s or "metric"


class MetricTemplateBuilder:
    """Create validated ``MetricTemplate`` nodes from fact tables and metric-like columns."""

    def __init__(self, store: KuzuStore) -> None:
        self._store = store

    def build(self, database_key: str) -> int:
        """Clear prior templates for this DB, then write new validated templates. Returns count."""
        rows_cn = self._store.query_all_rows(
            """
            MATCH (t:SchemaTable {database: $db})
            RETURN t.connection_name
            LIMIT 1
            """,
            {"db": database_key},
        )
        if not rows_cn:
            return 0
        connection_name = str(rows_cn[0][0])

        self._store.execute_write(
            """
            MATCH (m:MetricTemplate {connection_name: $cn, database: $db})
            DETACH DELETE m
            """,
            {"cn": connection_name, "db": database_key},
        )

        candidates = self._store.query_all_rows(
            """
            MATCH (t:SchemaTable {connection_name: $cn, database: $db, role: 'fact'})
                  -[:HAS_COLUMN]->(c:SchemaColumn)
            RETURN t.node_id, t.schema_name, t.table_name, c.column_name, c.data_type
            ORDER BY t.schema_name, t.table_name, c.ordinal_position, c.column_name
            """,
            {"cn": connection_name, "db": database_key},
        )

        written = 0
        now = datetime.now(timezone.utc).isoformat()
        validator = QueryValidator(
            self._store,
            connection_name=connection_name,
            database_key=database_key,
        )

        for tid, sn, tn, cname, dtype in candidates:
            if not _is_numeric_type(str(dtype or "")):
                continue
            if not _column_signals_metric(str(cname)):
                continue
            schema_s = str(sn)
            table_s = str(tn)
            col_s = str(cname)
            qschema = _pg_ident(schema_s)
            qtable = _pg_ident(table_s)
            qcol = _pg_ident(col_s)
            sql = f"SELECT SUM({qcol}) AS total FROM {qschema}.{qtable}"
            result = validator.validate(sql)
            name = _slugify_metric_name(schema_s, table_s, col_s)
            node_id = metric_template_node_id(connection_name, database_key, name)
            display = f"Sum of {col_s} ({schema_s}.{table_s})"
            desc = (
                f"Aggregate total of column {col_s} on fact table {schema_s}.{table_s}."
            )
            tables_used = [f"{schema_s}.{table_s}"]
            err_list: list[str] = []
            if not result.valid:
                err_list = (
                    result.syntax_errors
                    + [f"missing_table:{t}" for t in result.missing_tables]
                    + [f"missing_column:{c}" for c in result.missing_columns]
                    + [f"join:{j.message}" for j in result.invalid_joins]
                )
            self._store.upsert_metric_template(
                node_id=node_id,
                connection_name=connection_name,
                database=database_key,
                dialect=_DIALECT_POSTGRESQL,
                name=name,
                display_name=display,
                description=desc,
                sql_template=sql,
                tables_used=tables_used,
                validated=result.valid,
                validation_errors=err_list,
                generated_at_iso=now,
                stale=False,
                depends_on_table_node_ids=[str(tid)],
            )
            written += 1
            if not result.valid:
                logger.debug(
                    "Metric template %s not validated: %s", name, err_list[:3]
                )

        return written

    @staticmethod
    def mark_stale_for_database(store: KuzuStore, database_key: str) -> None:
        """Set ``stale`` on templates when the indexed graph may have drifted (e.g. reindex)."""
        store.execute_write(
            """
            MATCH (m:MetricTemplate {database: $db})
            SET m.stale = true
            """,
            {"db": database_key},
        )
