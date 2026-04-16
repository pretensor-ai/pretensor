"""MCP ``validate_sql`` tool payload — sqlglot + graph-backed SQL validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pretensor.validation.query_validator import QueryValidator

from ..service_registry import (
    load_registry,
    open_store_for_entry,
    resolve_registry_entry,
)

__all__ = ["validate_sql_payload"]


_DEFAULT_DIALECT = "postgres"


def validate_sql_payload(
    graph_dir: Path,
    *,
    sql: str,
    database: str,
    dialect: str = _DEFAULT_DIALECT,
) -> dict[str, Any]:
    """Build JSON-serializable payload for ``validate_sql``.

    Args:
        graph_dir: Root directory of the Pretensor graph store.
        sql: SQL statement to validate.
        database: ``connection_name`` or logical ``database`` key.
        dialect: sqlglot dialect (default ``postgres``).
    """
    if not sql.strip():
        return {"error": "Missing or empty `sql`"}
    if not database.strip():
        return {"error": "Missing or empty `database`"}

    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, database)
    if entry is None:
        return {"error": f"No registry entry matches database {database!r}"}

    store = open_store_for_entry(entry)
    try:
        validator = QueryValidator(
            store,
            connection_name=entry.connection_name,
            database_key=entry.database,
            dialect=dialect,
        )
        result = validator.validate(sql)
    finally:
        store.close()

    return {
        "valid": result.valid,
        "dialect": dialect,
        "syntax_errors": result.syntax_errors,
        "missing_tables": result.missing_tables,
        "missing_columns": result.missing_columns,
        "invalid_joins": [
            {
                "message": j.message,
                "left_table": j.left_table,
                "right_table": j.right_table,
            }
            for j in result.invalid_joins
        ],
        "suggestions": result.suggestions,
    }
