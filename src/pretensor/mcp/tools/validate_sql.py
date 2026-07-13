"""MCP ``validate_sql`` tool payload — sqlglot + graph-backed SQL validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pretensor.mcp.tool_registry import McpTool
from pretensor.validation.query_validator import QueryValidator

from ..service_registry import (
    load_registry,
    open_store_for_entry,
    release_store,
    resolve_registry_entry,
)

__all__ = ["create_tool", "validate_sql_payload"]

logger = logging.getLogger(__name__)

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

    try:
        store = open_store_for_entry(entry)
    except Exception:
        # Keep the on-disk graph path and engine error server-side; the client
        # sees a stable, connection-named message only.
        logger.exception(
            "validate_sql: failed to open graph for connection %r",
            entry.connection_name,
        )
        return {
            "error": (
                f"Could not open the graph for database {database!r}. "
                "The graph may be missing or corrupt — re-run indexing."
            )
        }
    try:
        validator = QueryValidator(
            store,
            connection_name=entry.connection_name,
            database_key=entry.database,
            dialect=dialect,
        )
        result = validator.validate(sql)
    finally:
        release_store(store)

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


def create_tool(graph_dir: Path) -> McpTool:
    from ._timed import timed_tool

    async def _handle(args: dict) -> dict:
        sql_s = str(args.get("sql", ""))
        db_t = str(args.get("database", "")).strip()
        dialect_raw = args.get("dialect")
        dialect_s = str(dialect_raw).strip() if dialect_raw is not None else "postgres"
        if not dialect_s:
            dialect_s = "postgres"
        if not sql_s.strip():
            return {"error": "Missing `sql`"}
        if not db_t:
            return {"error": "Missing `database`"}
        with timed_tool("validate_sql", graph_dir, database=db_t, dialect=dialect_s):
            return validate_sql_payload(
                graph_dir, sql=sql_s, database=db_t, dialect=dialect_s
            )

    return McpTool(
        name="validate_sql",
        description=(
            "Validate a SQL statement against the indexed graph before executing it. "
            "Catches: unknown tables, unknown columns, invalid joins (pairs not in "
            "FK_REFERENCES or INFERRED_JOIN edges), sqlglot parse errors. Returns "
            "fuzzy suggestions for misspelled identifiers. Read-only.\n\n"
            "Examples:\n"
            "  sql='SELECT * FROM custmer' -> missing_tables=['custmer'], "
            "suggestions=['customer']\n"
            "  sql='SELECT c.foo FROM customer c' -> "
            "missing_columns=[{'table':'customer','column':'foo'}]\n"
            "  sql='SELECT * FROM orders o JOIN customer c ON o.x = c.y' -> "
            "invalid_joins=[...] when no FK or inferred edge exists between columns"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL statement to validate",
                },
                "database": {
                    "type": "string",
                    "description": "Connection name or logical database",
                },
                "dialect": {
                    "type": "string",
                    "default": "postgres",
                    "description": "sqlglot dialect (default 'postgres')",
                },
            },
            "required": ["sql", "database"],
            "additionalProperties": False,
        },
        handler=_handle,
    )
