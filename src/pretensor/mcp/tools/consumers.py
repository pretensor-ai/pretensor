"""MCP ``consumers`` tool: external code consumers of a table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pretensor.mcp.tool_registry import McpTool
from pretensor.visibility.filter import VisibilityFilter

from ..service_context import get_effective_visibility_filter
from ..service_registry import (
    graph_path_for_entry,
    load_registry,
    open_store_for_entry,
    release_store,
    resolve_registry_entry,
)
from .context import resolve_table_node_id

__all__ = ["create_tool", "consumers_payload"]


def consumers_payload(
    graph_dir: Path,
    *,
    table: str,
    database: str,
    op: str | None = None,
    min_confidence: float = 0.6,
    visibility_filter: VisibilityFilter | None = None,
) -> dict[str, Any]:
    """External code locations that read or write ``table``, with provenance."""
    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, database)
    if entry is None:
        return {"error": "Unknown database connection or name; pass `database`."}
    gp = graph_path_for_entry(entry)
    if not gp.exists():
        return {"error": f"Graph file missing: {gp}"}

    db_key = str(entry.database)
    vf = visibility_filter or get_effective_visibility_filter()
    store = open_store_for_entry(entry)
    try:
        start_id, err = resolve_table_node_id(
            store, table, db_key, visibility_filter=vf
        )
        if err is not None:
            try:
                return json.loads(err)
            except json.JSONDecodeError:
                return {"error": err}
        if start_id is None:
            return {
                "error": "Could not resolve table; check the table name and database."
            }

        rows = store.query_all_rows(
            """
            MATCH (c:ExternalConsumer)-[r:CONSUMES]->(t:SchemaTable {node_id: $nid})
            WHERE r.confidence >= $minc
              AND ($op IS NULL OR r.op = $op)
            RETURN c.service_name, c.file_path, c.line_start, c.line_end, c.kind,
                   c.language, r.op, r.confidence, c.sql_fingerprint
            ORDER BY r.confidence DESC, c.service_name, c.file_path
            """,
            {"nid": start_id, "minc": min_confidence, "op": op},
        )

        consumers: list[dict[str, Any]] = []
        reads = writes = 0
        for service, file_path, ls, le, kind, lang, edge_op, conf, fp in rows:
            consumers.append(
                {
                    "service": service,
                    "file": file_path,
                    "line_start": ls,
                    "line_end": le,
                    "kind": kind,
                    "language": lang,
                    "op": edge_op,
                    "confidence": conf,
                    "sql_fingerprint": fp,
                }
            )
            if edge_op == "write":
                writes += 1
            else:
                reads += 1

        return {
            "table": table,
            "database": database,
            "consumers": consumers,
            "counts": {"read": reads, "write": writes, "total": len(consumers)},
        }
    finally:
        release_store(store)


def create_tool(graph_dir: Path) -> McpTool:
    from ._timed import timed_tool

    async def _handle(args: dict) -> dict:
        tbl = str(args.get("table", "")).strip()
        db_t = str(args.get("database", "")).strip()
        if not tbl:
            return {"error": "Missing `table`"}
        if not db_t:
            return {"error": "Missing `database`"}
        op = args.get("op")
        op_s = str(op).strip() if op is not None else None
        if op_s == "":
            op_s = None
        min_conf = float(args.get("min_confidence", 0.6))
        with timed_tool("consumers", graph_dir, table=tbl, database=db_t, op=op_s):
            return consumers_payload(
                graph_dir,
                table=tbl,
                database=db_t,
                op=op_s,
                min_confidence=min_conf,
            )

    return McpTool(
        name="consumers",
        description=(
            "External code locations (services) that read or write a table, from "
            "`pretensor analyze`. Each entry carries service, file, line range, op, "
            "kind, and confidence."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table name or schema.table",
                },
                "database": {
                    "type": "string",
                    "description": "Connection name or logical database",
                },
                "op": {
                    "type": ["string", "null"],
                    "enum": ["read", "write", None],
                    "description": "Filter to 'read' or 'write' consumers only",
                },
                "min_confidence": {
                    "type": "number",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["table", "database"],
            "additionalProperties": False,
        },
        handler=_handle,
    )
