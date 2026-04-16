"""MCP server (stdio) exposing Pretensor graph tools and resources."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl

from pretensor.config import PretensorConfig
from pretensor.core.schema import format_catalog_summary
from pretensor.mcp.service import (
    clusters_resource_markdown,
    compile_metric_payload,
    context_payload,
    cross_db_entities_resource_markdown,
    cypher_payload,
    databases_resource_markdown,
    db_overview_resource_markdown,
    detect_changes_payload,
    impact_payload,
    list_databases_payload,
    mcp_config_json,
    metrics_resource_markdown,
    query_payload,
    schema_payload,
    traverse_payload,
    validate_sql_payload,
)
from pretensor.mcp.service_context import build_server_context, set_server_context
from pretensor.mcp.tool_registry import McpTool, McpToolRegistry
from pretensor.observability import log_timed_operation

__all__ = ["create_server", "run_server", "print_mcp_config"]

logger = logging.getLogger(__name__)

_DB_OVERVIEW_PATTERN = re.compile(
    r"^pretensor://db/(?P<name>[^/]+)/overview$",
)
_DB_CLUSTERS_PATTERN = re.compile(
    r"^pretensor://db/(?P<name>[^/]+)/clusters$",
)
_DB_METRICS_PATTERN = re.compile(
    r"^pretensor://db/(?P<name>[^/]+)/metrics$",
)

def print_mcp_config(graph_dir: Path, *, stream: Any = None) -> None:
    """Print the ``mcpServers`` JSON block (for Claude / Cursor) to a text stream.

    Defaults to **stderr** so stdout stays reserved for MCP JSON-RPC over stdio
    after ``serve`` starts the server.
    """
    out = stream if stream is not None else __import__("sys").stderr
    out.write(mcp_config_json(graph_dir) + "\n")
    out.flush()


def _build_oss_registry(graph_dir: Path) -> McpToolRegistry:
    """Build a :class:`McpToolRegistry` pre-loaded with all OSS tools.

    Internal helper — use :func:`create_server` as the extension point (it
    accepts ``extra_tools`` for downstream registration).
    """
    registry = McpToolRegistry()

    def _timed_tool(tool_name: str, **fields: Any):
        return log_timed_operation(
            logger,
            event="mcp.tool_handler",
            tool=tool_name,
            graph_dir=str(graph_dir),
            **fields,
        )

    async def _handle_list_databases(args: dict[str, Any]) -> dict[str, Any]:
        with _timed_tool("list_databases"):
            return list_databases_payload(graph_dir)

    async def _handle_cypher(args: dict[str, Any]) -> dict[str, Any]:
        q = str(args.get("query", "")).strip()
        db_t = str(args.get("database", "")).strip()
        timeout_raw = args.get("timeout_seconds", 5)
        try:
            timeout_s = float(timeout_raw)
        except (TypeError, ValueError):
            return {"error": "Invalid `timeout_seconds`"}
        with _timed_tool("cypher", database=db_t, timeout_seconds=timeout_s):
            return cypher_payload(
                graph_dir, query=q, database=db_t, timeout_seconds=timeout_s
            )

    async def _handle_query(args: dict[str, Any]) -> dict[str, Any]:
        q = str(args.get("q", "")).strip()
        if not q:
            return {"error": "Missing or empty `q`"}
        limit = int(args.get("limit", 10))
        db = args.get("db")
        db_s = str(db) if db is not None else None
        with _timed_tool("query", db=db_s, limit=limit):
            return query_payload(graph_dir, q=q, db=db_s, limit=limit)

    async def _handle_context(args: dict[str, Any]) -> dict[str, Any]:
        table = str(args.get("table", "")).strip()
        if not table:
            return {"error": "Missing `table`"}
        db = args.get("db")
        db_s = str(db) if db is not None else None
        detail = str(args.get("detail", "standard"))
        if detail not in ("summary", "standard", "full"):
            detail = "standard"
        with _timed_tool("context", table=table, db=db_s, detail=detail):
            return context_payload(
                graph_dir,
                table=table,
                db=db_s,
                detail=detail,  # type: ignore[arg-type]
            )

    async def _handle_traverse(args: dict[str, Any]) -> dict[str, Any]:
        from_t = str(args.get("from_table", "")).strip()
        to_t = str(args.get("to_table", "")).strip()
        db_t = str(args.get("database", "")).strip()
        if not from_t or not to_t:
            return {"error": "Missing `from_table` or `to_table`"}
        if not db_t:
            return {"error": "Missing `database`"}
        to_db = args.get("to_database")
        to_db_s = str(to_db).strip() if to_db is not None else None
        if to_db_s == "":
            to_db_s = None
        max_depth = int(args.get("max_depth", 4))
        top_k = max(1, min(10, int(args.get("top_k", 3))))
        max_inf = max(0, min(8, int(args.get("max_inferred_hops", 2))))
        raw_kinds = args.get("edge_types")
        edge_types: tuple[str, ...] | None = None
        if isinstance(raw_kinds, list):
            cleaned = tuple(str(k).strip() for k in raw_kinds if isinstance(k, str))
            if cleaned:
                edge_types = cleaned
        with _timed_tool(
            "traverse",
            database=db_t,
            to_database=to_db_s,
            max_depth=max_depth,
            top_k=top_k,
        ):
            return traverse_payload(
                graph_dir,
                from_table=from_t,
                to_table=to_t,
                database=db_t,
                to_database=to_db_s,
                max_depth=max_depth,
                top_k=top_k,
                edge_types=edge_types,
                max_inferred_hops=max_inf,
            )

    async def _handle_impact(args: dict[str, Any]) -> dict[str, Any]:
        tbl = str(args.get("table", "")).strip()
        db_t = str(args.get("database", "")).strip()
        if not tbl:
            return {"error": "Missing `table`"}
        if not db_t:
            return {"error": "Missing `database`"}
        col = args.get("column")
        col_s = str(col).strip() if col is not None else None
        if col_s == "":
            col_s = None
        max_depth = int(args.get("max_depth", 3))
        with _timed_tool(
            "impact",
            table=tbl,
            database=db_t,
            column=col_s,
            max_depth=max_depth,
        ):
            return impact_payload(
                graph_dir,
                table=tbl,
                database=db_t,
                column=col_s,
                max_depth=max_depth,
            )

    async def _handle_detect_changes(args: dict[str, Any]) -> dict[str, Any]:
        db_t = str(args.get("database", "")).strip()
        if not db_t:
            return {"error": "Missing `database`"}
        with _timed_tool("detect_changes", database=db_t):
            return detect_changes_payload(graph_dir, database=db_t)

    async def _handle_validate_sql(args: dict[str, Any]) -> dict[str, Any]:
        sql_s = str(args.get("sql", ""))
        db_t = str(args.get("database", "")).strip()
        dialect_raw = args.get("dialect")
        dialect_s = (
            str(dialect_raw).strip() if dialect_raw is not None else "postgres"
        )
        if not dialect_s:
            dialect_s = "postgres"
        if not sql_s.strip():
            return {"error": "Missing `sql`"}
        if not db_t:
            return {"error": "Missing `database`"}
        with _timed_tool("validate_sql", database=db_t, dialect=dialect_s):
            return validate_sql_payload(
                graph_dir, sql=sql_s, database=db_t, dialect=dialect_s
            )

    async def _handle_schema(args: dict[str, Any]) -> dict[str, Any]:
        db_t = str(args.get("database", "")).strip()
        if not db_t:
            return {"error": "Missing `database`"}
        label_raw = args.get("label")
        label_s = str(label_raw).strip() if label_raw is not None else None
        if label_s == "":
            label_s = None
        with _timed_tool("schema", database=db_t, label=label_s):
            return schema_payload(graph_dir, database=db_t, label=label_s)

    async def _handle_compile_metric(args: dict[str, Any]) -> dict[str, Any]:
        yaml_s = str(args.get("semantic_yaml", ""))
        metric_s = str(args.get("metric", "")).strip()
        db_t = str(args.get("database", "")).strip()
        if not yaml_s.strip():
            return {"error": "Missing `semantic_yaml`"}
        if not metric_s:
            return {"error": "Missing `metric`"}
        if not db_t:
            return {"error": "Missing `database`"}
        with _timed_tool("compile_metric", database=db_t, metric=metric_s):
            return compile_metric_payload(
                graph_dir,
                semantic_yaml=yaml_s,
                metric=metric_s,
                database=db_t,
            )

    registry.register(
        McpTool(
            name="list_databases",
            description="List all indexed database connections with table counts and staleness.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_handle_list_databases,
        )
    )
    registry.register(
        McpTool(
            name="schema",
            description=(
                "Discover node labels, edge types, and properties in the graph. "
                "Call before writing Cypher."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Connection name or logical database",
                    },
                    "label": {
                        "type": ["string", "null"],
                        "description": "Optional node label or edge type to filter",
                    },
                },
                "required": ["database"],
                "additionalProperties": False,
            },
            handler=_handle_schema,
        )
    )
    registry.register(
        McpTool(
            name="cypher",
            description=(
                "Read-only Kuzu Cypher against one indexed graph. Returns JSON rows. "
                "Mutating clauses (CREATE, DELETE, MERGE, SET) are rejected.\n\n"
                + format_catalog_summary()
                + "\n\nCall the `schema` tool for full property lists per label."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Single Cypher read query",
                    },
                    "database": {
                        "type": "string",
                        "description": "Connection name or logical database",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "default": 5,
                        "minimum": 0.1,
                        "maximum": 120,
                        "description": "Query wall-clock timeout in seconds (default 5)",
                    },
                },
                "required": ["query", "database"],
                "additionalProperties": False,
            },
            handler=_handle_cypher,
        )
    )
    registry.register(
        McpTool(
            name="query",
            description="BM25 keyword search over table and entity metadata (FTS5).",
            input_schema={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                    "db": {
                        "type": ["string", "null"],
                        "description": "Filter by connection_name or logical database name",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["q"],
                "additionalProperties": False,
            },
            handler=_handle_query,
        )
    )
    registry.register(
        McpTool(
            name="context",
            description=(
                "Full context for one physical table: description (dbt-aware), columns (SchemaColumn), "
                "relationships (FK + inferred), LINEAGE in/out, optional deprecation signal, "
                "linked entity (if any), cluster when present."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table name or schema.table",
                    },
                    "db": {
                        "type": ["string", "null"],
                        "description": "Connection name or logical database when multiple graphs exist",
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["summary", "standard", "full"],
                        "default": "standard",
                        "description": "summary | standard (default) | full",
                    },
                },
                "required": ["table"],
                "additionalProperties": False,
            },
            handler=_handle_context,
        )
    )
    registry.register(
        McpTool(
            name="traverse",
            description=(
                "Find a join path between two physical tables (precomputed or on-demand), "
                "with SQL JOIN hints. Cross-database paths use confirmed SAME_ENTITY links only."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "from_table": {
                        "type": "string",
                        "description": "Source table (name or schema.table)",
                    },
                    "to_table": {
                        "type": "string",
                        "description": "Target table (name or schema.table)",
                    },
                    "database": {
                        "type": "string",
                        "description": "Connection name or logical database (required if multiple graphs)",
                    },
                    "to_database": {
                        "type": ["string", "null"],
                        "description": "Optional target connection for cross-database traverse",
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 4,
                        "minimum": 1,
                        "maximum": 8,
                        "description": "Max hops per database segment (cross-DB may need more)",
                    },
                    "top_k": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Yen's K — return up to this many ranked alternative paths",
                    },
                    "edge_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["fk", "inferred"]},
                        "description": "Restrict adjacency to these edge kinds (default: both)",
                    },
                    "max_inferred_hops": {
                        "type": "integer",
                        "default": 2,
                        "minimum": 0,
                        "maximum": 8,
                        "description": "Cap on inferred-join hops per path (FK hops uncapped beyond max_depth)",
                    },
                },
                "required": ["from_table", "to_table", "database"],
                "additionalProperties": False,
            },
            handler=_handle_traverse,
        )
    )
    registry.register(
        McpTool(
            name="impact",
            description=(
                "Downstream tables reachable via FK and inferred join edges from a table, "
                "grouped by hop depth."
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
                    "column": {
                        "type": ["string", "null"],
                        "description": "If set, only follow outgoing edges whose source column matches",
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 8,
                    },
                },
                "required": ["table", "database"],
                "additionalProperties": False,
            },
            handler=_handle_impact,
        )
    )
    registry.register(
        McpTool(
            name="detect_changes",
            description=(
                "Compare live database schema to the last indexed snapshot. Read-only; "
                "does not apply changes — use pretensor reindex to update the graph."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "description": "Connection name or logical database name",
                    },
                },
                "required": ["database"],
                "additionalProperties": False,
            },
            handler=_handle_detect_changes,
        )
    )
    registry.register(
        McpTool(
            name="compile_metric",
            description=(
                "Compile a user-authored YAML metric to validated SQL against the "
                "indexed graph. Accepts the semantic-layer YAML inline and a metric "
                "name; returns SQL plus a validation report (missing tables/columns, "
                "invalid joins, suggestions)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "semantic_yaml": {
                        "type": "string",
                        "description": (
                            "Full YAML text of the semantic layer "
                            "(metric definitions, joins, and base tables)."
                        ),
                    },
                    "metric": {
                        "type": "string",
                        "description": "Name of the metric to compile",
                    },
                    "database": {
                        "type": "string",
                        "description": "Connection name or logical database",
                    },
                },
                "required": ["semantic_yaml", "metric", "database"],
                "additionalProperties": False,
            },
            handler=_handle_compile_metric,
        )
    )
    registry.register(
        McpTool(
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
            handler=_handle_validate_sql,
        )
    )

    return registry


def create_server(
    graph_dir: Path,
    *,
    visibility_path: Path | None = None,
    profile: str | None = None,
    config: PretensorConfig | None = None,
    extra_tools: list[McpTool] | None = None,
) -> Server[object, object]:
    """Build the low-level MCP :class:`Server` bound to ``graph_dir``.

    Args:
        graph_dir: Root directory of the Pretensor graph store.
        visibility_path: Optional path to a visibility filter config file.
        profile: Optional named profile for the server context.
        config: Optional central Pretensor configuration for MCP runtime wiring.
        extra_tools: Additional :class:`McpTool` instances to register on top
            of the 7 built-in OSS tools (e.g. Cloud tools).
    """
    set_server_context(
        build_server_context(
            graph_dir,
            visibility_path=visibility_path,
            profile=profile,
            config=config,
        )
    )

    registry = _build_oss_registry(graph_dir)
    for tool in extra_tools or []:
        registry.register(tool)

    server = Server[object, object](
        "pretensor",
        instructions=(
            "Pretensor graph MCP: discover indexed databases, search metadata, "
            "and fetch full context for physical tables (Kuzu schema graph)."
        ),
    )

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return registry.list_tools()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        try:
            return await registry.call_tool(name, arguments)
        except Exception as exc:
            logger.exception("Unexpected error dispatching MCP tool %r", name)
            return {"error": f"Internal server error: {exc}", "tool": name}

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=AnyUrl("pretensor://databases"),
                name="databases",
                title="All indexed databases",
                description="Registry overview (markdown)",
                mimeType="text/markdown",
            ),
            types.Resource(
                uri=AnyUrl("pretensor://cross-db/entities"),
                name="cross-db-entities",
                title="Cross-database entity map",
                description="Confirmed SAME_ENTITY links (markdown)",
                mimeType="text/markdown",
            ),
        ]

    @server.list_resource_templates()
    async def _list_resource_templates() -> list[types.ResourceTemplate]:
        return [
            types.ResourceTemplate(
                uriTemplate="pretensor://db/{name}/overview",
                name="db-overview",
                title="Per-database overview",
                description="Table count, entity count, staleness (markdown)",
                mimeType="text/markdown",
            ),
            types.ResourceTemplate(
                uriTemplate="pretensor://db/{name}/clusters",
                name="db-clusters",
                title="Per-database domain clusters",
                description="Leiden clusters and table groupings (markdown)",
                mimeType="text/markdown",
            ),
            types.ResourceTemplate(
                uriTemplate="pretensor://db/{name}/metrics",
                name="db-metrics",
                title="Per-database metric templates",
                description="MetricTemplate nodes: SQL, validation, dependencies (markdown)",
                mimeType="text/markdown",
            ),
        ]

    @server.read_resource()
    async def _read_resource(uri: AnyUrl) -> str:
        key = str(uri)
        try:
            if key == "pretensor://databases":
                return databases_resource_markdown(graph_dir)
            if key == "pretensor://cross-db/entities":
                return cross_db_entities_resource_markdown(graph_dir)
            match = _DB_OVERVIEW_PATTERN.match(key)
            if match:
                return db_overview_resource_markdown(graph_dir, match.group("name"))
            match_c = _DB_CLUSTERS_PATTERN.match(key)
            if match_c:
                return clusters_resource_markdown(graph_dir, match_c.group("name"))
            match_m = _DB_METRICS_PATTERN.match(key)
            if match_m:
                return metrics_resource_markdown(graph_dir, match_m.group("name"))
            return f"# Resource not found\n\nUnknown URI: `{key}`"
        except Exception as exc:
            logger.exception("Error reading MCP resource %r", key)
            return f"# Error\n\nFailed to load resource `{key}`: {exc}"

    return server


async def _run_async(
    graph_dir: Path,
    *,
    visibility_path: Path | None = None,
    profile: str | None = None,
    config: PretensorConfig | None = None,
) -> None:
    server = create_server(
        graph_dir,
        visibility_path=visibility_path,
        profile=profile,
        config=config,
    )
    init = server.create_initialization_options(
        notification_options=NotificationOptions(resources_changed=False)
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init)


def run_server(
    graph_dir: Path,
    *,
    visibility_path: Path | None = None,
    profile: str | None = None,
    config: PretensorConfig | None = None,
) -> None:
    """Start the MCP server on stdio (blocking)."""
    asyncio.run(
        _run_async(
            graph_dir,
            visibility_path=visibility_path,
            profile=profile,
            config=config,
        )
    )
