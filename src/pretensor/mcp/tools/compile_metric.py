"""MCP ``compile_metric`` tool payload — compile YAML metrics to validated SQL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pretensor.introspection.models.semantic import (
    SemanticLayer as SemanticLayerModel,
)
from pretensor.mcp.tool_registry import McpTool
from pretensor.semantic.compiler import MetricCompileError, MetricSqlCompiler

from ..service_registry import (
    load_registry,
    open_store_for_entry,
    release_store,
    resolve_registry_entry,
)

__all__ = ["compile_metric_payload", "create_tool"]


def compile_metric_payload(
    graph_dir: Path,
    *,
    semantic_yaml: str,
    metric: str,
    database: str,
) -> dict[str, Any]:
    """Build JSON-serializable payload for ``compile_metric``.

    Args:
        graph_dir: Root directory of the Pretensor graph store.
        semantic_yaml: Full YAML text of the semantic layer (inline, not a path).
        metric: Name of the metric to compile.
        database: ``connection_name`` or logical ``database`` key.
    """
    if not semantic_yaml.strip():
        return {"error": "Missing or empty `semantic_yaml`"}
    if not metric.strip():
        return {"error": "Missing or empty `metric`"}
    if not database.strip():
        return {"error": "Missing or empty `database`"}

    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, database)
    if entry is None:
        return {"error": f"No registry entry matches database {database!r}"}

    try:
        layer = SemanticLayerModel.from_yaml(semantic_yaml)
    except Exception as exc:
        return {"error": f"Failed to parse semantic_yaml: {exc}"}

    store = open_store_for_entry(entry)
    try:
        compiler = MetricSqlCompiler(
            store,
            connection_name=entry.connection_name,
            database_key=entry.database,
        )
        try:
            compiled = compiler.compile(layer, metric)
        except MetricCompileError as exc:
            return {"error": str(exc)}
    finally:
        release_store(store)

    return {
        "metric": compiled.metric,
        "entity": compiled.entity,
        "sql": compiled.sql,
        "dialect": compiled.dialect,
        "warnings": compiled.warnings,
        "valid": compiled.validation.valid,
        "syntax_errors": compiled.validation.syntax_errors,
        "missing_tables": compiled.validation.missing_tables,
        "missing_columns": compiled.validation.missing_columns,
        "invalid_joins": [
            {
                "message": j.message,
                "left_table": j.left_table,
                "right_table": j.right_table,
            }
            for j in compiled.validation.invalid_joins
        ],
        "suggestions": compiled.validation.suggestions,
    }


def create_tool(graph_dir: Path) -> McpTool:
    from ._timed import timed_tool

    async def _handle(args: dict) -> dict:
        yaml_s = str(args.get("semantic_yaml", ""))
        metric_s = str(args.get("metric", "")).strip()
        db_t = str(args.get("database", "")).strip()
        if not yaml_s.strip():
            return {"error": "Missing `semantic_yaml`"}
        if not metric_s:
            return {"error": "Missing `metric`"}
        if not db_t:
            return {"error": "Missing `database`"}
        with timed_tool("compile_metric", graph_dir, database=db_t, metric=metric_s):
            return compile_metric_payload(
                graph_dir,
                semantic_yaml=yaml_s,
                metric=metric_s,
                database=db_t,
            )

    return McpTool(
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
        handler=_handle,
    )
