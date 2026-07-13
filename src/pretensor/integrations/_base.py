"""Shared tool-function factory for all agent-framework adapters.

Each returned callable has a typed signature and docstring so that
LangChain, LlamaIndex, and Google ADK can automatically infer the
tool schema without any manual JSON Schema duplication.

The functions delegate directly to the same payload functions used by
the MCP server — no logic is duplicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pretensor.mcp.tools.context import context_payload
from pretensor.mcp.tools.impact import impact_payload
from pretensor.mcp.tools.schema import schema_payload
from pretensor.mcp.tools.search import query_payload
from pretensor.mcp.tools.traverse import traverse_payload
from pretensor.mcp.tools.validate_sql import validate_sql_payload

__all__ = ["make_tool_functions"]


def _make_schema(graph_dir: Path) -> Callable[..., dict[str, Any]]:
    def schema(database: str, label: str | None = None) -> dict[str, Any]:
        """Discover node labels, edge types, and properties in the graph. Call before writing Cypher."""
        return schema_payload(graph_dir, database=database, label=label)

    return schema


def _make_context(graph_dir: Path) -> Callable[..., dict[str, Any]]:
    def context(
        table: str,
        db: str | None = None,
        detail: str = "standard",
    ) -> dict[str, Any]:
        """Full context for one physical table: columns, relationships, lineage, cluster. detail: summary | standard | full."""
        return context_payload(graph_dir, table=table, db=db, detail=detail)  # type: ignore[arg-type]

    return context


def _make_traverse(graph_dir: Path) -> Callable[..., dict[str, Any]]:
    def traverse(
        from_table: str,
        to_table: str,
        database: str,
        max_depth: int = 4,
        top_k: int = 3,
        edge_types: list[str] | None = None,
        max_inferred_hops: int = 2,
    ) -> dict[str, Any]:
        """Find a join path between two physical tables with SQL JOIN hints. edge_types: fk | inferred (default: both)."""
        et: tuple[str, ...] | None = None
        if edge_types:
            et = tuple(edge_types)
        return traverse_payload(
            graph_dir,
            from_table=from_table,
            to_table=to_table,
            database=database,
            max_depth=max_depth,
            top_k=max(1, min(10, top_k)),
            edge_types=et,
            max_inferred_hops=max(0, min(8, max_inferred_hops)),
        )

    return traverse


def _make_impact(graph_dir: Path) -> Callable[..., dict[str, Any]]:
    def impact(
        table: str,
        database: str,
        column: str | None = None,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Downstream tables reachable via FK and inferred join edges from a table, grouped by hop depth."""
        return impact_payload(
            graph_dir,
            table=table,
            database=database,
            column=column,
            max_depth=max_depth,
        )

    return impact


def _make_query(graph_dir: Path) -> Callable[..., dict[str, Any]]:
    def query(
        q: str,
        db: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """BM25 keyword search over table and entity metadata."""
        return query_payload(graph_dir, q=q, db=db, limit=limit)

    return query


def _make_validate_sql(graph_dir: Path) -> Callable[..., dict[str, Any]]:
    def validate_sql(
        sql: str,
        database: str,
        dialect: str = "postgres",
    ) -> dict[str, Any]:
        """Validate a SQL statement against the indexed graph: unknown tables/columns, invalid joins, fuzzy suggestions."""
        return validate_sql_payload(graph_dir, sql=sql, database=database, dialect=dialect)

    return validate_sql


def make_tool_functions(graph_dir: Path) -> list[Callable[..., dict[str, Any]]]:
    """Return six graph-tool callables bound to ``graph_dir``.

    Each callable has typed parameters and a docstring so that LangChain,
    LlamaIndex, and Google ADK can automatically derive the tool schema.

    Args:
        graph_dir: Root directory of the Pretensor graph store (must be
            pre-indexed via ``pretensor index``).

    Returns:
        [schema, context, traverse, impact, query, validate_sql]
    """
    resolved = Path(graph_dir).resolve()
    return [
        _make_schema(resolved),
        _make_context(resolved),
        _make_traverse(resolved),
        _make_impact(resolved),
        _make_query(resolved),
        _make_validate_sql(resolved),
    ]
