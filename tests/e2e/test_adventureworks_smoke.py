"""Scale smoke for MCP tools against AdventureWorks (~40 tables, 5 schemas).

Pagila (8 tables, 2 schemas) doesn't exercise multi-schema or wide-graph payload
shapes; AdventureWorks does. These are not regression tests — they catch
pagination, schema-resolution, and serialization bugs that only surface on
larger graphs.
"""

from __future__ import annotations

from pathlib import Path

from pretensor.mcp.tools.context import context_payload
from pretensor.mcp.tools.list import list_databases_payload


def test_list_databases_includes_adventureworks(
    graph_dir_adventureworks: Path,
) -> None:
    result = list_databases_payload(graph_dir_adventureworks)
    databases = result.get("databases", [])
    aw = next((db for db in databases if db.get("name") == "adventureworks"), None)
    assert aw is not None, (
        f"'adventureworks' not found in databases: {[db.get('name') for db in databases]}"
    )
    # AW DDL ships ~39 tables across 5 schemas — sanity that the introspect
    # → build pipeline crossed schema boundaries.
    assert aw.get("table_count", 0) >= 30, (
        f"Expected ≥30 tables for adventureworks, got {aw.get('table_count')}"
    )


def test_context_product_returns_rich_graph(
    graph_dir_adventureworks: Path,
) -> None:
    """``production.product`` is a hub table — context must resolve schema-qualified
    names without pagination or serialization errors at AW scale."""
    result = context_payload(
        graph_dir_adventureworks,
        table="production.product",
        db="adventureworks",
    )
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    columns = result.get("columns", [])
    assert len(columns) >= 5, f"Expected ≥5 columns for production.product, got {len(columns)}"
