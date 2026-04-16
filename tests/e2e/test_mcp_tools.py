"""E2E tests — all MCP payload functions against live Pagila graph."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pretensor.mcp.tools.context import context_payload
from pretensor.mcp.tools.cypher import cypher_payload
from pretensor.mcp.tools.detect_changes import detect_changes_payload
from pretensor.mcp.tools.impact import impact_payload
from pretensor.mcp.tools.list import list_databases_payload
from pretensor.mcp.tools.search import query_payload
from pretensor.mcp.tools.traverse import traverse_payload

if not os.getenv("PRETENSOR_E2E"):
    pytest.skip("set PRETENSOR_E2E=1", allow_module_level=True)

pytestmark = pytest.mark.e2e


def test_list_databases(graph_dir: Path) -> None:
    result = list_databases_payload(graph_dir)
    databases = [db.get("connection_name") or db.get("name") for db in result.get("databases", [])]
    assert any("pagila" in str(d) for d in databases), (
        f"'pagila' not found in databases: {databases}"
    )
    total = sum(db.get("table_count", 0) for db in result.get("databases", []))
    assert total >= 6, f"Expected ≥6 total tables across databases, got {total}"


def test_query_hits_for_film(graph_dir: Path) -> None:
    result = query_payload(graph_dir, q="film rental inventory", db="pagila")
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    hits = result.get("results", result.get("hits", []))
    assert len(hits) >= 1, f"Expected ≥1 hit for 'film rental inventory', got {hits}"


def test_context_film(graph_dir: Path) -> None:
    result = context_payload(graph_dir, table="film", db="pagila")
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    columns = result.get("columns", [])
    assert len(columns) >= 5, f"Expected ≥5 columns for 'film', got {len(columns)}"


def test_context_full_detail_rental(graph_dir: Path) -> None:
    result = context_payload(graph_dir, table="rental", db="pagila", detail="full")
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert "relationships" in result, (
        f"Expected 'relationships' key in full detail response; got keys: {list(result)}"
    )


def test_cypher_reads_tables(graph_dir: Path) -> None:
    result = cypher_payload(
        graph_dir,
        query="MATCH (t:SchemaTable) RETURN t.table_name LIMIT 5",
        database="pagila",
    )
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    rows = result.get("rows", [])
    assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"


def test_cypher_mutation_blocked(graph_dir: Path) -> None:
    result = cypher_payload(
        graph_dir,
        query="CREATE (n:Foo {x:1})",
        database="pagila",
    )
    assert "error" in result, (
        f"Expected 'error' key for mutation query; got: {result}"
    )


def test_traverse_film_to_rental(graph_dir: Path) -> None:
    # Should return a path or a graceful no-path — must not raise
    result = traverse_payload(
        graph_dir,
        from_table="film",
        to_table="rental",
        database="pagila",
    )
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"


def test_impact_film(graph_dir: Path) -> None:
    result = impact_payload(graph_dir, table="film", database="pagila")
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert "total_affected" in result, (
        f"Expected 'total_affected' key in impact result; got keys: {list(result)}"
    )


def test_detect_changes_no_drift(graph_dir: Path) -> None:
    result = detect_changes_payload(graph_dir, database="pagila")
    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    changes = result.get("changes", None)
    up_to_date = result.get("up_to_date", None)
    assert changes == [] or up_to_date is True, (
        f"Expected no drift; got changes={changes}, up_to_date={up_to_date}"
    )


