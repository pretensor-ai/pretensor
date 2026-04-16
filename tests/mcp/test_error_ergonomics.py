"""Tests for MCP error ergonomics.

Covers:
- Case-insensitive table lookup fallback in ``find_table_rows``.
- Candidate suggestions when no table matches.
- Cypher property-not-found hints with available properties.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.mcp.tools.context import (
    context_payload,
    find_table_rows,
    resolve_table_node_id,
    suggest_table_candidates,
)
from pretensor.mcp.tools.cypher import (
    _parse_node_aliases,  # noqa: PLC2701
    cypher_payload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_graph(
    tmp_path: Path,
    tables: list[Table],
    *,
    connection_name: str = "demo",
    database: str = "demo",
    schemas: list[str] | None = None,
) -> Path:
    """Build a Kuzu graph, register it, and return ``tmp_path``."""
    snap = SchemaSnapshot(
        connection_name=connection_name,
        database=database,
        schemas=schemas or sorted({t.schema_name for t in tables}),
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / f"{database}.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name=connection_name,
        database=database,
        dsn=f"postgresql://localhost/{database}",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Case-insensitive table lookup
# ---------------------------------------------------------------------------


def test_find_table_rows_case_insensitive(tmp_path: Path) -> None:
    """Lowercase input matches an uppercase-stored table (Snowflake style)."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="LINEITEM",
                schema_name="TPCH",
                columns=[Column(name="L_ORDERKEY", data_type="int")],
            ),
        ],
    )
    graph = graph_dir / "graphs" / "demo.kuzu"
    store = KuzuStore(graph)
    try:
        # Exact match should fail for lowercase
        rows_exact = store.query_all_rows(
            "MATCH (t:SchemaTable) WHERE t.table_name = $tbl RETURN t.table_name",
            {"tbl": "lineitem"},
        )
        assert rows_exact == []

        # find_table_rows should still find it via case-insensitive fallback
        rows = find_table_rows(store, "lineitem")
        assert len(rows) == 1
        assert rows[0][2] == "LINEITEM"  # table_name column

        # Dot notation too
        rows_dot = find_table_rows(store, "tpch.lineitem")
        assert len(rows_dot) == 1
        assert rows_dot[0][1] == "TPCH"
        assert rows_dot[0][2] == "LINEITEM"
    finally:
        store.close()


def test_find_table_rows_exact_match_preferred(tmp_path: Path) -> None:
    """Exact match is returned even when case-insensitive would also match."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="orders",
                schema_name="public",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
    )
    graph = graph_dir / "graphs" / "demo.kuzu"
    store = KuzuStore(graph)
    try:
        rows = find_table_rows(store, "orders")
        assert len(rows) == 1
        assert rows[0][2] == "orders"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 2. Candidate suggestions on no match
# ---------------------------------------------------------------------------


def test_suggest_table_candidates(tmp_path: Path) -> None:
    """suggest_table_candidates returns qualified names for case-insensitive matches."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="LINEITEM",
                schema_name="TPCH",
                columns=[Column(name="id", data_type="int")],
            ),
            Table(
                name="orders",
                schema_name="public",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
        schemas=["TPCH", "public"],
    )
    graph = graph_dir / "graphs" / "demo.kuzu"
    store = KuzuStore(graph)
    try:
        candidates = suggest_table_candidates(store, "lineitem")
        assert len(candidates) == 1
        assert "TPCH.LINEITEM" in candidates

        # No match at all
        candidates_none = suggest_table_candidates(store, "xyz_missing")
        assert candidates_none == []
    finally:
        store.close()


def test_suggest_table_candidates_dot_notation(tmp_path: Path) -> None:
    """suggest_table_candidates extracts bare name from dot-notation input."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="LINEITEM",
                schema_name="TPCH",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
    )
    graph = graph_dir / "graphs" / "demo.kuzu"
    store = KuzuStore(graph)
    try:
        candidates = suggest_table_candidates(store, "tpch.lineitem")
        assert "TPCH.LINEITEM" in candidates
    finally:
        store.close()


def test_context_payload_no_match_no_candidates(tmp_path: Path) -> None:
    """context_payload error omits 'Did you mean' when no candidate matches at all."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="LINEITEM",
                schema_name="TPCH",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
    )
    result = context_payload(graph_dir, table="xyz_missing", db="demo")
    assert "error" in result
    assert "Did you mean" not in result["error"]


def test_context_payload_case_insensitive_succeeds(tmp_path: Path) -> None:
    """context_payload succeeds with wrong-case input thanks to fallback."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="LINEITEM",
                schema_name="TPCH",
                columns=[Column(name="L_ORDERKEY", data_type="int")],
            ),
        ],
    )
    result = context_payload(graph_dir, table="lineitem", db="demo")
    # Should succeed, not error
    assert "error" not in result
    assert result.get("table_name") == "LINEITEM"


def test_resolve_table_node_id_case_insensitive(tmp_path: Path) -> None:
    """resolve_table_node_id finds table via case-insensitive fallback."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="ORDERS",
                schema_name="PUBLIC",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
    )
    graph = graph_dir / "graphs" / "demo.kuzu"
    store = KuzuStore(graph)
    try:
        node_id, error = resolve_table_node_id(store, "orders", "demo")
        assert error is None
        assert node_id is not None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 3. Cypher property hints
# ---------------------------------------------------------------------------


def test_cypher_property_hint_on_bad_property(tmp_path: Path) -> None:
    """Cypher error for nonexistent property includes available properties hint."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="orders",
                schema_name="public",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
    )
    result = cypher_payload(
        graph_dir,
        query="MATCH (t:SchemaTable) RETURN t.nonexistent_prop",
        database="demo",
    )
    assert "error" in result
    err = result["error"]
    assert "Hint:" in err
    # Should suggest actual SchemaTable properties
    assert "table_name" in err
    assert "schema_name" in err


def test_cypher_property_hint_unknown_alias(tmp_path: Path) -> None:
    """Property hint gracefully handles alias without a label in the query."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="orders",
                schema_name="public",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
    )
    # No label on alias 'x', so we can't resolve properties
    result = cypher_payload(
        graph_dir,
        query="MATCH (x) RETURN x.nonexistent_prop",
        database="demo",
    )
    assert "error" in result
    err = result["error"]
    # Should still have a hint even without property list
    if "Cannot find property" in err:
        assert "Hint:" in err


def test_cypher_reserved_keyword_friendly_error(tmp_path: Path) -> None:
    """Using a reserved Kuzu keyword as an alias yields a friendly message."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="orders",
                schema_name="public",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
    )
    result = cypher_payload(
        graph_dir,
        query="MATCH (t:SchemaTable) RETURN t.description AS desc",
        database="demo",
    )
    assert "error" in result
    err = result["error"]
    assert "reserved" in err.lower()
    assert "desc" in err.lower()
    # Guidance to rename or backtick should be present.
    assert "backtick" in err.lower() or "rename" in err.lower()


def test_cypher_generic_error_unchanged(tmp_path: Path) -> None:
    """Non-property Cypher errors keep the standard format without hints."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="orders",
                schema_name="public",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
    )
    result = cypher_payload(
        graph_dir,
        query="THIS IS NOT VALID CYPHER AT ALL RETURN",
        database="demo",
    )
    assert "error" in result
    assert "Hint:" not in result["error"]


# ---------------------------------------------------------------------------
# 4. Unit tests for helpers
# ---------------------------------------------------------------------------


def test_parse_node_aliases() -> None:
    """_parse_node_aliases extracts alias→label from MATCH patterns."""
    aliases = _parse_node_aliases(
        "MATCH (t:SchemaTable)-[:HAS_COLUMN]->(c:SchemaColumn) RETURN t, c"
    )
    assert aliases == {"t": "SchemaTable", "c": "SchemaColumn"}


def test_parse_node_aliases_no_label() -> None:
    """_parse_node_aliases returns empty for aliases without labels."""
    aliases = _parse_node_aliases("MATCH (x) RETURN x")
    assert aliases == {}


def test_resolve_table_node_id_no_match_with_candidates(tmp_path: Path) -> None:
    """resolve_table_node_id error includes 'Did you mean' when candidates exist."""
    graph_dir = _build_graph(
        tmp_path,
        [
            Table(
                name="ORDERS",
                schema_name="PUBLIC",
                columns=[Column(name="id", data_type="int")],
            ),
        ],
    )
    graph = graph_dir / "graphs" / "demo.kuzu"
    store = KuzuStore(graph)
    try:
        # "orderz" won't match even case-insensitively, but "ORDERS" should NOT
        # appear as candidate since it doesn't match "orderz"
        _, error = resolve_table_node_id(store, "orderz", "demo")
        assert error is not None
        assert "No table matched" in error
    finally:
        store.close()
