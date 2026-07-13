"""Security regression tests for the ``cypher`` read-only guard.

The guard moved from a 4-keyword blocklist to a positive allowlist plus a
pre-execution multi-statement rejection plus a read-only driver backstop.
These tests assert the previously-bypassing statements are rejected *and* leave
the graph and filesystem unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.mcp.service import cypher_payload
from pretensor.mcp.tools.cypher import assert_read_only_cypher


def _build_graph(tmp_path: Path) -> Path:
    """Build a tiny indexed graph and register it under the name ``demo``."""
    users = Table(
        name="users",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        comment="People who DROP by",
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[users],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://localhost/demo",
        graph_path=graph,
    )
    reg.save()
    return graph


def _table_count(graph: Path) -> int:
    store = KuzuStore(graph, read_only=True)
    try:
        rows = store.query_all_rows("MATCH (t:SchemaTable) RETURN count(*)")
        return int(rows[0][0])
    finally:
        store.close()


# --- guard-level (no graph required) ---------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "COPY (MATCH (t:SchemaTable) RETURN t.table_name) TO '/tmp/exfil.csv'",
        "DROP TABLE SchemaTable",
        "ALTER TABLE SchemaTable ADD evil STRING",
        "EXPORT DATABASE '/tmp/dump'",
        "IMPORT DATABASE '/tmp/dump'",
        "ATTACH 'host=h user=postgres' AS pg (dbtype postgres)",
        "INSTALL httpfs",
        "LOAD EXTENSION httpfs",
        "CREATE (n:Foo)",
        "MATCH (t:SchemaTable) DELETE t",
        "MATCH (t:SchemaTable) SET t.table_name = 'x'",
        "MATCH (t:SchemaTable) DETACH DELETE t",
        "MATCH (t:SchemaTable) RETURN t.table_name; DROP TABLE SchemaTable;",
        "MATCH (t:SchemaTable) RETURN t LIMIT 1; CREATE (:Evil);",
    ],
)
def test_guard_rejects_dangerous_statements(query: str) -> None:
    with pytest.raises(ValueError, match="read queries"):
        assert_read_only_cypher(query)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (t:SchemaTable) RETURN t.table_name LIMIT 1",
        "OPTIONAL MATCH (t:SchemaTable) RETURN t",
        "MATCH (c:SchemaColumn) RETURN c.comment AS comment",
        "MATCH (t:SchemaTable) WHERE t.comment CONTAINS 'DROP me' RETURN t",
        "CALL table_info('SchemaTable') RETURN *",
        "MATCH (t:SchemaTable) RETURN count(*);",
        "WITH 1 AS x RETURN x",
        "UNWIND [1, 2, 3] AS n RETURN n",
    ],
)
def test_guard_allows_legitimate_reads(query: str) -> None:
    assert_read_only_cypher(query)  # must not raise


# --- end-to-end: graph + filesystem must be unchanged ----------------------


def test_copy_to_does_not_write_host_file(tmp_path: Path) -> None:
    _build_graph(tmp_path)
    target = tmp_path / "exfil.csv"
    out = cypher_payload(
        tmp_path,
        query=f"COPY (MATCH (t:SchemaTable) RETURN t.table_name) TO '{target}'",
        database="demo",
    )
    assert "rows" not in out
    assert "error" in out
    assert not target.exists()


def test_export_database_does_not_write_host_dir(tmp_path: Path) -> None:
    _build_graph(tmp_path)
    target = tmp_path / "dump"
    out = cypher_payload(
        tmp_path,
        query=f"EXPORT DATABASE '{target}'",
        database="demo",
    )
    assert "error" in out
    assert not target.exists()


def test_multi_statement_chain_leaves_graph_unchanged(tmp_path: Path) -> None:
    graph = _build_graph(tmp_path)
    before = _table_count(graph)
    out = cypher_payload(
        tmp_path,
        query="MATCH (t:SchemaTable) RETURN t.table_name; DROP TABLE SchemaTable;",
        database="demo",
    )
    assert "error" in out
    assert "rows" not in out
    assert _table_count(graph) == before


def test_driver_read_only_blocks_mutation(tmp_path: Path) -> None:
    """Defense-in-depth: a read-only store cannot commit a graph mutation."""
    graph = _build_graph(tmp_path)
    store = KuzuStore(graph, read_only=True)
    try:
        with pytest.raises(Exception, match="read-only"):
            store.execute("CREATE (:SchemaTable {node_id: 'evil'})")
    finally:
        store.close()


def test_benign_read_still_returns_rows(tmp_path: Path) -> None:
    _build_graph(tmp_path)
    out = cypher_payload(
        tmp_path,
        query="MATCH (t:SchemaTable) RETURN t.table_name AS name ORDER BY name",
        database="demo",
    )
    assert "error" not in out
    assert {row["name"] for row in out["rows"]} == {"users"}
