"""Smoke tests for GraphStore – delegation and write/read round-trips."""

from __future__ import annotations

from pathlib import Path

from pretensor.core.graph_store import GraphStore, TableEmbeddingRow
from pretensor.core.store import KuzuStore
from pretensor.core.store import TableEmbeddingRow as StoreTableEmbeddingRow
from pretensor.graph_models.node import GraphNode


def _make_node(node_id: str = "cn::db::public::users") -> GraphNode:
    return GraphNode(
        node_id=node_id,
        connection_name="cn",
        database="db",
        schema_name="public",
        table_name="users",
    )


def test_graph_store_is_composed(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "test.kuzu")
    try:
        assert isinstance(store._graph, GraphStore)
    finally:
        store.close()


def test_upsert_table_round_trip_via_facade(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "test.kuzu")
    store.ensure_schema()
    try:
        store.upsert_table(_make_node())
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable) RETURN t.node_id, t.table_name"
        )
        assert len(rows) == 1
        assert rows[0][0] == "cn::db::public::users"
        assert rows[0][1] == "users"
    finally:
        store.close()


def test_upsert_table_round_trip_via_graph_store(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "test.kuzu")
    store.ensure_schema()
    try:
        store._graph.upsert_table(_make_node())
        rows = store._graph._runner.query_all_rows(
            "MATCH (t:SchemaTable) RETURN t.node_id"
        )
        assert len(rows) == 1
        assert rows[0][0] == "cn::db::public::users"
    finally:
        store.close()


def test_table_embedding_row_importable_from_store() -> None:
    assert TableEmbeddingRow is StoreTableEmbeddingRow


def test_iter_table_embeddings_no_vectors(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "test.kuzu")
    store.ensure_schema()
    try:
        store.upsert_table(_make_node())
        rows = list(store.iter_table_embeddings())
        assert len(rows) == 1
        assert rows[0].table_name == "users"
        assert rows[0].embedding is None
    finally:
        store.close()


def test_clear_graph_removes_tables(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "test.kuzu")
    store.ensure_schema()
    try:
        store.upsert_table(_make_node())
        store.clear_graph()
        rows = store.query_all_rows("MATCH (t:SchemaTable) RETURN t.node_id")
        assert rows == []
    finally:
        store.close()
