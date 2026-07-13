"""Tests for SchemaTable.embedding column and KuzuStore.set_table_embedding."""

from __future__ import annotations

import pytest
from tests.query_helpers import first_cell, single_query_result

from pretensor.core.ids import table_node_id
from pretensor.core.store import _SCHEMA_TABLE_EMBEDDING_DIM, KuzuStore
from pretensor.intelligence.embeddings import EMBEDDING_DIM


def test_schema_table_embedding_dim_matches_source_of_truth() -> None:
    """Drift guard: the constant duplicated in ``core.store`` (to avoid a circular
    import with ``intelligence/__init__.py``) must match the canonical
    ``EMBEDDING_DIM`` in ``intelligence.embeddings``.
    """
    assert _SCHEMA_TABLE_EMBEDDING_DIM == EMBEDDING_DIM


def _seed_table(store: KuzuStore, tid: str) -> None:
    store.execute(
        """
        CREATE (t:SchemaTable {
            node_id: $tid, connection_name: 'c', database: 'd',
            schema_name: 's', table_name: 't'
        })
        """,
        {"tid": tid},
    )


def test_embedding_column_present_reads_null(graph_store: KuzuStore) -> None:
    """(a) column added to schema; (c) None read on a table with no vector."""
    tid = table_node_id("c", "s", "t")
    _seed_table(graph_store, tid)
    result = single_query_result(
        graph_store,
        "MATCH (t:SchemaTable {node_id: $id}) RETURN t.embedding",
        {"id": tid},
    )
    assert first_cell(result) is None


def test_set_table_embedding_round_trip(graph_store: KuzuStore) -> None:
    """(b) set/get round-trip on a 384-dim vector."""
    tid = table_node_id("c", "s", "t")
    _seed_table(graph_store, tid)
    vec = [float(i) / EMBEDDING_DIM for i in range(EMBEDDING_DIM)]
    graph_store.set_table_embedding(tid, vec)
    result = single_query_result(
        graph_store,
        "MATCH (t:SchemaTable {node_id: $id}) RETURN t.embedding",
        {"id": tid},
    )
    got = first_cell(result)
    assert got is not None
    assert len(got) == EMBEDDING_DIM
    assert got[0] == pytest.approx(vec[0])
    assert got[-1] == pytest.approx(vec[-1])


def test_set_table_embedding_none_clears(graph_store: KuzuStore) -> None:
    """Writing None after a real vector clears the field."""
    tid = table_node_id("c", "s", "t")
    _seed_table(graph_store, tid)
    graph_store.set_table_embedding(tid, [0.1] * EMBEDDING_DIM)
    graph_store.set_table_embedding(tid, None)
    result = single_query_result(
        graph_store,
        "MATCH (t:SchemaTable {node_id: $id}) RETURN t.embedding",
        {"id": tid},
    )
    assert first_cell(result) is None


@pytest.mark.parametrize("bad_len", [0, 1, 383, 385, 768])
def test_set_table_embedding_wrong_dim_raises(
    graph_store: KuzuStore, bad_len: int
) -> None:
    """(d) wrong-dim write raises ValueError before hitting Kuzu."""
    tid = table_node_id("c", "s", "t")
    _seed_table(graph_store, tid)
    with pytest.raises(ValueError, match=str(EMBEDDING_DIM)):
        graph_store.set_table_embedding(tid, [0.0] * bad_len)


def test_has_any_table_embeddings_false_then_true(graph_store: KuzuStore) -> None:
    """has_any_table_embeddings flips once any table carries a vector."""
    tid = table_node_id("c", "s", "t")
    _seed_table(graph_store, tid)
    assert graph_store.has_any_table_embeddings() is False
    graph_store.set_table_embedding(tid, [0.1] * EMBEDDING_DIM)
    assert graph_store.has_any_table_embeddings() is True
    graph_store.set_table_embedding(tid, None)
    assert graph_store.has_any_table_embeddings() is False
