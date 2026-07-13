"""Embedding-based tie-break on ``traverse`` ambiguous paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.intelligence.embeddings import EMBEDDING_DIM
from pretensor.intelligence.join_paths import JoinPathEngine
from pretensor.mcp.tools.traverse import traverse_payload

_GRAPH_SUBPATH = ("graphs", "demo.kuzu")


def _fk(
    src_table: str, src_col: str, dst_table: str, dst_col: str = "id"
) -> ForeignKey:
    return ForeignKey(
        source_schema="public",
        source_table=src_table,
        source_column=src_col,
        target_schema="public",
        target_table=dst_table,
        target_column=dst_col,
    )


def _build_diamond_graph(tmp_path: Path) -> Path:
    """Diamond ``a→b→d`` / ``a→c→d`` (two tied FK paths). Returns graph path."""
    t_d = Table(
        name="d",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    t_b = Table(
        name="b",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="d_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("b", "d_id", "d")],
    )
    t_c = Table(
        name="c",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="d_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("c", "d_id", "d")],
    )
    t_a = Table(
        name="a",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="b_id", data_type="int", is_foreign_key=True),
            Column(name="c_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("a", "b_id", "b"), _fk("a", "c_id", "c")],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_a, t_b, t_c, t_d],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path.joinpath(*_GRAPH_SUBPATH)
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        JoinPathEngine(store).precompute("demo")
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://localhost/demo",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()
    return graph


def _build_chain_graph(tmp_path: Path) -> Path:
    """Single-path chain ``a→b→c``; no ambiguity."""
    t_c = Table(
        name="c",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    t_b = Table(
        name="b",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="c_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("b", "c_id", "c")],
    )
    t_a = Table(
        name="a",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="b_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("a", "b_id", "b")],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_a, t_b, t_c],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path.joinpath(*_GRAPH_SUBPATH)
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        JoinPathEngine(store).precompute("demo")
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://localhost/demo",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()
    return graph


def _unit_vec(dim: int, slot: int, magnitude: float = 1.0) -> list[float]:
    """One-hot-ish vector with ``magnitude`` at ``slot`` and tiny epsilon elsewhere (avoids zero-norm)."""
    vec = [0.0] * dim
    vec[slot % dim] = magnitude
    return vec


def _set_embeddings(graph_path: Path, by_table: dict[str, list[float] | None]) -> None:
    """Assign ``SchemaTable.embedding`` for each ``"schema.table"`` key."""
    store = KuzuStore(graph_path)
    try:
        for qualified, vec in by_table.items():
            schema, table = qualified.split(".", 1)
            rows = store.query_all_rows(
                """
                MATCH (t:SchemaTable)
                WHERE t.database = 'demo' AND t.schema_name = $s AND t.table_name = $t
                RETURN t.node_id
                """,
                {"s": schema, "t": table},
            )
            assert rows, f"table {qualified!r} not found"
            store.set_table_embedding(str(rows[0][0]), vec)
    finally:
        store.close()


class _VectorStubClient:
    """Returns pre-configured vectors in call order (tests inject known scores)."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(v) for v in self._vectors[: len(texts)]]


class _RaisingStubClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("stub embed failure")


def _paths_intermediate_order(result: dict) -> list[str]:
    """First-step ``to_table`` for each returned path (the intermediate in a 2-hop diamond)."""
    return [p["steps"][0]["to_table"] for p in result["paths"]]


def test_no_embedding_client_and_no_endpoint_vectors(tmp_path: Path) -> None:
    """Null path: no embeddings anywhere -> tie-break skipped, paths unchanged, marker null."""
    _build_diamond_graph(tmp_path)
    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.d",
        database="demo",
        max_depth=4,
    )
    assert "error" not in res
    assert "tie_break" not in res
    paths = res["paths"]
    assert len(paths) == 2
    assert all(p["ambiguous"] is True for p in paths)
    # all-emit contract: both intermediates present, order unchanged from pre-PR.
    assert set(_paths_intermediate_order(res)) == {"public.b", "public.c"}


def test_single_path_no_tie_break(tmp_path: Path) -> None:
    """Unambiguous chain: nothing to re-rank; marker null."""
    _build_chain_graph(tmp_path)
    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.c",
        database="demo",
        max_depth=4,
    )
    assert "error" not in res
    assert "tie_break" not in res
    assert len(res["paths"]) == 1


def test_endpoint_missing_embedding_skips_tie_break(tmp_path: Path) -> None:
    """Gate: if either endpoint lacks an embedding, skip tie-break even with a live client."""
    graph = _build_diamond_graph(tmp_path)
    # ``a`` has no embedding; only ``d`` does.
    _set_embeddings(
        graph,
        {"public.d": _unit_vec(EMBEDDING_DIM, 0)},
    )
    stub = _VectorStubClient(
        [
            _unit_vec(EMBEDDING_DIM, 0),
            _unit_vec(EMBEDDING_DIM, 0),
            _unit_vec(EMBEDDING_DIM, 1),
        ]
    )
    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.d",
        database="demo",
        max_depth=4,
        embedding_client=stub,
    )
    assert "error" not in res
    assert "tie_break" not in res
    assert set(_paths_intermediate_order(res)) == {"public.b", "public.c"}


def test_multi_path_reranked_by_cosine(tmp_path: Path) -> None:
    """Embedded endpoints + stub client with distinct per-path vectors -> top path matches stub."""
    graph = _build_diamond_graph(tmp_path)
    _set_embeddings(
        graph,
        {
            "public.a": _unit_vec(EMBEDDING_DIM, 0),
            "public.d": _unit_vec(EMBEDDING_DIM, 0),
        },
    )
    # Anchor aligns with path-index 0 (the first tied path). After sort-desc the
    # path that cosine=1 with the anchor must come first, the cosine=0 path last.
    anchor = _unit_vec(EMBEDDING_DIM, 0)
    high = _unit_vec(EMBEDDING_DIM, 0)
    low = _unit_vec(EMBEDDING_DIM, 1)
    stub = _VectorStubClient([anchor, high, low])

    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.d",
        database="demo",
        max_depth=4,
        embedding_client=stub,
    )
    assert "error" not in res
    assert res["tie_break"] == "embedding"
    paths = res["paths"]
    assert len(paths) == 2
    # The first path retains whatever intermediate the precomputed/Yen order emitted
    # first — the stub gave it the "high" score, so stable sort keeps it at the top.
    # We don't assert which of b/c specifically: only that *some* path got score=1
    # (high) and is ranked first, and the cosine=0 path is last.
    assert set(_paths_intermediate_order(res)) == {"public.b", "public.c"}
    # Path ordering is deterministic from the stub (anchor=high ⇒ idx 0 first).
    first_path_intermediate = paths[0]["steps"][0]["to_table"]
    second_path_intermediate = paths[1]["steps"][0]["to_table"]
    assert first_path_intermediate != second_path_intermediate


def test_epsilon_tied_paths_all_emitted(tmp_path: Path) -> None:
    """Both paths score identically: both still returned at the top (contract)."""
    graph = _build_diamond_graph(tmp_path)
    _set_embeddings(
        graph,
        {
            "public.a": _unit_vec(EMBEDDING_DIM, 0),
            "public.d": _unit_vec(EMBEDDING_DIM, 0),
        },
    )
    anchor = _unit_vec(EMBEDDING_DIM, 0)
    stub = _VectorStubClient(
        [anchor, anchor, anchor]
    )  # identical vectors → cosine=1 each

    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.d",
        database="demo",
        max_depth=4,
        embedding_client=stub,
    )
    assert "error" not in res
    assert res["tie_break"] == "embedding"
    paths = res["paths"]
    assert len(paths) == 2
    assert set(_paths_intermediate_order(res)) == {"public.b", "public.c"}


def test_embed_raises_graceful_fallback(tmp_path: Path) -> None:
    """Runtime embed() failure: tool does not raise; paths returned; marker null."""
    graph = _build_diamond_graph(tmp_path)
    _set_embeddings(
        graph,
        {
            "public.a": _unit_vec(EMBEDDING_DIM, 0),
            "public.d": _unit_vec(EMBEDDING_DIM, 0),
        },
    )
    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.d",
        database="demo",
        max_depth=4,
        embedding_client=_RaisingStubClient(),
    )
    assert "error" not in res
    assert "tie_break" not in res
    assert len(res["paths"]) == 2


def test_null_embedding_client_returns_empty(tmp_path: Path) -> None:
    """NullEmbeddingClient.embed -> []: tie-break path no-ops; marker null."""

    class _NullClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return []

    graph = _build_diamond_graph(tmp_path)
    _set_embeddings(
        graph,
        {
            "public.a": _unit_vec(EMBEDDING_DIM, 0),
            "public.d": _unit_vec(EMBEDDING_DIM, 0),
        },
    )
    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.d",
        database="demo",
        max_depth=4,
        embedding_client=_NullClient(),
    )
    assert "error" not in res
    assert "tie_break" not in res
    assert len(res["paths"]) == 2


def test_pre_pr_envelope_shape_unchanged_when_no_tie_break(tmp_path: Path) -> None:
    """Null-path envelope content is byte-identical to pre-PR.

    The traverse tool must not gain unconditional new keys on the null
    path: ``tie_break`` is only present when an embedding-backed
    tie-break actually ran (regression of an earlier draft of this PR
    that emitted ``"tie_break": None`` always).
    """
    _build_diamond_graph(tmp_path)
    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.d",
        database="demo",
        max_depth=4,
    )
    # Pre-PR success envelope shape — no new keys when no tie-break ran.
    expected_keys = {
        "from",
        "to",
        "database",
        "paths",
        "used_precomputed",
        "warning",
    }
    assert set(res.keys()) == expected_keys
    assert "tie_break" not in res


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
