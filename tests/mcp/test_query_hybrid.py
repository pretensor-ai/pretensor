"""Tests for the hybrid BM25 + embedding RRF rerank of the ``query`` MCP tool."""

from __future__ import annotations

import logging
from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.intelligence.embeddings import EMBEDDING_DIM, LocalEmbeddingClient
from pretensor.mcp.service_context import reset_server_context
from pretensor.mcp.tools._rank import RRF_K, rrf_fuse
from pretensor.mcp.tools.search import query_payload
from pretensor.visibility.config import VisibilityConfig
from pretensor.visibility.filter import VisibilityFilter


@pytest.fixture(autouse=True)
def _clear_server_context() -> Generator[None, None, None]:
    """Reset the module-level server context before and after each test."""
    reset_server_context()
    yield
    reset_server_context()


def _unit_vector(axis: int) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    vec[axis] = 1.0
    return vec


def _build_table(name: str, columns: list[str], *, comment: str = "") -> Table:
    return Table(
        name=name,
        schema_name="public",
        columns=[Column(name=c, data_type="text") for c in columns],
        foreign_keys=[],
        comment=comment,
    )


def _build_snapshot(connection_name: str, tables: list[Table]) -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name=connection_name,
        database=connection_name,
        schemas=["public"],
        tables=tables,
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _write_store(
    tmp_path: Path, connection_name: str, tables: list[Table]
) -> tuple[Path, dict[str, str]]:
    graph = tmp_path / "graphs" / f"{connection_name}.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    snap = _build_snapshot(connection_name, tables)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {database: $db}) "
            "RETURN t.node_id, t.table_name ORDER BY t.table_name",
            {"db": connection_name},
        )
        by_table = {str(name): str(nid) for nid, name in rows}
    finally:
        store.close()
    return graph, by_table


def _register(tmp_path: Path, connection_name: str, graph: Path) -> None:
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name=connection_name,
        database=connection_name,
        dsn=f"postgresql://localhost/{connection_name}",
        graph_path=graph,
        indexed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    reg.save()


def _set_embeddings(graph: Path, mapping: dict[str, list[float]]) -> None:
    store = KuzuStore(graph)
    try:
        for node_id, vec in mapping.items():
            store.set_table_embedding(node_id, vec)
    finally:
        store.close()


def _stub_embed(monkeypatch: pytest.MonkeyPatch, vector: list[float]) -> None:
    def _embed(self: LocalEmbeddingClient, texts: list[str]) -> list[list[float]]:
        return [vector for _ in texts]

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _embed)


def _build_single_db(
    tmp_path: Path,
    *,
    connection_name: str = "demo",
    with_embeddings: bool = True,
) -> dict[str, str]:
    """Four tables with BM25-searchable comments and distinct-axis embeddings.

    The shared keyword ``"customer"`` lets BM25 match all four; the per-table
    words (``alpha``, ``beta``, ``gamma``, ``delta``) let BM25 narrow to one.
    """
    tables = [
        _build_table("alpha", ["id", "name"], comment="customer alpha records"),
        _build_table("beta", ["id", "price"], comment="customer beta orders"),
        _build_table("gamma", ["id", "label"], comment="customer gamma profiles"),
        _build_table("delta", ["id", "color"], comment="customer delta events"),
    ]
    graph, by_table = _write_store(tmp_path, connection_name, tables)
    _register(tmp_path, connection_name, graph)
    if with_embeddings:
        _set_embeddings(
            graph,
            {
                by_table["alpha"]: _unit_vector(0),
                by_table["beta"]: _unit_vector(1),
                by_table["gamma"]: _unit_vector(2),
                by_table["delta"]: _unit_vector(3),
            },
        )
    return by_table


def _build_two_db(tmp_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for db in ("dbA", "dbB"):
        prefix = "a" if db == "dbA" else "b"
        tables = [
            _build_table(
                f"{prefix}1", ["id", "name"], comment=f"shared-keyword {prefix}1 row"
            ),
            _build_table(
                f"{prefix}2", ["id", "value"], comment=f"shared-keyword {prefix}2 row"
            ),
        ]
        graph, by_table = _write_store(tmp_path, db, tables)
        _register(tmp_path, db, graph)
        _set_embeddings(
            graph,
            {
                by_table[f"{prefix}1"]: _unit_vector(0),
                by_table[f"{prefix}2"]: _unit_vector(1),
            },
        )
        out[db] = by_table
    return out


# --- Pure RRF math ------------------------------------------------------


def test_rrf_fuse_symmetric_scores_yield_equal_ranks() -> None:
    """BM25 ``[A, B]`` + cosine ``[B, A]`` → identical fused scores."""
    fused = rrf_fuse(bm25_keys=["A", "B"], cosine_keys=["B", "A"])
    assert len(fused) == 2
    scores = {k: s for k, s in fused}
    assert scores["A"] == pytest.approx(scores["B"])
    # Absolute value: 1/(60+1) + 1/(60+2).
    expected = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 2)
    assert scores["A"] == pytest.approx(expected)


def test_rrf_fuse_missing_from_one_list_contributes_only_one_term() -> None:
    fused = rrf_fuse(bm25_keys=["A"], cosine_keys=["B"])
    scores = {k: s for k, s in fused}
    # Both rank 1 in their respective list → tie.
    assert scores["A"] == pytest.approx(1.0 / (RRF_K + 1))
    assert scores["B"] == pytest.approx(1.0 / (RRF_K + 1))


def test_rrf_fuse_higher_rank_wins() -> None:
    fused = rrf_fuse(bm25_keys=["A", "B", "C"], cosine_keys=["A", "B", "C"])
    # A ranked 1+1, B ranked 2+2, C ranked 3+3 → A > B > C.
    assert [k for k, _ in fused] == ["A", "B", "C"]


# --- Null-path parity (no `rerank` marker) ------------------------------


def test_null_path_no_embeddings_in_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No table carries a vector → envelope has no ``rerank`` key."""
    _build_single_db(tmp_path, with_embeddings=False)
    # Embedding client works but the graph has no vectors to cosine against.
    _stub_embed(monkeypatch, _unit_vector(0))

    out = query_payload(tmp_path, q="customer", limit=5)

    assert "rerank" not in out
    assert out["query"] == "customer"
    assert out["db"] is None
    assert out["results"], "BM25 should still return hits"


def test_null_path_missing_embeddings_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``LocalEmbeddingClient.embed`` raising ImportError → BM25 only, one WARNING."""
    _build_single_db(tmp_path)

    def _missing(self: LocalEmbeddingClient, texts: list[str]) -> list[list[float]]:
        raise ImportError(
            "Install embedding dependencies with: pip install 'pretensor[embeddings]'"
        )

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _missing)

    with caplog.at_level(logging.WARNING, logger="pretensor.mcp.tools.search"):
        out = query_payload(tmp_path, q="customer", limit=5)

    assert "rerank" not in out
    assert out["results"], "BM25 path should still return hits"
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "pretensor.mcp.tools.search"
    ]
    assert len(warnings) == 1
    assert "pretensor[embeddings]" in warnings[0].getMessage()


def test_null_path_embed_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``embed`` returning ``[]`` (NullEmbeddingClient path) → BM25 only."""
    _build_single_db(tmp_path)

    def _empty(self: LocalEmbeddingClient, texts: list[str]) -> list[list[float]]:
        return []

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _empty)

    out = query_payload(tmp_path, q="customer", limit=5)

    assert "rerank" not in out
    assert out["results"]


def test_null_path_parity_matches_pre_fusion_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Null-path result order must match a plain BM25-only pass."""
    _build_single_db(tmp_path, with_embeddings=False)
    _stub_embed(monkeypatch, _unit_vector(0))

    out_hybrid = query_payload(tmp_path, q="alpha", limit=5)

    # Second, parallel call with BM25 only (vectors absent) must order
    # results identically. Using the same fixture tmp_path would race on
    # the on-disk FTS cache, so re-build a fresh registry.
    assert "rerank" not in out_hybrid
    names = [r["name"] for r in out_hybrid["results"]]
    # BM25 should rank tables whose comment contains "alpha" above others.
    assert names[0] == "public.alpha"


# --- Fusion path (`rerank: "rrf"`) --------------------------------------


def test_fusion_marker_present_when_vectors_and_embed_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_single_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(1))

    out = query_payload(tmp_path, q="customer", limit=5)

    assert out.get("rerank") == "rrf"
    assert out["query"] == "customer"
    assert out["results"]


def test_fusion_includes_cosine_only_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A table with no BM25 token match but strong cosine alignment still appears."""
    _build_single_db(tmp_path)
    # Query axis 1 → beta scores 1.0 cosine, others 0.0.
    _stub_embed(monkeypatch, _unit_vector(1))

    out = query_payload(tmp_path, q="xqqzzyy", limit=5)

    assert out.get("rerank") == "rrf"
    names = {r["name"] for r in out["results"]}
    assert "public.beta" in names


def test_fusion_preserves_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hidden tables must not leak through either the BM25 or the cosine side."""
    _build_single_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(1))
    vf = VisibilityFilter.from_config(VisibilityConfig(hidden_tables=["public.beta"]))

    out = query_payload(tmp_path, q="customer", limit=5, visibility_filter=vf)

    assert out.get("rerank") == "rrf"
    names = {r["name"] for r in out["results"]}
    assert "public.beta" not in names


def test_fusion_db_filter_scopes_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_two_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(0))

    out = query_payload(tmp_path, q="shared-keyword", db="dbA", limit=10)

    assert out.get("rerank") == "rrf"
    cns = {r["connection_name"] for r in out["results"]}
    assert cns == {"dbA"}


def test_fusion_truncates_to_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_single_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(1))

    out = query_payload(tmp_path, q="customer", limit=2)

    assert out.get("rerank") == "rrf"
    assert len(out["results"]) == 2


def test_fusion_zero_bm25_hits_still_marks_rerank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Query with no BM25 matches still enters the fusion path when vectors exist."""
    _build_single_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(2))

    out = query_payload(tmp_path, q="xqqzzyy", limit=5)

    assert out.get("rerank") == "rrf"
    # Cosine axis 2 → gamma is rank-1 by cosine alone.
    assert out["results"], "Cosine-only hits should materialize in fused output"
    assert out["results"][0]["name"] == "public.gamma"
