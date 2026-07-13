"""Tests for the ``semantic_search`` MCP tool."""

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
from pretensor.mcp.tools.semantic_search import semantic_search_payload
from pretensor.visibility.config import VisibilityConfig
from pretensor.visibility.filter import VisibilityFilter


@pytest.fixture(autouse=True)
def _clear_server_context() -> Generator[None, None, None]:
    """Reset the module-level server context before and after each test."""
    reset_server_context()
    yield
    reset_server_context()


def _unit_vector(axis: int) -> list[float]:
    """Return a unit vector pointing along ``axis`` in 384-dim space."""
    vec = [0.0] * EMBEDDING_DIM
    vec[axis] = 1.0
    return vec


def _build_table(name: str, columns: list[str]) -> Table:
    return Table(
        name=name,
        schema_name="public",
        columns=[Column(name=c, data_type="text") for c in columns],
        foreign_keys=[],
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
) -> tuple[Path, list[str]]:
    """Build a Kuzu graph for ``connection_name`` and return (path, ordered node_ids)."""
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
        node_ids = [str(r[0]) for r in rows]
    finally:
        store.close()
    return graph, node_ids


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


def _add_cluster(
    graph: Path,
    *,
    cluster_id: str,
    database_key: str,
    table_node_ids: list[str],
) -> None:
    store = KuzuStore(graph)
    try:
        store.upsert_cluster(
            node_id=cluster_id,
            database_key=database_key,
            label=cluster_id,
            description="test cluster",
            cohesion_score=0.9,
            table_count=len(table_node_ids),
        )
        for tid in table_node_ids:
            store.upsert_in_cluster(tid, cluster_id)
    finally:
        store.close()


def _build_single_db(
    tmp_path: Path,
    *,
    connection_name: str = "demo",
    with_embeddings: bool = True,
) -> dict[str, str]:
    """Build one DB with four tables (`alpha`, `beta`, `gamma`, `delta`).

    When ``with_embeddings`` is True, each table gets a unit vector along a
    distinct axis (alpha→0, beta→1, gamma→2, delta→3) so cosine ordering is
    deterministic. Returns a mapping of table_name → node_id.
    """
    tables = [
        _build_table("alpha", ["id", "name"]),
        _build_table("beta", ["id", "price"]),
        _build_table("gamma", ["id", "label"]),
        _build_table("delta", ["id", "color"]),
    ]
    graph, node_ids = _write_store(tmp_path, connection_name, tables)
    _register(tmp_path, connection_name, graph)

    # node_ids came back sorted by table_name: alpha, beta, delta, gamma
    by_table = {
        "alpha": node_ids[0],
        "beta": node_ids[1],
        "delta": node_ids[2],
        "gamma": node_ids[3],
    }
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
    """Build two databases (dbA, dbB) each with two tables, each with embeddings.

    Layout:
      dbA: a1 → axis 0 ; a2 → axis 1
      dbB: b1 → axis 0 ; b2 → axis 1

    Returns ``{"dbA": {...}, "dbB": {...}}`` mapping table_name → node_id.
    """
    out: dict[str, dict[str, str]] = {}
    for db in ("dbA", "dbB"):
        prefix = "a" if db == "dbA" else "b"
        tables = [
            _build_table(f"{prefix}1", ["id", "name"]),
            _build_table(f"{prefix}2", ["id", "value"]),
        ]
        graph, node_ids = _write_store(tmp_path, db, tables)
        _register(tmp_path, db, graph)
        by_table = {f"{prefix}1": node_ids[0], f"{prefix}2": node_ids[1]}
        _set_embeddings(
            graph,
            {
                by_table[f"{prefix}1"]: _unit_vector(0),
                by_table[f"{prefix}2"]: _unit_vector(1),
            },
        )
        out[db] = by_table
    return out


def _stub_embed(monkeypatch: pytest.MonkeyPatch, vector: list[float]) -> None:
    def _embed(self: LocalEmbeddingClient, texts: list[str]) -> list[list[float]]:
        return [vector for _ in texts]

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _embed)


# --- Tests --------------------------------------------------------------


def test_happy_path_top_k_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Query vector aligned with alpha's axis → alpha ranks first."""
    by_table = _build_single_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(0))

    out = semantic_search_payload(tmp_path, query="looking for alpha", k=10)

    assert out.get("mode") == "semantic", out
    assert "error" not in out
    results = out["results"]
    assert len(results) == 4
    assert results[0]["name"] == "public.alpha"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert results[0]["node_type"] == "SchemaTable"
    assert results[0]["connection_name"] == "demo"
    # Remaining three are orthogonal to the query → score 0.0 each.
    for hit in results[1:]:
        assert hit["score"] == pytest.approx(0.0, abs=1e-6)
    assert by_table  # fixture sanity


def test_k_caps_result_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _build_single_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(0))

    out = semantic_search_payload(tmp_path, query="x", k=2)

    assert out.get("mode") == "semantic"
    assert len(out["results"]) == 2


def test_empty_vectors_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No table carries a vector → fallback_bm25 envelope."""
    _build_single_db(tmp_path, with_embeddings=False)
    _stub_embed(monkeypatch, _unit_vector(0))

    out = semantic_search_payload(tmp_path, query="x", k=10)

    assert out.get("mode") == "fallback_bm25"
    assert out["results"] == []
    assert "pretensor[embeddings]" in out["hint"]
    # Fallback echoes the inputs for caller correlation.
    assert out.get("query") == "x"


def test_missing_extra_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing ``[embeddings]`` extra → fallback envelope + single WARNING."""
    _build_single_db(tmp_path)

    def _missing_deps(
        self: LocalEmbeddingClient, texts: list[str]
    ) -> list[list[float]]:
        raise ImportError(
            "Install embedding dependencies with: pip install 'pretensor[embeddings]'"
        )

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _missing_deps)

    with caplog.at_level(logging.WARNING, logger="pretensor.mcp.tools.semantic_search"):
        out = semantic_search_payload(tmp_path, query="x", k=5)

    assert out.get("mode") == "fallback_bm25"
    assert out["results"] == []
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name == "pretensor.mcp.tools.semantic_search"
    ]
    assert len(warnings) == 1
    assert "pretensor[embeddings]" in warnings[0].getMessage()


def test_visibility_filter_hides_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hidden tables never appear even when they score highest."""
    _build_single_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(0))
    vf = VisibilityFilter.from_config(VisibilityConfig(hidden_tables=["public.alpha"]))

    out = semantic_search_payload(tmp_path, query="x", k=10, visibility_filter=vf)

    assert out.get("mode") == "semantic"
    names = {r["name"] for r in out["results"]}
    assert "public.alpha" not in names
    assert len(out["results"]) == 3


def test_visibility_filter_hides_every_embedded_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When visibility hides ALL embedded tables, the envelope stays in
    ``semantic`` mode with an empty result list — distinct from the
    ``fallback_bm25`` envelope reserved for "no tables carry vectors."

    Locks down the contract: hidden-vs-unembedded is a meaningful
    distinction. BM25 (``query``) also respects visibility, so falling
    back to it wouldn't return any more rows; the empty semantic
    envelope is the honest answer.
    """
    _build_single_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(0))
    # Hide every table in the fixture.
    vf = VisibilityFilter.from_config(
        VisibilityConfig(
            hidden_tables=[
                "public.alpha",
                "public.beta",
                "public.gamma",
                "public.delta",
            ]
        )
    )

    out = semantic_search_payload(tmp_path, query="x", k=10, visibility_filter=vf)

    assert out.get("mode") == "semantic", (
        f"hidden-but-vectors-exist must keep semantic mode, got {out!r}"
    )
    assert out["results"] == []


def test_database_filter_narrows_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``database='dbA'`` returns only dbA hits."""
    _build_two_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(0))

    out = semantic_search_payload(tmp_path, query="x", k=10, database="dbA")

    assert out.get("mode") == "semantic"
    cns = {r["connection_name"] for r in out["results"]}
    assert cns == {"dbA"}
    assert len(out["results"]) == 2


def test_cluster_filter_narrows_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cluster='demo::c1'`` returns only tables in that cluster."""
    by_table = _build_single_db(tmp_path)
    graph = tmp_path / "graphs" / "demo.kuzu"
    _add_cluster(
        graph,
        cluster_id="demo::c1",
        database_key="demo",
        table_node_ids=[by_table["alpha"], by_table["beta"]],
    )
    _add_cluster(
        graph,
        cluster_id="demo::c2",
        database_key="demo",
        table_node_ids=[by_table["gamma"], by_table["delta"]],
    )
    _stub_embed(monkeypatch, _unit_vector(0))

    out = semantic_search_payload(tmp_path, query="x", k=10, cluster="demo::c1")

    assert out.get("mode") == "semantic"
    names = {r["name"] for r in out["results"]}
    assert names == {"public.alpha", "public.beta"}
    cluster_ids = {r.get("cluster_id") for r in out["results"]}
    assert cluster_ids == {"demo::c1"}


def test_database_and_cluster_and(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Combined ``database`` + ``cluster`` filters AND together."""
    tables_by_db = _build_two_db(tmp_path)
    # Cluster c-shared exists in both dbA and dbB with different members.
    _add_cluster(
        tmp_path / "graphs" / "dbA.kuzu",
        cluster_id="shared",
        database_key="dbA",
        table_node_ids=[tables_by_db["dbA"]["a1"]],
    )
    _add_cluster(
        tmp_path / "graphs" / "dbB.kuzu",
        cluster_id="shared",
        database_key="dbB",
        table_node_ids=[tables_by_db["dbB"]["b1"]],
    )
    _stub_embed(monkeypatch, _unit_vector(0))

    out = semantic_search_payload(
        tmp_path, query="x", k=10, database="dbA", cluster="shared"
    )

    assert out.get("mode") == "semantic"
    names = {r["name"] for r in out["results"]}
    assert names == {"public.a1"}


def test_unknown_database_returns_error_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``database='nonexistent'`` returns a plain error envelope — NOT the
    ``fallback_bm25`` shape.

    The fallback envelope is reserved for "embeddings unavailable"; a
    caller mistake (unresolvable database name) must surface as an
    ``error`` so MCP clients correct the argument instead of silently
    routing to BM25, matching the contract of sibling tools.
    """
    _build_single_db(tmp_path)
    _stub_embed(monkeypatch, _unit_vector(0))

    out = semantic_search_payload(tmp_path, query="customer", k=5, database="nope")

    assert "mode" not in out, (
        f"unknown-database path must not use the fallback envelope, got {out!r}"
    )
    assert "error" in out and "nope" in str(out["error"])
    assert out["database"] == "nope"
    assert "hint" in out
