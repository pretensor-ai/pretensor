"""``PRETENSOR_EMBEDDINGS_DISABLED`` must gate every QUERY-TIME embedding path.

Sibling of ``tests/intelligence/test_embedding_index_step.py`` (which covers
the index-time step): the kill switch is documented as forcing the null path
even when the ``[embeddings]`` extra is installed, toggles are on, and the
store already carries vectors.  Each test here builds a store WITH vectors,
stubs ``LocalEmbeddingClient.embed`` to fail loudly if it is ever invoked,
sets the env var, and asserts the tool returns its null/disabled shape.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.intelligence.embeddings import (
    EMBEDDING_DIM,
    LocalEmbeddingClient,
)
from pretensor.intelligence.scoring import ScorerRegistry
from pretensor.intelligence.semantic import extend_with_embedding_scorer
from pretensor.mcp.service_context import reset_server_context
from pretensor.mcp.tools.context import context_payload
from pretensor.mcp.tools.search import query_payload
from pretensor.mcp.tools.semantic_search import semantic_search_payload
from pretensor.mcp.tools.traverse import _embedding_tie_break
from pretensor.semantic.compiler import _suggest_did_you_mean

_ENV = "PRETENSOR_EMBEDDINGS_DISABLED"


@pytest.fixture(autouse=True)
def _clear_server_context() -> Generator[None, None, None]:
    reset_server_context()
    yield
    reset_server_context()


def _unit_vector(axis: int) -> list[float]:
    vec = [0.0] * EMBEDDING_DIM
    vec[axis] = 1.0
    return vec


def _forbid_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``embed`` to fail the test if any code path reaches the client."""

    def _explode(self: LocalEmbeddingClient, texts: list[str]) -> list[list[float]]:
        raise AssertionError(
            "LocalEmbeddingClient.embed was called despite "
            "PRETENSOR_EMBEDDINGS_DISABLED being set"
        )

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _explode)


def _build_embedded_store(tmp_path: Path, connection_name: str = "demo") -> None:
    """Two BM25-searchable tables, both carrying stored vectors."""
    tables = [
        Table(
            name="alpha",
            schema_name="public",
            columns=[Column(name="id", data_type="text")],
            foreign_keys=[],
            comment="customer alpha records",
        ),
        Table(
            name="beta",
            schema_name="public",
            columns=[Column(name="id", data_type="text")],
            foreign_keys=[],
            comment="customer beta orders",
        ),
    ]
    snap = SchemaSnapshot(
        connection_name=connection_name,
        database=connection_name,
        schemas=["public"],
        tables=tables,
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    graph = tmp_path / "graphs" / f"{connection_name}.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {database: $db}) RETURN t.node_id, t.table_name",
            {"db": connection_name},
        )
        for axis, (nid, _name) in enumerate(rows):
            store.set_table_embedding(str(nid), _unit_vector(axis))
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name=connection_name,
        database=connection_name,
        dsn=f"postgresql://localhost/{connection_name}",
        graph_path=graph,
        indexed_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    reg.save()


def test_query_kill_switch_skips_rerank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``query`` with vectors present + env set → BM25-only envelope, no embed call."""
    _build_embedded_store(tmp_path)
    _forbid_embed(monkeypatch)
    monkeypatch.setenv(_ENV, "1")

    out = query_payload(tmp_path, q="customer", limit=5)

    assert "rerank" not in out, (
        f"kill switch must suppress the hybrid rerank: {out.get('rerank')!r}"
    )
    assert out["results"], "BM25 path itself must still return hits"


def test_semantic_search_kill_switch_returns_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``semantic_search`` with vectors present + env set → fallback envelope."""
    _build_embedded_store(tmp_path)
    _forbid_embed(monkeypatch)
    monkeypatch.setenv(_ENV, "1")

    out = semantic_search_payload(tmp_path, query="customer", k=5)

    assert out.get("mode") == "fallback_bm25"
    assert out["results"] == []
    assert "PRETENSOR_EMBEDDINGS_DISABLED" in out.get("hint", "")


def test_context_kill_switch_reports_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``context(include_similar=True)`` with vectors + env set → ``disabled`` reason."""
    _build_embedded_store(tmp_path)
    _forbid_embed(monkeypatch)
    monkeypatch.setenv(_ENV, "1")

    out = context_payload(tmp_path, table="alpha", db="demo", include_similar=True)

    assert out.get("similar_tables") == []
    assert out.get("similar_reason") == "disabled"


def test_traverse_tie_break_kill_switch_returns_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tie-break helper bails before touching the store when env is set.

    Passing a sentinel object as the store proves the early return: any
    attribute access on it would raise ``AttributeError``.
    """
    monkeypatch.setenv(_ENV, "1")
    paths = [{"hops": 1}, {"hops": 2}]

    out_paths, applied = _embedding_tie_break(
        cast(KuzuStore, object()),
        db_key="demo",
        from_id="a",
        to_id="b",
        paths=paths,  # type: ignore[arg-type]
    )

    assert out_paths is paths
    assert applied is False


def test_compile_did_you_mean_kill_switch_uses_difflib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Suggestion ranking falls back to difflib without touching the client."""
    _forbid_embed(monkeypatch)
    monkeypatch.setenv(_ENV, "1")

    out = _suggest_did_you_mean("custmer", ["customer", "supplier", "invoice"])

    assert out == ["customer"]


def test_extend_with_embedding_scorer_kill_switch_returns_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relationship-scorer registry is returned unchanged when env is set."""
    monkeypatch.setenv(_ENV, "1")
    base = ScorerRegistry()

    graph = tmp_path / "scorer.kuzu"
    store = KuzuStore(graph)
    try:
        out = extend_with_embedding_scorer(
            base, store=store, database_key="demo", threshold=0.8
        )
    finally:
        store.close()

    assert out is base
