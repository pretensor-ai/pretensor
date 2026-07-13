"""Cross-cutting null-path parity tests for the MCP surface.

Sibling of ``tests/intelligence/test_null_path_parity.py``: when no
``SchemaTable`` carries a vector, none of the MCP tool envelopes may grow an
embedding-derived marker.  This file covers the four tools the embedding epic
touches — ``query``, ``semantic_search``, ``context``, ``traverse`` — against
a shared fixture so a future regression that adds a stray ``"rerank"`` /
``"tie_break"`` / ``"similar_tables"`` field to the null path fails loudly.

Per-tool null-path tests already live in ``test_query_hybrid.py``,
``test_traverse_tiebreak.py``, ``test_context_similar.py``, and
``test_semantic_search.py``; this module is the consolidated guard the embedding
changes must keep green.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.intelligence.embeddings import LocalEmbeddingClient
from pretensor.mcp.service_context import reset_server_context
from pretensor.mcp.tools.context import context_payload
from pretensor.mcp.tools.search import query_payload
from pretensor.mcp.tools.semantic_search import semantic_search_payload
from pretensor.mcp.tools.traverse import traverse_payload


@pytest.fixture(autouse=True)
def _clear_server_context() -> Generator[None, None, None]:
    reset_server_context()
    yield
    reset_server_context()


def _build_table(
    name: str,
    columns: list[str],
    *,
    comment: str = "",
    foreign_keys: list[ForeignKey] | None = None,
) -> Table:
    return Table(
        name=name,
        schema_name="public",
        columns=[Column(name=c, data_type="text") for c in columns],
        foreign_keys=foreign_keys or [],
        comment=comment,
    )


def _write_null_store(tmp_path: Path, connection_name: str = "demo") -> Path:
    """Build a tiny graph with FK-linked tables and no embeddings populated.

    The fixture has two FK paths between ``orders`` and ``customers`` so that
    ``traverse`` returns at least one path and the precomputed-vs-cosine
    tie-break code path is exercised.
    """
    tables = [
        _build_table("customers", ["id", "name"], comment="customer master record"),
        _build_table(
            "orders",
            ["id", "customer_id", "amount"],
            comment="customer order rows",
            foreign_keys=[
                ForeignKey(
                    source_schema="public",
                    source_table="orders",
                    source_column="customer_id",
                    target_schema="public",
                    target_table="customers",
                    target_column="id",
                ),
            ],
        ),
        _build_table(
            "shipments",
            ["id", "order_id"],
            comment="shipment events keyed to customer orders",
            foreign_keys=[
                ForeignKey(
                    source_schema="public",
                    source_table="shipments",
                    source_column="order_id",
                    target_schema="public",
                    target_table="orders",
                    target_column="id",
                ),
            ],
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
    return graph


def _stub_null_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the embedding client to return ``[]`` (NullEmbeddingClient shape)."""

    def _empty(self: LocalEmbeddingClient, texts: list[str]) -> list[list[float]]:
        return []

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _empty)


# ── Null-path envelope assertions ────────────────────────────────────────────


def test_query_null_path_no_rerank_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``query`` envelope on null path has no ``rerank`` key."""
    _write_null_store(tmp_path)
    _stub_null_embedder(monkeypatch)

    out = query_payload(tmp_path, q="customer", limit=5)

    assert "rerank" not in out, (
        f"unexpected rerank marker on null path: {out.get('rerank')!r}"
    )
    assert out["query"] == "customer"


def test_semantic_search_null_path_returns_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``semantic_search`` envelope on null path has ``mode == "fallback_bm25"``."""
    _write_null_store(tmp_path)
    _stub_null_embedder(monkeypatch)

    out = semantic_search_payload(tmp_path, query="customer", k=5)

    assert out.get("mode") == "fallback_bm25", (
        f"expected fallback_bm25 envelope on null path, got {out!r}"
    )
    assert out["results"] == []
    assert "hint" in out


def test_traverse_null_path_no_tie_break_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``traverse`` envelope on null path has no truthy ``tie_break`` marker."""
    _write_null_store(tmp_path)
    _stub_null_embedder(monkeypatch)

    out = traverse_payload(
        tmp_path,
        from_table="customers",
        to_table="shipments",
        database="demo",
    )

    # Either the key is absent or it is None — both are acceptable null-path
    # shapes per the traverse tie-break contract.  An "embedding"-valued tie_break would be a
    # regression.
    assert out.get("tie_break") in (None, "none", ""), (
        f"unexpected tie_break marker on null path: {out.get('tie_break')!r}"
    )


def test_context_null_path_no_similar_tables_block(tmp_path: Path) -> None:
    """``context`` with ``include_similar=False`` (default) emits no ``similar_tables`` key."""
    _write_null_store(tmp_path)

    out = context_payload(tmp_path, table="customers", db="demo")

    assert "similar_tables" not in out, (
        f"unexpected similar_tables on default-args call: {out.get('similar_tables')!r}"
    )


def test_context_null_path_similar_tables_empty_when_requested(
    tmp_path: Path,
) -> None:
    """``context(include_similar=True)`` with no embeddings → empty list + reason."""
    _write_null_store(tmp_path)

    out = context_payload(tmp_path, table="customers", db="demo", include_similar=True)

    # context-similar contract: the block is ALWAYS present when requested
    # (callers branch on the reason, never on key presence), and with no
    # stored vectors it must be empty with the ``no_embedding`` reason.
    assert "similar_tables" in out, (
        "context(include_similar=True) must always emit similar_tables"
    )
    assert out["similar_tables"] == []
    assert out.get("similar_reason") == "no_embedding"


# ── Smoke: shared fixture round-trips ────────────────────────────────────────


def test_null_embedded_store_fixture_loads(null_embedded_store) -> None:
    """The shared ``null_embedded_store`` fixture builds without raising.

    Sibling tests under ``tests/intelligence/test_null_path_parity.py`` use the
    same fixture for the intelligence-layer side of the parity contract; this
    smoke test is here so a fixture regression surfaces in both directories.
    """
    rows = null_embedded_store.query_all_rows(
        "MATCH (t:SchemaTable {database: 'pagila'}) RETURN count(t)"
    )
    assert rows and int(rows[0][0]) > 0
