"""Tests for the ``include_similar`` arg on the ``context`` MCP tool."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.intelligence.embeddings import EMBEDDING_DIM
from pretensor.mcp.service_context import reset_server_context
from pretensor.mcp.tools.context import context_payload
from pretensor.visibility.config import VisibilityConfig
from pretensor.visibility.filter import VisibilityFilter


@pytest.fixture(autouse=True)
def _clear_server_context() -> Generator[None, None, None]:
    reset_server_context()
    yield
    reset_server_context()


def _unit_vector(axis: int) -> list[float]:
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
        by_table = {str(r[1]): str(r[0]) for r in rows}
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


def _build_fixture(
    tmp_path: Path,
    *,
    connection_name: str = "demo",
    embed_alpha: bool = True,
) -> tuple[Path, dict[str, str]]:
    """Four tables across two clusters: {alpha, beta}=c1, {gamma, delta}=c2.

    Embeddings: alpha→axis 0, beta→axis 1, gamma→axis 2, delta→axis 3.
    When ``embed_alpha=False``, alpha is left with a NULL embedding so the
    ``no_embedding`` path can be exercised.
    """
    tables = [
        _build_table("alpha", ["id", "name"]),
        _build_table("beta", ["id", "price"]),
        _build_table("gamma", ["id", "label"]),
        _build_table("delta", ["id", "color"]),
    ]
    graph, by_table = _write_store(tmp_path, connection_name, tables)
    _register(tmp_path, connection_name, graph)

    embeddings: dict[str, list[float]] = {
        by_table["beta"]: _unit_vector(1),
        by_table["gamma"]: _unit_vector(2),
        by_table["delta"]: _unit_vector(3),
    }
    if embed_alpha:
        embeddings[by_table["alpha"]] = _unit_vector(0)
    _set_embeddings(graph, embeddings)

    _add_cluster(
        graph,
        cluster_id="demo::c1",
        database_key=connection_name,
        table_node_ids=[by_table["alpha"], by_table["beta"]],
    )
    _add_cluster(
        graph,
        cluster_id="demo::c2",
        database_key=connection_name,
        table_node_ids=[by_table["gamma"], by_table["delta"]],
    )
    return graph, by_table


# --- Tests --------------------------------------------------------------


def test_default_off_is_byte_identical(tmp_path: Path) -> None:
    """``include_similar=False`` (explicit or default) → no similar_* keys
    and responses are equal."""
    _build_fixture(tmp_path)

    out_default = context_payload(tmp_path, table="public.alpha")
    out_explicit = context_payload(
        tmp_path, table="public.alpha", include_similar=False
    )

    assert "similar_tables" not in out_default
    assert "similar_reason" not in out_default
    assert "similar_tables" not in out_explicit
    assert "similar_reason" not in out_explicit
    assert out_default == out_explicit


def test_embedded_target_cross_cluster_top_k(tmp_path: Path) -> None:
    """``include_similar=True`` returns cross-cluster neighbors, sorted desc,
    and excludes the target and same-cluster tables."""
    _build_fixture(tmp_path)

    out = context_payload(
        tmp_path, table="public.alpha", include_similar=True, similar_k=5
    )

    assert "error" not in out
    assert "similar_reason" not in out
    hits = out["similar_tables"]
    names = {h["qualified_name"] for h in hits}
    # Neither alpha (self) nor beta (same cluster c1) should appear.
    assert names == {"public.gamma", "public.delta"}
    for h in hits:
        assert h["cluster_id"] == "demo::c2"
    # Sorted descending by score.
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_unembedded_target_returns_no_embedding_reason(tmp_path: Path) -> None:
    """Target with NULL embedding → empty similar_tables + reason hint."""
    _build_fixture(tmp_path, embed_alpha=False)

    out = context_payload(
        tmp_path, table="public.alpha", include_similar=True, similar_k=5
    )

    assert out["similar_tables"] == []
    assert out["similar_reason"] == "no_embedding"


def test_visibility_filter_drops_hidden_neighbors(tmp_path: Path) -> None:
    """Hidden tables never appear in similar_tables even when they rank."""
    _build_fixture(tmp_path)
    vf = VisibilityFilter.from_config(VisibilityConfig(hidden_tables=["public.gamma"]))

    out = context_payload(
        tmp_path,
        table="public.alpha",
        include_similar=True,
        similar_k=5,
        visibility_filter=vf,
    )

    names = {h["qualified_name"] for h in out["similar_tables"]}
    assert "public.gamma" not in names
    assert "public.delta" in names


def test_similar_k_caps_result_count(tmp_path: Path) -> None:
    """``similar_k=1`` yields at most one neighbor."""
    _build_fixture(tmp_path)

    out = context_payload(
        tmp_path, table="public.alpha", include_similar=True, similar_k=1
    )

    assert len(out["similar_tables"]) == 1


def test_summary_detail_suppresses_similar_tables(tmp_path: Path) -> None:
    """``detail='summary'`` drops the ``similar_tables`` block silently.

    The summary detail level is the smallest envelope and intentionally
    omits the embedding-derived neighbor list — including it would
    bloat what's meant to be a compact preview.  Locks the contract so
    a future refactor can't accidentally re-add it.
    """
    _build_fixture(tmp_path)

    out = context_payload(
        tmp_path,
        table="public.alpha",
        include_similar=True,
        similar_k=5,
        detail="summary",
    )

    assert "similar_tables" not in out, (
        f"detail='summary' must omit the similar_tables block; got {out!r}"
    )


def test_multi_cluster_target_filters_all_shared_clusters(tmp_path: Path) -> None:
    """When the target belongs to multiple clusters, candidates that share
    ANY of the target's clusters are filtered — including via a
    non-first-discovered cluster membership.

    Regression guard for the bug where ``score_embedding_rows`` dedupes
    its output to one row per node_id (preserving only the first-seen
    cluster), which had broken the cross-cluster filter for tables with
    >1 cluster membership.
    """
    graph, by_table = _build_fixture(tmp_path)
    # Add ``alpha`` to a SECOND cluster ("demo::c2") that already
    # contains gamma + delta. After this, alpha is in both c1 and c2.
    # The cross-cluster filter must then exclude gamma/delta from
    # alpha's similar_tables result (they share c2 with alpha).
    _add_cluster(
        graph,
        cluster_id="demo::c2",
        database_key="demo",
        table_node_ids=[by_table["alpha"]],
    )

    out = context_payload(
        tmp_path, table="public.alpha", include_similar=True, similar_k=5
    )

    similar_ids = {hit["table_id"] for hit in out["similar_tables"]}
    assert by_table["gamma"] not in similar_ids, (
        "gamma shares cluster c2 with multi-cluster alpha; must be filtered"
    )
    assert by_table["delta"] not in similar_ids, (
        "delta shares cluster c2 with multi-cluster alpha; must be filtered"
    )


def test_unclustered_target_returns_top_k_neighbors(tmp_path: Path) -> None:
    """When the target has no cluster memberships, the cross-cluster filter
    has nothing to subtract — return the raw top-K cosine neighbors.

    Pinned because the alternative ("return empty when target is
    unclustered") would be useless for any agent calling
    ``context(include_similar=True)`` against a table that hasn't yet
    been picked up by clustering.  The current 'best effort' answer is
    documented in ``similar_tables_for_table``'s docstring.
    """
    graph, by_table = _build_fixture(tmp_path)

    # Strip alpha out of every cluster so the target becomes unclustered.
    store = KuzuStore(graph)
    try:
        store.execute(
            "MATCH (t:SchemaTable {node_id: $tid})-[r:IN_CLUSTER]->(c:Cluster) "
            "DELETE r",
            {"tid": by_table["alpha"]},
        )
    finally:
        store.close()

    out = context_payload(
        tmp_path, table="public.alpha", include_similar=True, similar_k=3
    )

    similar = out["similar_tables"]
    # Best-effort: return up to similar_k visible embedded tables.
    assert len(similar) > 0, (
        f"expected similar_tables to fall back to raw top-K when target is "
        f"unclustered; got {out!r}"
    )
    # The target itself is excluded.
    similar_ids = {hit["table_id"] for hit in similar}
    assert by_table["alpha"] not in similar_ids
