"""Tests for SQLite FTS5 metadata search."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.graph_models.entity import EntityNode
from pretensor.search.base import BaseSearchIndex
from pretensor.search.index import KeywordSearchIndex, SearchIndex


def _snapshot_customers() -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name="shop",
        database="shop",
        schemas=["public"],
        tables=[
            Table(
                name="customers",
                schema_name="public",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                comment="Customer accounts and contact info",
            ),
            Table(
                name="revenue_daily",
                schema_name="public",
                columns=[Column(name="day", data_type="date")],
                comment="Daily revenue totals by region",
            ),
        ],
        introspected_at=datetime.now(timezone.utc),
    )


def _snapshot_pre203_recall_cases() -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[
            Table(
                name="users",
                schema_name="public",
                columns=[
                    Column(name="id", data_type="int", is_primary_key=True),
                    Column(name="email", data_type="text"),
                ],
            ),
            Table(
                name="subscriptions",
                schema_name="public",
                columns=[
                    Column(name="subscription_id", data_type="int", is_primary_key=True),
                    Column(name="product_id", data_type="int"),
                ],
            ),
            Table(
                name="fact_sales",
                schema_name="public",
                columns=[
                    Column(name="sale_id", data_type="int", is_primary_key=True),
                    Column(name="amount", data_type="numeric"),
                ],
            ),
            Table(
                name="dim_customer",
                schema_name="public",
                columns=[
                    Column(name="customer_id", data_type="int", is_primary_key=True),
                    Column(name="region", data_type="text"),
                ],
            ),
            Table(
                name="dim_geography",
                schema_name="public",
                columns=[
                    Column(name="country", data_type="text"),
                    Column(name="city", data_type="text"),
                ],
            ),
            Table(
                name="noise_logs",
                schema_name="public",
                columns=[
                    Column(name="id", data_type="int", is_primary_key=True),
                    Column(name="payload", data_type="text"),
                ],
                comment="Debug payload archive",
            ),
        ],
        introspected_at=datetime.now(timezone.utc),
    )


def _seed_representative_entity_and_cluster_signals(graph: Path) -> None:
    store = KuzuStore(graph)
    try:
        store.upsert_entity(
            node=EntityNode(
                node_id="demo::entity::User Account",
                connection_name="demo",
                database="demo",
                name="User Account",
                description="Represents customer login identity records.",
            )
        )
        store.upsert_represents(
            entity_node_id="demo::entity::User Account",
            table_node_id=table_node_id("demo", "public", "users"),
        )
        store.upsert_cluster(
            node_id="demo::cluster::1",
            database_key="demo",
            label="Customer Analytics",
            description="Customer behavior and retention tables.",
            cohesion_score=0.9,
            table_count=3,
        )
        store.upsert_in_cluster(
            table_node_id=table_node_id("demo", "public", "dim_customer"),
            cluster_node_id="demo::cluster::1",
        )
        store.upsert_in_cluster(
            table_node_id=table_node_id("demo", "public", "dim_geography"),
            cluster_node_id="demo::cluster::1",
        )
    finally:
        store.close()


def _build_index_for_snapshot(tmp_path: Path, snapshot: SchemaSnapshot) -> SearchIndex:
    graph = tmp_path / "graphs" / f"{snapshot.connection_name}.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snapshot, store, run_relationship_discovery=False)
    finally:
        store.close()

    if snapshot.connection_name == "demo":
        _seed_representative_entity_and_cluster_signals(graph)

    reg_path = tmp_path / "registry.json"
    reg = GraphRegistry(reg_path).load()
    reg.upsert(
        connection_name=snapshot.connection_name,
        database=snapshot.database,
        dsn="postgresql://x",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()
    idx_path = SearchIndex.default_path(tmp_path)
    SearchIndex.build(reg.load(), idx_path)
    return SearchIndex(idx_path)


def test_search_customers_table_surfaces_first(tmp_path: Path) -> None:
    idx = _build_index_for_snapshot(tmp_path, _snapshot_customers())
    hits = idx.search("customer", limit=5)
    assert hits, "expected at least one hit"
    assert hits[0].node_type == "SchemaTable"
    assert "customers" in hits[0].name


def test_search_revenue_comment_surfaces_table(tmp_path: Path) -> None:
    hits = _build_index_for_snapshot(tmp_path, _snapshot_customers()).search(
        "revenue", limit=5
    )
    names = {h.name for h in hits}
    assert any("revenue_daily" in n for n in names)


def test_needs_rebuild_when_kuzu_newer(tmp_path: Path) -> None:
    import os
    import time

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="x",
        database="x",
        dsn="postgresql://x",
        graph_path=tmp_path / "g.kuzu",
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()
    reg = reg.load()
    idx_path = tmp_path / "search_metadata.fts.sqlite"
    conn = sqlite3.connect(str(idx_path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '2')"
        )
        conn.commit()
    finally:
        conn.close()
    old = time.time() - 5.0
    os.utime(idx_path, (old, old))
    # graph file missing -> should not require rebuild based on mtime loop
    assert not SearchIndex.needs_rebuild(reg, idx_path)
    gp = tmp_path / "g.kuzu"
    gp.write_bytes(b"x")
    assert SearchIndex.needs_rebuild(reg, idx_path)


def test_search_index_alias_is_keyword_search_index() -> None:
    assert SearchIndex is KeywordSearchIndex


def test_keyword_search_index_is_subclass_of_base() -> None:
    assert issubclass(KeywordSearchIndex, BaseSearchIndex)


def test_base_search_index_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseSearchIndex()  # type: ignore[abstract]


def test_similar_excludes_exact_match(tmp_path: Path) -> None:
    idx = _build_index_for_snapshot(tmp_path, _snapshot_customers())
    results = idx.similar("public.customers", limit=5)
    names = {r.name for r in results}
    assert "public.customers" not in names


def test_index_graph_rebuilds_index(tmp_path: Path) -> None:
    graph = tmp_path / "graphs" / "shop.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(
            _snapshot_customers(), store, run_relationship_discovery=False
        )
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="shop",
        database="shop",
        dsn="postgresql://x",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()

    idx_path = KeywordSearchIndex.default_path(tmp_path)
    idx = KeywordSearchIndex(idx_path)
    assert not idx_path.exists()
    idx.index_graph(reg.load())
    assert idx_path.exists()
    hits = idx.search("customer", limit=5)
    assert hits


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("user email", "public.users"),
        ("subscription product", "public.subscriptions"),
        ("sales amount", "public.fact_sales"),
        ("customer region", "public.dim_customer"),
        ("country city", "public.dim_geography"),
    ],
)
def test_pre203_recall_cases_rank_expected_table_first(
    tmp_path: Path, query: str, expected: str
) -> None:
    idx = _build_index_for_snapshot(tmp_path, _snapshot_pre203_recall_cases())
    hits = idx.search(query, limit=3)
    assert hits, f"expected hits for query {query!r}"
    assert hits[0].node_type == "SchemaTable"
    assert hits[0].name == expected


def test_entity_signal_lifts_table_rank_for_identity_query(tmp_path: Path) -> None:
    idx = _build_index_for_snapshot(tmp_path, _snapshot_pre203_recall_cases())
    hits = idx.search("identity account", limit=5)
    assert hits, "expected at least one hit"
    table_hits = [h for h in hits if h.node_type == "SchemaTable"]
    assert table_hits, "expected at least one SchemaTable hit"
    assert table_hits[0].name == "public.users"


def test_needs_rebuild_when_schema_version_is_outdated(tmp_path: Path) -> None:
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.save()
    idx_path = SearchIndex.default_path(tmp_path)
    conn = sqlite3.connect(str(idx_path))
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '1')"
        )
        conn.commit()
    finally:
        conn.close()
    assert SearchIndex.needs_rebuild(reg, idx_path)
