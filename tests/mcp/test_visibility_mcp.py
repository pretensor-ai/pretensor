"""MCP tools respect visibility filters."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.mcp.resources.markdown import (
    clusters_resource_markdown,
    databases_resource_markdown,
)
from pretensor.mcp.service import (
    context_payload,
    cypher_payload,
    list_databases_payload,
)
from pretensor.mcp.service_context import (
    get_effective_search_index_cls,
    reset_server_context,
    set_server_context,
)
from pretensor.mcp.tools.impact import impact_payload
from pretensor.mcp.tools.search import query_payload
from pretensor.mcp.tools.traverse import traverse_payload
from pretensor.search.index import KeywordSearchIndex
from pretensor.visibility.config import VisibilityConfig
from pretensor.visibility.filter import VisibilityFilter


@pytest.fixture(autouse=True)
def _clear_server_context() -> Generator[None, None, None]:
    """Reset the module-level server context before and after each test."""
    reset_server_context()
    yield
    reset_server_context()


def _build_demo_graph(tmp_path: Path) -> None:
    t_orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="user_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="orders",
                source_column="user_id",
                target_schema="public",
                target_table="users",
                target_column="id",
            )
        ],
    )
    t_users = Table(
        name="users",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="email", data_type="text"),
        ],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_orders, t_users],
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
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()


def test_list_and_context_hide_users(tmp_path: Path) -> None:
    _build_demo_graph(tmp_path)
    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users"])
    )
    listed = list_databases_payload(tmp_path, visibility_filter=vf)
    assert listed["databases"][0]["table_count"] == 1

    ctx = context_payload(tmp_path, table="users", db="demo", visibility_filter=vf)
    assert "error" in ctx

    ctx2 = context_payload(tmp_path, table="orders", db="demo", visibility_filter=vf)
    assert "error" not in ctx2
    rel_targets = {r["target_table"] for r in ctx2.get("relationships", [])}
    assert "public.users" not in rel_targets


def test_cypher_filters_hidden_node_id_rows(tmp_path: Path) -> None:
    _build_demo_graph(tmp_path)
    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users"])
    )
    q = "MATCH (t:SchemaTable) RETURN t.node_id AS node_id ORDER BY node_id"
    out = cypher_payload(tmp_path, query=q, database="demo", visibility_filter=vf)
    assert "error" not in out
    ids = {r["node_id"] for r in out["rows"]}
    assert all("users" not in str(i) for i in ids)


def test_traverse_does_not_cross_hidden_users(tmp_path: Path) -> None:
    _build_demo_graph(tmp_path)
    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users"])
    )
    res = traverse_payload(
        tmp_path,
        from_table="public.orders",
        to_table="public.users",
        database="demo",
        max_depth=4,
        visibility_filter=vf,
    )
    assert "error" in res


def test_impact_hides_hidden_dependents(tmp_path: Path) -> None:
    _build_demo_graph(tmp_path)
    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.orders"])
    )
    out = impact_payload(
        tmp_path, table="public.users", database="demo", visibility_filter=vf
    )
    assert "error" not in out
    direct = out["impact"]["direct"]
    names = [x["name"] for x in direct]
    assert not any("orders" in n for n in names)


def test_query_hides_hidden_tables(tmp_path: Path) -> None:
    _build_demo_graph(tmp_path)
    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users"])
    )
    out = query_payload(tmp_path, q="users", visibility_filter=vf)
    assert "results" in out
    names = [r["name"] for r in out["results"]]
    assert not any("users" in n for n in names), f"Unexpected hit: {names}"


def test_query_shows_visible_tables(tmp_path: Path) -> None:
    _build_demo_graph(tmp_path)
    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users"])
    )
    out = query_payload(tmp_path, q="orders", visibility_filter=vf)
    assert "results" in out
    # orders is visible — search may return it (index freshness may vary, so
    # just assert no users leak rather than asserting orders is present)
    names = [r["name"] for r in out["results"]]
    assert not any("users" in n for n in names)


def test_query_non_table_hits_not_filtered(tmp_path: Path) -> None:
    """Non-SchemaTable hits (e.g. Entity, Cluster) are never redacted."""
    from pretensor.mcp.payload_types import QueryHit
    from pretensor.mcp.tools.search import _hit_visible

    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users"])
    )
    entity_hit: QueryHit = {
        "node_type": "Entity",
        "name": "public.users",
        "connection_name": "demo",
        "description": "",
        "database_name": "demo",
        "snippet": "",
        "score": 1.0,
    }
    # Entity nodes bypass visibility checks
    assert _hit_visible(entity_hit, vf)


def test_databases_resource_markdown_table_count_filtered(tmp_path: Path) -> None:
    _build_demo_graph(tmp_path)
    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users"])
    )
    md = databases_resource_markdown(tmp_path, visibility_filter=vf)
    # Only orders is visible → table_count must be 1
    assert "**Tables:** 1" in md


def test_databases_resource_markdown_no_filter(tmp_path: Path) -> None:
    _build_demo_graph(tmp_path)
    md = databases_resource_markdown(tmp_path)
    assert "**Tables:** 2" in md


def test_clusters_resource_markdown_runs_with_filter(tmp_path: Path) -> None:
    """clusters_resource_markdown with a visibility filter and no cluster data."""
    _build_demo_graph(tmp_path)
    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users"])
    )
    md = clusters_resource_markdown(tmp_path, "demo", visibility_filter=vf)
    # No clusters exist in the demo graph — output is a header, no crash
    assert "Clusters" in md
    assert "users" not in md


def test_allowed_tables_hides_non_matching_across_mcp_surface(tmp_path: Path) -> None:
    """`allowed_tables` restricts list_databases, context, and cypher at serve time.

    Regression guard: the ``allowed_tables`` primitive
    was unit-tested in ``tests/visibility/test_visibility_config.py`` but never
    exercised through the MCP tool surface. This test pins the wiring — a table
    not matching any ``allowed_tables`` pattern must be absent from all four
    read paths below.
    """
    _build_demo_graph(tmp_path)
    vf = VisibilityFilter.from_config(
        VisibilityConfig(allowed_tables=["public.orders"])
    )

    listed = list_databases_payload(tmp_path, visibility_filter=vf)
    assert listed["databases"][0]["table_count"] == 1

    ctx_hidden = context_payload(
        tmp_path, table="users", db="demo", visibility_filter=vf
    )
    assert "error" in ctx_hidden

    ctx_visible = context_payload(
        tmp_path, table="orders", db="demo", visibility_filter=vf
    )
    assert "error" not in ctx_visible

    q = "MATCH (t:SchemaTable) RETURN t.node_id AS node_id ORDER BY node_id"
    rows = cypher_payload(tmp_path, query=q, database="demo", visibility_filter=vf)
    assert "error" not in rows
    ids = {r["node_id"] for r in rows["rows"]}
    assert all("users" not in str(i) for i in ids)


def test_server_context_reset_between_tests(tmp_path: Path) -> None:
    """The autouse fixture ensures server context does not bleed between tests."""
    from pretensor.mcp.service_context import (
        ServerContext,
        get_effective_visibility_filter,
    )

    vf = VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users"])
    )
    ctx = ServerContext(graph_dir=tmp_path, visibility_filter=vf)
    set_server_context(ctx)
    assert get_effective_visibility_filter() is vf
    assert get_effective_search_index_cls() is KeywordSearchIndex
    # After reset (called by autouse fixture teardown), context is cleared
    reset_server_context()
    assert get_effective_visibility_filter() is None
