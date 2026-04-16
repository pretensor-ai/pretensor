"""Tests for shadow-alias detection helpers."""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.config import GraphConfig
from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.graph_models.edge import LineageEdge
from pretensor.graph_models.node import GraphNode
from pretensor.intelligence.discovery import RelationshipDiscovery
from pretensor.intelligence.graph_export import GraphExporter
from pretensor.intelligence.shadow_alias import (
    get_shadow_alias_node_ids,
    is_shadow_alias,
)
from pretensor.mcp.tools.context import (
    relationships_for_table,
    shadow_aliases_for_base,
    shadow_of_for_view,
)

DB_KEY = "db"
CONN = "conn"


def _table(schema: str, name: str, *, table_type: str = "table") -> GraphNode:
    return GraphNode(
        node_id=table_node_id(CONN, schema, name),
        connection_name=CONN,
        database=DB_KEY,
        schema_name=schema,
        table_name=name,
        row_count=None,
        comment=None,
        entity_type=None,
        table_type=table_type,
        seq_scan_count=None,
        idx_scan_count=None,
        insert_count=None,
        update_count=None,
        delete_count=None,
        is_partitioned=None,
        partition_key=None,
        grants_json=None,
        access_read_count=None,
        access_write_count=None,
        days_since_last_access=None,
        potentially_unused=None,
        table_bytes=None,
        clustering_key=None,
    )


def _lineage(src_schema: str, src_name: str, tgt_schema: str, tgt_name: str) -> LineageEdge:
    return LineageEdge(
        edge_id=f"lineage::{src_schema}.{src_name}->{tgt_schema}.{tgt_name}",
        source_node_id=table_node_id(CONN, src_schema, src_name),
        target_node_id=table_node_id(CONN, tgt_schema, tgt_name),
        source=f"{CONN}:{tgt_schema}.{tgt_name}",
        lineage_type="VIEW",
        confidence=1.0,
    )


@pytest.fixture()
def store(tmp_path: Path) -> Generator[KuzuStore]:
    """Build a small graph mimicking AdventureWorks shadow schemas."""
    s = KuzuStore(tmp_path / "test.kuzu")
    s.ensure_schema()

    # Base tables
    s.upsert_table(_table("humanresources", "department"))
    s.upsert_table(_table("production", "product"))
    s.upsert_table(_table("sales", "salesorderheader"))

    # Shadow alias views (single-source, lineage_in == 1)
    s.upsert_table(_table("hr", "d", table_type="view"))
    s.upsert_table(_table("pr", "p", table_type="view"))
    s.upsert_lineage_edge(_lineage("humanresources", "department", "hr", "d"))
    s.upsert_lineage_edge(_lineage("production", "product", "pr", "p"))

    # Multi-source analytical view (lineage_in == 3, NOT a shadow alias)
    s.upsert_table(_table("sales", "vsalesperson", table_type="view"))
    s.upsert_lineage_edge(_lineage("humanresources", "department", "sales", "vsalesperson"))
    s.upsert_lineage_edge(_lineage("production", "product", "sales", "vsalesperson"))
    s.upsert_lineage_edge(_lineage("sales", "salesorderheader", "sales", "vsalesperson"))

    # Orphan view with no lineage (not a shadow alias either)
    s.upsert_table(_table("public", "orphan_view", table_type="view"))

    yield s
    s.close()


class TestGetShadowAliasNodeIds:
    def test_detects_shadow_aliases(self, store: KuzuStore) -> None:
        ids = get_shadow_alias_node_ids(store, DB_KEY)
        expected = {
            table_node_id(CONN, "hr", "d"),
            table_node_id(CONN, "pr", "p"),
        }
        assert ids == expected

    def test_excludes_multi_source_view(self, store: KuzuStore) -> None:
        ids = get_shadow_alias_node_ids(store, DB_KEY)
        assert table_node_id(CONN, "sales", "vsalesperson") not in ids

    def test_excludes_base_tables(self, store: KuzuStore) -> None:
        ids = get_shadow_alias_node_ids(store, DB_KEY)
        assert table_node_id(CONN, "production", "product") not in ids

    def test_excludes_orphan_view(self, store: KuzuStore) -> None:
        ids = get_shadow_alias_node_ids(store, DB_KEY)
        assert table_node_id(CONN, "public", "orphan_view") not in ids

    def test_disabled_by_config(self, store: KuzuStore) -> None:
        cfg = GraphConfig(collapse_shadow_aliases=False)
        ids = get_shadow_alias_node_ids(store, DB_KEY, config=cfg)
        assert ids == frozenset()

    def test_custom_max_lineage_in(self, store: KuzuStore) -> None:
        # With max=3, the multi-source view (lineage_in=3) also qualifies
        cfg = GraphConfig(lineage_in_max_for_alias=3)
        ids = get_shadow_alias_node_ids(store, DB_KEY, config=cfg)
        assert table_node_id(CONN, "sales", "vsalesperson") in ids
        assert table_node_id(CONN, "hr", "d") in ids

    def test_returns_frozenset(self, store: KuzuStore) -> None:
        ids = get_shadow_alias_node_ids(store, DB_KEY)
        assert isinstance(ids, frozenset)


class TestIsShadowAlias:
    def test_shadow_view_returns_true(self, store: KuzuStore) -> None:
        assert is_shadow_alias(store, table_node_id(CONN, "hr", "d")) is True

    def test_base_table_returns_false(self, store: KuzuStore) -> None:
        assert is_shadow_alias(store, table_node_id(CONN, "production", "product")) is False

    def test_multi_source_view_returns_false(self, store: KuzuStore) -> None:
        assert is_shadow_alias(store, table_node_id(CONN, "sales", "vsalesperson")) is False

    def test_orphan_view_returns_false(self, store: KuzuStore) -> None:
        assert is_shadow_alias(store, table_node_id(CONN, "public", "orphan_view")) is False

    def test_disabled_by_config(self, store: KuzuStore) -> None:
        cfg = GraphConfig(collapse_shadow_aliases=False)
        assert is_shadow_alias(store, table_node_id(CONN, "hr", "d"), config=cfg) is False


class TestGraphExporterShadowFiltering:
    """Shadow aliases excluded from clustering igraph."""

    def test_shadow_aliases_excluded_from_igraph(self, store: KuzuStore) -> None:
        g = GraphExporter(store).to_igraph(DB_KEY)
        node_ids_in_graph = set(g.vs["node_id"])
        # Shadow aliases should be excluded
        assert table_node_id(CONN, "hr", "d") not in node_ids_in_graph
        assert table_node_id(CONN, "pr", "p") not in node_ids_in_graph
        # Base tables should remain
        assert table_node_id(CONN, "humanresources", "department") in node_ids_in_graph
        assert table_node_id(CONN, "production", "product") in node_ids_in_graph
        # Multi-source views should remain
        assert table_node_id(CONN, "sales", "vsalesperson") in node_ids_in_graph

    def test_shadow_aliases_included_when_disabled(self, store: KuzuStore) -> None:
        cfg = GraphConfig(collapse_shadow_aliases=False)
        g = GraphExporter(store).to_igraph(DB_KEY, config=cfg)
        node_ids_in_graph = set(g.vs["node_id"])
        # Everything included when disabled
        assert table_node_id(CONN, "hr", "d") in node_ids_in_graph
        assert table_node_id(CONN, "pr", "p") in node_ids_in_graph

    def test_vertex_count_drops_by_shadow_count(self, store: KuzuStore) -> None:
        g_with = GraphExporter(store).to_igraph(
            DB_KEY, config=GraphConfig(collapse_shadow_aliases=False)
        )
        g_without = GraphExporter(store).to_igraph(DB_KEY)
        # 2 shadow aliases (hr.d, pr.p) should be removed
        assert g_with.vcount() - g_without.vcount() == 2


class TestDiscoveryShadowFiltering:
    """Shadow aliases filtered from INFERRED_JOIN candidates."""

    def test_candidates_involving_shadow_aliases_dropped(self, tmp_path: Path) -> None:
        """Candidates touching a shadow view should not appear in discover() output."""
        s = KuzuStore(tmp_path / "disc.kuzu")
        s.ensure_schema()

        # Base table + shadow view with shared column name → would normally
        # generate a heuristic_same_name candidate.
        base = _table("public", "product")
        shadow = _table("pr", "p", table_type="view")
        s.upsert_table(base)
        s.upsert_table(shadow)
        s.upsert_lineage_edge(_lineage("public", "product", "pr", "p"))

        # Another real table to get cross-table candidates
        other = _table("public", "category")
        s.upsert_table(other)

        snap = SchemaSnapshot(
            connection_name=CONN,
            database=DB_KEY,
            schemas=["public", "pr"],
            tables=[
                Table(
                    name="product",
                    schema_name="public",
                    columns=[
                        Column(name="id", data_type="int", is_primary_key=True),
                        Column(name="category_id", data_type="int"),
                    ],
                ),
                Table(
                    name="p",
                    schema_name="pr",
                    columns=[
                        Column(name="id", data_type="int", is_primary_key=True),
                        Column(name="category_id", data_type="int"),
                    ],
                ),
                Table(
                    name="category",
                    schema_name="public",
                    columns=[
                        Column(name="id", data_type="int", is_primary_key=True),
                    ],
                ),
            ],
            introspected_at=datetime.now(timezone.utc),
        )

        disco = RelationshipDiscovery(s, graph_config=GraphConfig())
        results = disco.discover(snap)

        shadow_nid = table_node_id(CONN, "pr", "p")
        # No candidate should reference the shadow alias
        for c in results:
            assert c.source_node_id != shadow_nid, f"shadow alias in source: {c}"
            assert c.target_node_id != shadow_nid, f"shadow alias in target: {c}"

        s.close()

    def test_candidates_preserved_when_disabled(self, tmp_path: Path) -> None:
        """With collapse disabled, shadow alias candidates should be kept."""
        s = KuzuStore(tmp_path / "disc2.kuzu")
        s.ensure_schema()

        s.upsert_table(_table("public", "product"))
        s.upsert_table(_table("pr", "p", table_type="view"))
        s.upsert_lineage_edge(_lineage("public", "product", "pr", "p"))
        s.upsert_table(_table("public", "category"))

        snap = SchemaSnapshot(
            connection_name=CONN,
            database=DB_KEY,
            schemas=["public", "pr"],
            tables=[
                Table(
                    name="product",
                    schema_name="public",
                    columns=[
                        Column(name="id", data_type="int", is_primary_key=True),
                        Column(name="category_id", data_type="int"),
                    ],
                ),
                Table(
                    name="p",
                    schema_name="pr",
                    columns=[
                        Column(name="id", data_type="int", is_primary_key=True),
                        Column(name="category_id", data_type="int"),
                    ],
                ),
                Table(
                    name="category",
                    schema_name="public",
                    columns=[
                        Column(name="id", data_type="int", is_primary_key=True),
                    ],
                ),
            ],
            introspected_at=datetime.now(timezone.utc),
        )

        cfg = GraphConfig(collapse_shadow_aliases=False)
        disco = RelationshipDiscovery(s, graph_config=cfg)
        results = disco.discover(snap)

        shadow_nid = table_node_id(CONN, "pr", "p")
        # With collapsing disabled, shadow alias candidates should exist
        shadow_candidates = [
            c
            for c in results
            if c.source_node_id == shadow_nid or c.target_node_id == shadow_nid
        ]
        assert len(shadow_candidates) > 0

        s.close()


class TestContextShadowAliasFields:
    """Shadow alias context fields and relationship suppression."""

    def test_base_table_lists_shadow_aliases(self, store: KuzuStore) -> None:
        shadow_ids = get_shadow_alias_node_ids(store, DB_KEY)
        base_nid = table_node_id(CONN, "humanresources", "department")
        aliases = shadow_aliases_for_base(store, base_nid, shadow_ids)
        assert aliases == ["hr.d"]

    def test_base_table_with_multiple_aliases(self, store: KuzuStore) -> None:
        shadow_ids = get_shadow_alias_node_ids(store, DB_KEY)
        base_nid = table_node_id(CONN, "production", "product")
        aliases = shadow_aliases_for_base(store, base_nid, shadow_ids)
        assert aliases == ["pr.p"]

    def test_non_base_table_no_aliases(self, store: KuzuStore) -> None:
        shadow_ids = get_shadow_alias_node_ids(store, DB_KEY)
        nid = table_node_id(CONN, "sales", "salesorderheader")
        aliases = shadow_aliases_for_base(store, nid, shadow_ids)
        assert aliases == []

    def test_shadow_view_has_shadow_of(self, store: KuzuStore) -> None:
        shadow_ids = get_shadow_alias_node_ids(store, DB_KEY)
        shadow_nid = table_node_id(CONN, "hr", "d")
        result = shadow_of_for_view(store, shadow_nid, shadow_ids)
        assert result == "humanresources.department"

    def test_base_table_no_shadow_of(self, store: KuzuStore) -> None:
        shadow_ids = get_shadow_alias_node_ids(store, DB_KEY)
        base_nid = table_node_id(CONN, "humanresources", "department")
        result = shadow_of_for_view(store, base_nid, shadow_ids)
        assert result is None

    def test_relationships_suppress_shadow_targets(self, store: KuzuStore) -> None:
        """INFERRED_JOIN rows targeting shadow aliases are filtered out."""
        shadow_ids = get_shadow_alias_node_ids(store, DB_KEY)
        # Use any base table node_id — the point is shadow targets are suppressed
        base_nid = table_node_id(CONN, "humanresources", "department")
        rels = relationships_for_table(store, base_nid, shadow_alias_ids=shadow_ids)
        shadow_tables = {"hr.d", "pr.p"}
        for r in rels:
            target = r.get("target_table", "")
            assert target not in shadow_tables, (
                f"shadow alias {target} should be suppressed"
            )
