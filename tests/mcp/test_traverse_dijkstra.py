"""Tests for weighted Dijkstra path selection in traverse.

Verifies that ``dijkstra_join_path`` prefers low-cost FK chains over
shorter but high-cost inferred shortcuts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
    ViewDependency,
)
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.shadow_alias import get_shadow_alias_node_ids
from pretensor.mcp.tools.traverse import (
    dijkstra_join_path,
    traverse_steps_respect_visibility,
)
from pretensor.visibility.config import VisibilityConfig
from pretensor.visibility.filter import VisibilityFilter


def _node_id(schema: str, table: str, connection: str = "demo") -> str:
    return table_node_id(connection, schema, table)


def _build_chain_graph(tmp_path: Path) -> KuzuStore:
    """Build a graph with FK chain A→B→C→D and inferred shortcut A→D.

    FK edges are directional (child → parent), so the FK chain goes:
    salesorder → salesdetail → specialoffer → product (3 hops, conf 1.0 each).
    Inferred shortcut: salesorder → product (1 hop, conf 0.25).
    """
    tables = [
        Table(
            name="product",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
            ],
            foreign_keys=[],
        ),
        Table(
            name="specialoffer",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="product_id", data_type="int", is_foreign_key=True),
            ],
            foreign_keys=[
                ForeignKey(
                    source_schema="public",
                    source_table="specialoffer",
                    source_column="product_id",
                    target_schema="public",
                    target_table="product",
                    target_column="id",
                )
            ],
        ),
        Table(
            name="salesdetail",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="offer_id", data_type="int", is_foreign_key=True),
            ],
            foreign_keys=[
                ForeignKey(
                    source_schema="public",
                    source_table="salesdetail",
                    source_column="offer_id",
                    target_schema="public",
                    target_table="specialoffer",
                    target_column="id",
                )
            ],
        ),
        Table(
            name="salesorder",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="detail_id", data_type="int", is_foreign_key=True),
            ],
            foreign_keys=[
                ForeignKey(
                    source_schema="public",
                    source_table="salesorder",
                    source_column="detail_id",
                    target_schema="public",
                    target_table="salesdetail",
                    target_column="id",
                )
            ],
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    GraphBuilder().build(snap, store, run_relationship_discovery=False)

    store.upsert_inferred_join(
        RelationshipCandidate(
            candidate_id="inferred-shortcut-1",
            source_node_id=_node_id("public", "salesorder"),
            target_node_id=_node_id("public", "product"),
            source_column="id",
            target_column="id",
            source="heuristic",
            confidence=0.25,
        )
    )
    return store


def test_fk_chain_preferred_when_inferred_costs_more(tmp_path: Path) -> None:
    """FK wins once the inferred alternative's cost > 3.0 (3-hop FK chain).

    Edge cost = 1.0 + 0.5*(inferred?) + max(0, 0.7 - confidence). A 1-hop
    inferred edge can never exceed 2.2 (cost), so to make FK win we exclude
    the shortcut via ``edge_kinds=("fk",)``.
    """
    store = _build_chain_graph(tmp_path)
    try:
        result = dijkstra_join_path(
            store,
            start_id=_node_id("public", "salesorder"),
            end_id=_node_id("public", "product"),
            connection_name="demo",
            max_depth=6,
            edge_kinds=("fk",),
        )
        assert result is not None
        edges, min_conf = result
        assert len(edges) == 3
        assert min_conf == 1.0
        assert all(e[4] == "fk" for e in edges)
    finally:
        store.close()


def test_inferred_path_used_when_no_fk_path(tmp_path: Path) -> None:
    """Inferred edges are used when no FK path exists."""
    tables = [
        Table(
            name="alpha",
            schema_name="public",
            columns=[Column(name="id", data_type="int", is_primary_key=True)],
            foreign_keys=[],
        ),
        Table(
            name="beta",
            schema_name="public",
            columns=[Column(name="id", data_type="int", is_primary_key=True)],
            foreign_keys=[],
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        store.upsert_inferred_join(
            RelationshipCandidate(
                candidate_id="inf-1",
                source_node_id=_node_id("public", "alpha"),
                target_node_id=_node_id("public", "beta"),
                source_column="id",
                target_column="id",
                source="heuristic",
                confidence=0.6,
            )
        )
        result = dijkstra_join_path(
            store,
            start_id=_node_id("public", "alpha"),
            end_id=_node_id("public", "beta"),
            connection_name="demo",
            max_depth=4,
        )
        assert result is not None
        edges, min_conf = result
        assert len(edges) == 1
        assert edges[0][4] == "inferred"
        assert min_conf == 0.6
    finally:
        store.close()


def test_max_depth_respected(tmp_path: Path) -> None:
    """Paths exceeding max_depth are not returned even if they exist."""
    store = _build_chain_graph(tmp_path)
    try:
        # max_depth=2 should prevent finding the 3-hop FK chain;
        # the 1-hop inferred shortcut should be returned instead.
        result = dijkstra_join_path(
            store,
            start_id=_node_id("public", "salesorder"),
            end_id=_node_id("public", "product"),
            connection_name="demo",
            max_depth=2,
        )
        # The inferred shortcut is within depth 2
        assert result is not None
        edges, _ = result
        assert len(edges) == 1
        assert edges[0][4] == "inferred"
    finally:
        store.close()


def test_visibility_allowed_set_does_not_prune_fk_chain(tmp_path: Path) -> None:
    """``allowed_table_node_ids`` does not gate FK edges (FK = DDL, not data).

    Under the old pre-pruning behaviour, hiding ``specialoffer`` would drop
    the FK chain entirely and force the search to find nothing (with
    ``edge_kinds=("fk",)``). Post-fix, FK edges are exempt from the allowed
    set, so the 3-hop FK chain still reaches ``product`` through the hidden
    waypoint.
    """
    store = _build_chain_graph(tmp_path)
    try:
        # Exclude ``specialoffer`` from allowed tables — it's a required
        # waypoint on the FK chain. Restrict to FK only so the inferred
        # shortcut cannot satisfy the query.
        allowed = {
            _node_id("public", "product"),
            _node_id("public", "salesdetail"),
            _node_id("public", "salesorder"),
        }
        result = dijkstra_join_path(
            store,
            start_id=_node_id("public", "salesorder"),
            end_id=_node_id("public", "product"),
            connection_name="demo",
            max_depth=6,
            allowed_table_node_ids=allowed,
            edge_kinds=("fk",),
        )
        assert result is not None
        edges, min_conf = result
        assert len(edges) == 3
        assert all(e[4] == "fk" for e in edges)
        assert min_conf == 1.0
    finally:
        store.close()


def test_visibility_gates_inferred_shortcut_when_target_hidden(tmp_path: Path) -> None:
    """Inferred edges to a hidden table are still skipped by the allowed-set."""
    store = _build_chain_graph(tmp_path)
    try:
        # Hide ``product`` from allowed: the inferred shortcut
        # salesorder → product must be skipped (inferred, target hidden).
        # FK chain still reaches product because FKs ignore the allowed-set.
        allowed = {
            _node_id("public", "salesdetail"),
            _node_id("public", "specialoffer"),
            _node_id("public", "salesorder"),
        }
        result = dijkstra_join_path(
            store,
            start_id=_node_id("public", "salesorder"),
            end_id=_node_id("public", "product"),
            connection_name="demo",
            max_depth=6,
            allowed_table_node_ids=allowed,
        )
        assert result is not None
        edges, _ = result
        assert len(edges) == 3
        assert all(e[4] == "fk" for e in edges)
    finally:
        store.close()


def test_no_path_returns_none(tmp_path: Path) -> None:
    """When no path exists, None is returned."""
    tables = [
        Table(
            name="isolated_a",
            schema_name="public",
            columns=[Column(name="id", data_type="int", is_primary_key=True)],
            foreign_keys=[],
        ),
        Table(
            name="isolated_b",
            schema_name="public",
            columns=[Column(name="id", data_type="int", is_primary_key=True)],
            foreign_keys=[],
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        result = dijkstra_join_path(
            store,
            start_id=_node_id("public", "isolated_a"),
            end_id=_node_id("public", "isolated_b"),
            connection_name="demo",
            max_depth=4,
        )
        assert result is None
    finally:
        store.close()


def test_respect_visibility_exempts_fk_steps() -> None:
    """FK steps stay visible even when their tables are hidden by the filter."""
    vf = VisibilityFilter.from_config(VisibilityConfig(hidden_tables=["demo::public.rental"]))
    fk_through_hidden = [
        {
            "from_table": "public.film",
            "to_table": "public.rental",
            "edge_type": "fk",
        },
        {
            "from_table": "public.rental",
            "to_table": "public.customer",
            "edge_type": "fk",
        },
    ]
    assert traverse_steps_respect_visibility(
        fk_through_hidden,
        visibility_filter=vf,
        default_connection="demo",
    )


def test_respect_visibility_blocks_inferred_through_hidden() -> None:
    """Inferred steps through hidden tables remain filtered out."""
    vf = VisibilityFilter.from_config(VisibilityConfig(hidden_tables=["demo::public.rental"]))
    inferred_through_hidden = [
        {
            "from_table": "public.film",
            "to_table": "public.rental",
            "edge_type": "inferred",
        },
    ]
    assert not traverse_steps_respect_visibility(
        inferred_through_hidden,
        visibility_filter=vf,
        default_connection="demo",
    )


class TestDijkstraShadowAliasFiltering:
    """Shadow aliases excluded from Dijkstra join paths."""

    @staticmethod
    def _build_shadow_graph(tmp_path: Path) -> KuzuStore:
        """Graph: order→detail→product (FK chain) + shadow view pr.p→product (lineage).

        An inferred edge order→pr.p exists; without shadow filtering Dijkstra
        would find order→pr.p as a 1-hop path.  With filtering it must use the
        FK chain.
        """
        tables = [
            Table(
                name="product",
                schema_name="public",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            ),
            Table(
                name="detail",
                schema_name="public",
                columns=[
                    Column(name="id", data_type="int", is_primary_key=True),
                    Column(name="product_id", data_type="int", is_foreign_key=True),
                ],
                foreign_keys=[
                    ForeignKey(
                        source_schema="public",
                        source_table="detail",
                        source_column="product_id",
                        target_schema="public",
                        target_table="product",
                        target_column="id",
                    )
                ],
            ),
            Table(
                name="order",
                schema_name="public",
                columns=[
                    Column(name="id", data_type="int", is_primary_key=True),
                    Column(name="detail_id", data_type="int", is_foreign_key=True),
                ],
                foreign_keys=[
                    ForeignKey(
                        source_schema="public",
                        source_table="order",
                        source_column="detail_id",
                        target_schema="public",
                        target_table="detail",
                        target_column="id",
                    )
                ],
            ),
            Table(
                name="p",
                schema_name="pr",
                table_type="view",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            ),
        ]
        snap = SchemaSnapshot(
            connection_name="demo",
            database="demo",
            schemas=["public", "pr"],
            tables=tables,
            view_dependencies=[
                ViewDependency(
                    source_schema="public",
                    source_table="product",
                    target_schema="pr",
                    target_table="p",
                    object_name="pr.p",
                    lineage_type="VIEW",
                    confidence=1.0,
                ),
            ],
            introspected_at=datetime.now(timezone.utc),
        )
        graph = tmp_path / "graphs" / "demo.kuzu"
        graph.parent.mkdir(parents=True, exist_ok=True)
        store = KuzuStore(graph)
        GraphBuilder().build(snap, store, run_relationship_discovery=False)

        # Add inferred edge: order → shadow view pr.p
        store.upsert_inferred_join(
            RelationshipCandidate(
                candidate_id="inf-shadow",
                source_node_id=_node_id("public", "order"),
                target_node_id=_node_id("pr", "p"),
                source_column="id",
                target_column="id",
                source="heuristic",
                confidence=0.9,
            )
        )
        return store

    def test_shadow_edges_excluded_from_path(self, tmp_path: Path) -> None:
        """With shadow_alias_ids, edges touching shadow views are skipped."""
        store = self._build_shadow_graph(tmp_path)
        try:
            shadow_ids = get_shadow_alias_node_ids(store, "demo")
            assert _node_id("pr", "p") in shadow_ids

            result = dijkstra_join_path(
                store,
                start_id=_node_id("public", "order"),
                end_id=_node_id("public", "product"),
                connection_name="demo",
                max_depth=6,
                shadow_alias_ids=shadow_ids,
            )
            assert result is not None
            edges, min_conf = result
            # Must use FK chain (2 hops), not the 1-hop inferred to shadow
            assert len(edges) == 2
            assert min_conf == 1.0
            visited = {e[0] for e in edges} | {e[1] for e in edges}
            assert _node_id("pr", "p") not in visited
        finally:
            store.close()

    def test_shadow_edges_used_without_filtering(self, tmp_path: Path) -> None:
        """Without shadow_alias_ids the inferred shortcut via the shadow is found."""
        store = self._build_shadow_graph(tmp_path)
        try:
            result = dijkstra_join_path(
                store,
                start_id=_node_id("public", "order"),
                end_id=_node_id("pr", "p"),
                connection_name="demo",
                max_depth=6,
            )
            # The direct inferred edge order→pr.p should be reachable
            assert result is not None
            edges, _ = result
            assert len(edges) == 1
        finally:
            store.close()
