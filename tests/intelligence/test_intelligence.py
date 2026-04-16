"""Tests for graph export, clustering, and join paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.intelligence.graph_export import GraphExporter
from pretensor.intelligence.pipeline import run_intelligence_layer_sync


def _three_table_chain(tmp_path: Path) -> Path:
    """public.a -> public.b -> public.c FK chain."""
    ta = Table(
        name="a",
        schema_name="public",
        columns=[Column(name="id", data_type="int")],
        foreign_keys=[],
    )
    tb = Table(
        name="b",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int"),
            Column(name="a_id", data_type="int"),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="b",
                source_column="a_id",
                target_schema="public",
                target_table="a",
                target_column="id",
            )
        ],
    )
    tc = Table(
        name="c",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int"),
            Column(name="b_id", data_type="int"),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="c",
                source_column="b_id",
                target_schema="public",
                target_table="b",
                target_column="id",
            )
        ],
    )
    snap = SchemaSnapshot(
        connection_name="chain",
        database="chaindb",
        schemas=["public"],
        tables=[ta, tb, tc],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "chain.kuzu"
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()
    return graph


def test_graph_exporter_builds_igraph(tmp_path: Path) -> None:
    graph_path = _three_table_chain(tmp_path)
    store = KuzuStore(graph_path)
    try:
        g = GraphExporter(store).to_igraph("chaindb")
        assert g.vcount() == 3
        assert g.ecount() == 2
        assert set(g.vs["name"]) == {"public.a", "public.b", "public.c"}
    finally:
        store.close()


def test_join_path_engine_persists_both_directions(tmp_path: Path) -> None:
    graph_path = _three_table_chain(tmp_path)
    store = KuzuStore(graph_path)
    try:
        run_intelligence_layer_sync(store, "chaindb")
        n = store.query_all_rows("MATCH (p:JoinPath) RETURN count(*)")[0][0]
        assert int(n) >= 2
    finally:
        store.close()
