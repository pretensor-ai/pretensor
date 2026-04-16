"""Traverse emits every top-tied path when ``ambiguous=true``."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.intelligence.join_paths import JoinPathEngine
from pretensor.mcp.tools.traverse import traverse_payload


def _fk(src_table: str, src_col: str, dst_table: str, dst_col: str = "id") -> ForeignKey:
    return ForeignKey(
        source_schema="public",
        source_table=src_table,
        source_column=src_col,
        target_schema="public",
        target_table=dst_table,
        target_column=dst_col,
    )


def _build_diamond_graph(tmp_path: Path) -> None:
    """Two tied FK paths from ``a`` to ``d``: a→b→d and a→c→d."""
    t_d = Table(
        name="d",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    t_b = Table(
        name="b",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="d_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("b", "d_id", "d")],
    )
    t_c = Table(
        name="c",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="d_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("c", "d_id", "d")],
    )
    t_a = Table(
        name="a",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="b_id", data_type="int", is_foreign_key=True),
            Column(name="c_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("a", "b_id", "b"), _fk("a", "c_id", "c")],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_a, t_b, t_c, t_d],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        JoinPathEngine(store).precompute("demo")
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


def _build_chain_graph(tmp_path: Path) -> None:
    """Single FK path a→b→c; no ambiguity."""
    t_c = Table(
        name="c",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    t_b = Table(
        name="b",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="c_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("b", "c_id", "c")],
    )
    t_a = Table(
        name="a",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="b_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[_fk("a", "b_id", "b")],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_a, t_b, t_c],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        JoinPathEngine(store).precompute("demo")
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


def test_precomputed_ambiguous_emits_all_tied_paths(tmp_path: Path) -> None:
    _build_diamond_graph(tmp_path)
    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.d",
        database="demo",
        max_depth=4,
    )
    assert "error" not in res
    paths = res["paths"]
    assert len(paths) == 2
    assert all(p["ambiguous"] is True for p in paths)
    intermediates = {p["steps"][0]["to_table"] for p in paths}
    assert intermediates == {"public.b", "public.c"}
    assert res["used_precomputed"] is True


def test_precomputed_unambiguous_returns_single_path(tmp_path: Path) -> None:
    _build_chain_graph(tmp_path)
    res = traverse_payload(
        tmp_path,
        from_table="public.a",
        to_table="public.c",
        database="demo",
        max_depth=4,
    )
    assert "error" not in res
    paths = res["paths"]
    assert len(paths) == 1
    assert paths[0]["ambiguous"] is False
