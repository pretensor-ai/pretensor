"""Impact low_confidence entries carry hop depth."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.mcp.tools.impact import impact_payload


def _nid(schema: str, table: str, connection: str = "demo") -> str:
    return table_node_id(connection, schema, table)


def _build_low_confidence_chain(tmp_path: Path) -> None:
    """``root`` ← FK ``a`` ← INFERRED(0.3) ``b`` ← INFERRED(0.2) ``c``."""
    t_root = Table(
        name="root",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    t_a = Table(
        name="a",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="root_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="a",
                source_column="root_id",
                target_schema="public",
                target_table="root",
                target_column="id",
            )
        ],
    )
    t_b = Table(
        name="b",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    t_c = Table(
        name="c",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_root, t_a, t_b, t_c],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        store.upsert_inferred_join(
            RelationshipCandidate(
                candidate_id="inf-b-to-a",
                source_node_id=_nid("public", "b"),
                target_node_id=_nid("public", "a"),
                source_column="id",
                target_column="id",
                source="heuristic",
                confidence=0.3,
            )
        )
        store.upsert_inferred_join(
            RelationshipCandidate(
                candidate_id="inf-c-to-b",
                source_node_id=_nid("public", "c"),
                target_node_id=_nid("public", "b"),
                source_column="id",
                target_column="id",
                source="heuristic",
                confidence=0.2,
            )
        )
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


def test_low_confidence_entries_carry_hop(tmp_path: Path) -> None:
    _build_low_confidence_chain(tmp_path)
    res = impact_payload(tmp_path, table="public.root", database="demo", max_depth=3)
    assert "error" not in res

    direct = res["impact"]["direct"]
    low = res["impact"]["low_confidence"]

    # Direct FK hit on ``a`` keeps full confidence and lands in ``direct``.
    assert [x["name"] for x in direct] == ["public.a"]
    # ``direct`` is unchanged: no hop field required there.
    assert "hop" not in direct[0]

    by_name = {x["name"]: x for x in low}
    assert set(by_name) == {"public.b", "public.c"}
    assert by_name["public.b"]["hop"] == 2
    assert by_name["public.c"]["hop"] == 3
    assert by_name["public.b"]["confidence"] == 0.3
    # path_min collapses at the weakest edge (0.2) for the 3-hop entry.
    assert by_name["public.c"]["confidence"] == 0.2


def test_impact_unambiguous_shape_unchanged(tmp_path: Path) -> None:
    _build_low_confidence_chain(tmp_path)
    res = impact_payload(tmp_path, table="public.root", database="demo", max_depth=3)
    impact = res["impact"]
    assert set(impact) == {"direct", "two_hop", "three_hop", "low_confidence"}
    # With low-confidence entries siphoned off, two_hop/three_hop stay empty.
    assert impact["two_hop"] == []
    assert impact["three_hop"] == []
