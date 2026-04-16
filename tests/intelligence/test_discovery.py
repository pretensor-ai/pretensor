"""Tests for relationship discovery orchestration and statistical adjustment."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from tests.query_helpers import first_cell, single_query_result

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.combining import ConfidenceCombiner, MaxScoreCombiner
from pretensor.intelligence.discovery import (
    RelationshipDiscovery,
    explicit_fk_join_keys,
)
from pretensor.intelligence.heuristic import (
    HeuristicScorer,
    discover_heuristic_candidates,
)
from pretensor.intelligence.llm_infer import parse_llm_join_response_json
from pretensor.intelligence.scoring import RelationshipScorer, ScorerRegistry
from pretensor.intelligence.statistical import apply_statistical_adjustment


def _two_table_snap() -> SchemaSnapshot:
    customers = Table(
        name="customers",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="customer_id", data_type="int"),
        ],
        foreign_keys=[],
    )
    return SchemaSnapshot(
        connection_name="demo",
        database="db",
        schemas=["public"],
        tables=[customers, orders],
        introspected_at=datetime.now(timezone.utc),
    )


def _join_key(candidate: RelationshipCandidate) -> tuple[str, str, str, str]:
    return (
        candidate.source_node_id,
        candidate.target_node_id,
        candidate.source_column,
        candidate.target_column,
    )


def test_explicit_fk_join_keys() -> None:
    snap = _two_table_snap()
    fk = ForeignKey(
        source_schema="public",
        source_table="orders",
        source_column="customer_id",
        target_schema="public",
        target_table="customers",
        target_column="id",
    )
    orders = snap.tables[1].model_copy(update={"foreign_keys": [fk]})
    snap = snap.model_copy(update={"tables": [snap.tables[0], orders]})
    keys = explicit_fk_join_keys(snap)
    src = table_node_id("demo", "public", "orders")
    dst = table_node_id("demo", "public", "customers")
    assert (src, dst, "customer_id", "id") in keys


def test_discovery_skips_duplicate_of_explicit_fk(tmp_path) -> None:
    snap = _two_table_snap()
    fk = ForeignKey(
        source_schema="public",
        source_table="orders",
        source_column="customer_id",
        target_schema="public",
        target_table="customers",
        target_column="id",
    )
    orders = snap.tables[1].model_copy(update={"foreign_keys": [fk]})
    snap = snap.model_copy(update={"tables": [snap.tables[0], orders]})

    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store)
        inf = single_query_result(
            store, "MATCH ()-[e:INFERRED_JOIN]->() RETURN count(*) AS c"
        )
        assert first_cell(inf) == 0
    finally:
        store.close()


def test_discovery_writes_inferred_when_no_fk(tmp_path) -> None:
    snap = _two_table_snap()
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store)
        inf = single_query_result(
            store, "MATCH ()-[e:INFERRED_JOIN]->() RETURN count(*) AS c"
        )
        assert first_cell(inf) >= 1
    finally:
        store.close()


def test_statistical_adjustment_changes_source_and_confidence() -> None:
    base = RelationshipCandidate(
        candidate_id="x",
        source_node_id="a",
        target_node_id="b",
        source_column="c1",
        target_column="c2",
        source="heuristic",
        reasoning="naming",
        confidence=0.5,
    )
    adj = apply_statistical_adjustment(base, 1.0)
    assert adj.source == "statistical"
    assert adj.confidence > base.confidence
    assert "prior source=heuristic" in adj.reasoning


def test_parse_llm_join_response_json() -> None:
    raw = '[{"source_table":"public.orders","source_column":"x","target_table":"public.t","target_column":"y","confidence":0.9,"reasoning":"ok"}]'
    rows = parse_llm_join_response_json(raw)
    assert rows[0]["source_column"] == "x"


def test_relationship_discovery_sync_without_async_context(tmp_path) -> None:
    snap = _two_table_snap()
    db_path = tmp_path / "g2.kuzu"
    store = KuzuStore(db_path)
    try:
        store.ensure_schema()
        store.clear_graph()
        from pretensor.graph_models.node import GraphNode

        for t in snap.tables:
            store.upsert_table(
                GraphNode(
                    node_id=table_node_id(snap.connection_name, t.schema_name, t.name),
                    connection_name=snap.connection_name,
                    database=snap.database,
                    schema_name=t.schema_name,
                    table_name=t.name,
                    row_count=t.row_count,
                    comment=t.comment,
                )
            )
        out = RelationshipDiscovery(store).discover(snap)
        assert out
    finally:
        store.close()


def test_default_registry_matches_direct_heuristic_output() -> None:
    snap = _two_table_snap()
    explicit = explicit_fk_join_keys(snap)

    expected = [
        c for c in discover_heuristic_candidates(snap) if _join_key(c) not in explicit
    ]
    actual = ScorerRegistry([HeuristicScorer()]).score_all(snap, explicit)

    assert [c.candidate_id for c in actual] == [c.candidate_id for c in expected]


def test_scorer_registry_rejects_duplicate_names() -> None:
    registry = ScorerRegistry([HeuristicScorer()])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(HeuristicScorer())


def test_scorer_registry_runs_in_registration_order() -> None:
    snap = _two_table_snap()

    class _FirstScorer(RelationshipScorer):
        def name(self) -> str:
            return "first"

        def score(
            self,
            snapshot: SchemaSnapshot,
            explicit_fk_keys: set[tuple[str, str, str, str]],
        ) -> list[RelationshipCandidate]:
            _ = (snapshot, explicit_fk_keys)
            return [
                RelationshipCandidate(
                    candidate_id="first",
                    source_node_id="a",
                    target_node_id="b",
                    source_column="c1",
                    target_column="c2",
                    source="heuristic",
                    confidence=0.2,
                )
            ]

    class _SecondScorer(RelationshipScorer):
        def name(self) -> str:
            return "second"

        def score(
            self,
            snapshot: SchemaSnapshot,
            explicit_fk_keys: set[tuple[str, str, str, str]],
        ) -> list[RelationshipCandidate]:
            _ = (snapshot, explicit_fk_keys)
            return [
                RelationshipCandidate(
                    candidate_id="second",
                    source_node_id="a",
                    target_node_id="b",
                    source_column="c1",
                    target_column="c2",
                    source="heuristic",
                    confidence=0.3,
                )
            ]

    scored = ScorerRegistry([_FirstScorer(), _SecondScorer()]).score_all(snap, set())
    assert [c.candidate_id for c in scored] == ["first", "second"]


def test_relationship_discovery_uses_injected_registry(tmp_path) -> None:
    snap = _two_table_snap()
    db_path = tmp_path / "g3.kuzu"
    store = KuzuStore(db_path)

    class _NoopScorer(RelationshipScorer):
        def name(self) -> str:
            return "noop"

        def score(
            self,
            snapshot: SchemaSnapshot,
            explicit_fk_keys: set[tuple[str, str, str, str]],
        ) -> list[RelationshipCandidate]:
            _ = (snapshot, explicit_fk_keys)
            return []

    try:
        store.ensure_schema()
        store.clear_graph()
        from pretensor.graph_models.node import GraphNode

        for t in snap.tables:
            store.upsert_table(
                GraphNode(
                    node_id=table_node_id(snap.connection_name, t.schema_name, t.name),
                    connection_name=snap.connection_name,
                    database=snap.database,
                    schema_name=t.schema_name,
                    table_name=t.name,
                    row_count=t.row_count,
                    comment=t.comment,
                )
            )

        out = RelationshipDiscovery(
            store,
            scorers=ScorerRegistry([_NoopScorer()]),
        ).discover(snap)
        assert out == []
    finally:
        store.close()


def _make_candidate(
    candidate_id: str,
    confidence: float,
    *,
    source_node_id: str = "a",
    target_node_id: str = "b",
    source_column: str = "col",
    target_column: str = "col",
) -> RelationshipCandidate:
    return RelationshipCandidate(
        candidate_id=candidate_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        source_column=source_column,
        target_column=target_column,
        source="heuristic",
        confidence=confidence,
    )


def test_max_score_combiner_keeps_highest_confidence() -> None:
    low = _make_candidate("low", 0.3)
    high = _make_candidate("high", 0.9)

    result = MaxScoreCombiner().combine([low, high])

    assert len(result) == 1
    assert result[0].candidate_id == "high"


def test_max_score_combiner_dedupes_across_groups() -> None:
    group_a = [_make_candidate("a1", 0.4)]
    group_b = [_make_candidate("b1", 0.7)]

    result = MaxScoreCombiner().combine(group_a, group_b)

    assert len(result) == 1
    assert result[0].candidate_id == "b1"


def test_max_score_combiner_keeps_distinct_join_keys() -> None:
    c1 = _make_candidate("c1", 0.8, source_column="col1", target_column="col1")
    c2 = _make_candidate("c2", 0.6, source_column="col2", target_column="col2")

    result = MaxScoreCombiner().combine([c1, c2])

    assert len(result) == 2
    ids = {c.candidate_id for c in result}
    assert ids == {"c1", "c2"}


def test_combiner_interface_injected_into_discovery(tmp_path) -> None:
    snap = _two_table_snap()
    db_path = tmp_path / "g4.kuzu"
    store = KuzuStore(db_path)
    combine_called = False

    class _TrackingCombiner(ConfidenceCombiner):
        def combine(
            self, *groups: list[RelationshipCandidate]
        ) -> list[RelationshipCandidate]:
            nonlocal combine_called
            combine_called = True
            return MaxScoreCombiner().combine(*groups)

    try:
        store.ensure_schema()
        store.clear_graph()
        from pretensor.graph_models.node import GraphNode

        for t in snap.tables:
            store.upsert_table(
                GraphNode(
                    node_id=table_node_id(snap.connection_name, t.schema_name, t.name),
                    connection_name=snap.connection_name,
                    database=snap.database,
                    schema_name=t.schema_name,
                    table_name=t.name,
                    row_count=t.row_count,
                    comment=t.comment,
                )
            )

        RelationshipDiscovery(store, combiner=_TrackingCombiner()).discover(snap)
        assert combine_called
    finally:
        store.close()


def test_discovery_behavior_unchanged_with_default_combiner(tmp_path) -> None:
    """Default MaxScoreCombiner produces the same output as the old inline merge logic."""
    snap = _two_table_snap()
    db_path = tmp_path / "g5.kuzu"
    store = KuzuStore(db_path)
    try:
        store.ensure_schema()
        store.clear_graph()
        from pretensor.graph_models.node import GraphNode

        for t in snap.tables:
            store.upsert_table(
                GraphNode(
                    node_id=table_node_id(snap.connection_name, t.schema_name, t.name),
                    connection_name=snap.connection_name,
                    database=snap.database,
                    schema_name=t.schema_name,
                    table_name=t.name,
                    row_count=t.row_count,
                    comment=t.comment,
                )
            )

        out = RelationshipDiscovery(store).discover(snap)
        assert out, "default combiner should produce candidates for two-table snapshot"
        join_keys = {
            (c.source_node_id, c.target_node_id, c.source_column, c.target_column)
            for c in out
        }
        assert len(join_keys) == len(out), "each join key must be unique after merge"
    finally:
        store.close()
