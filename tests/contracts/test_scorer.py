"""Contract tests for RelationshipScorer extension point.

``ScorerContractTest`` is an abstract base class that any implementation of
:class:`~pretensor.intelligence.scoring.RelationshipScorer` must pass.  Cloud
implementations import this class and bind ``make_scorer`` to their own factory.

Concrete tests for the OSS :class:`~pretensor.intelligence.heuristic.HeuristicScorer`
are defined at the bottom of this file.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.heuristic import HeuristicScorer
from pretensor.intelligence.scoring import JoinKey, RelationshipScorer

# ---------------------------------------------------------------------------
# Shared snapshot helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    tables: list[Table],
    *,
    connection_name: str = "test_conn",
    database: str = "test_db",
) -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name=connection_name,
        database=database,
        schemas=["public"],
        tables=tables,
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _simple_table(
    name: str,
    columns: list[Column],
    schema: str = "public",
) -> Table:
    return Table(name=name, schema_name=schema, columns=columns)


def _pk_col(name: str = "id") -> Column:
    return Column(name=name, data_type="int", is_primary_key=True)


def _fk_col(name: str) -> Column:
    return Column(name=name, data_type="int", is_foreign_key=True)


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


class ScorerContractTest(abc.ABC):
    """Reusable contract suite for :class:`RelationshipScorer` implementations.

    Subclass this and implement :meth:`make_scorer` to verify any scorer.

    Example (Cloud)::

        class TestMyCloudScorer(ScorerContractTest):
            def make_scorer(self) -> RelationshipScorer:
                return MyCloudScorer()
    """

    @abc.abstractmethod
    def make_scorer(self) -> RelationshipScorer:
        """Return an instance of the scorer under test."""

    # -- Interface shape -------------------------------------------------------

    def test_is_relationship_scorer_instance(self) -> None:
        """Scorer must be a RelationshipScorer subclass."""
        scorer = self.make_scorer()
        assert isinstance(scorer, RelationshipScorer)

    def test_name_returns_non_empty_string(self) -> None:
        """name() must return a non-empty stable string."""
        scorer = self.make_scorer()
        n = scorer.name()
        assert isinstance(n, str)
        assert len(n) > 0

    def test_name_is_stable(self) -> None:
        """name() must return the same value on repeated calls."""
        scorer = self.make_scorer()
        assert scorer.name() == scorer.name()

    # -- score() return type ---------------------------------------------------

    def test_score_returns_list(self) -> None:
        """score() must return a list."""
        scorer = self.make_scorer()
        snapshot = _make_snapshot([_simple_table("orders", [_pk_col()])])
        result = scorer.score(snapshot, set())
        assert isinstance(result, list)

    def test_score_returns_relationship_candidates(self) -> None:
        """Every element returned must be a RelationshipCandidate."""
        scorer = self.make_scorer()
        tables = [
            _simple_table("orders", [_pk_col(), _fk_col("customer_id")]),
            _simple_table("customers", [_pk_col()]),
        ]
        snapshot = _make_snapshot(tables)
        results = scorer.score(snapshot, set())
        for item in results:
            assert isinstance(item, RelationshipCandidate), (
                f"Expected RelationshipCandidate, got {type(item)}"
            )

    # -- score() edge cases ----------------------------------------------------

    def test_score_empty_snapshot_returns_list(self) -> None:
        """score() on an empty snapshot must return a list (possibly empty)."""
        scorer = self.make_scorer()
        snapshot = _make_snapshot([])
        result = scorer.score(snapshot, set())
        assert isinstance(result, list)

    def test_score_single_table_returns_list(self) -> None:
        """score() with a single table (no pairs) must return a list."""
        scorer = self.make_scorer()
        snapshot = _make_snapshot([_simple_table("solo", [_pk_col()])])
        result = scorer.score(snapshot, set())
        assert isinstance(result, list)

    def test_score_respects_explicit_fk_exclusion(self) -> None:
        """Candidates whose join key is in explicit_fk_keys must be excluded."""
        scorer = self.make_scorer()
        tables = [
            _simple_table("orders", [_pk_col(), _fk_col("customer_id")]),
            _simple_table("customers", [_pk_col()]),
        ]
        snapshot = _make_snapshot(tables)
        no_exclusions = scorer.score(snapshot, set())
        if not no_exclusions:
            pytest.skip("scorer produced no candidates for this schema")

        first = no_exclusions[0]
        explicit: set[JoinKey] = {
            (
                first.source_node_id,
                first.target_node_id,
                first.source_column,
                first.target_column,
            )
        }
        with_exclusions = scorer.score(snapshot, explicit)
        result_keys = {
            (
                c.source_node_id,
                c.target_node_id,
                c.source_column,
                c.target_column,
            )
            for c in with_exclusions
        }
        assert first not in with_exclusions or (
            first.source_node_id,
            first.target_node_id,
            first.source_column,
            first.target_column,
        ) not in result_keys

    def test_candidate_confidence_in_range(self) -> None:
        """Every candidate confidence must be in [0.0, 1.0]."""
        scorer = self.make_scorer()
        tables = [
            _simple_table("orders", [_pk_col(), _fk_col("customer_id")]),
            _simple_table("customers", [_pk_col()]),
        ]
        snapshot = _make_snapshot(tables)
        for candidate in scorer.score(snapshot, set()):
            assert 0.0 <= candidate.confidence <= 1.0, (
                f"confidence {candidate.confidence} out of range for {candidate}"
            )

    def test_candidate_fields_are_non_empty_strings(self) -> None:
        """Each candidate must have non-empty required string fields."""
        scorer = self.make_scorer()
        tables = [
            _simple_table("orders", [_pk_col(), _fk_col("customer_id")]),
            _simple_table("customers", [_pk_col()]),
        ]
        snapshot = _make_snapshot(tables)
        for candidate in scorer.score(snapshot, set()):
            assert candidate.candidate_id, "candidate_id must not be empty"
            assert candidate.source_node_id, "source_node_id must not be empty"
            assert candidate.target_node_id, "target_node_id must not be empty"
            assert candidate.source_column, "source_column must not be empty"
            assert candidate.target_column, "target_column must not be empty"


# ---------------------------------------------------------------------------
# Concrete: HeuristicScorer
# ---------------------------------------------------------------------------


class TestHeuristicScorerContract(ScorerContractTest):
    """Run the full scorer contract against :class:`HeuristicScorer`."""

    def make_scorer(self) -> RelationshipScorer:
        return HeuristicScorer()

    # -- HeuristicScorer-specific behaviour ------------------------------------

    def test_name_is_heuristic(self) -> None:
        assert self.make_scorer().name() == "heuristic"

    def test_suffix_id_column_generates_candidate(self) -> None:
        """A ``*_id`` column naming convention must produce at least one candidate."""
        tables = [
            _simple_table("orders", [_pk_col(), _fk_col("customer_id")]),
            _simple_table("customers", [_pk_col()]),
        ]
        snapshot = _make_snapshot(tables)
        candidates = self.make_scorer().score(snapshot, set())
        source_cols = {c.source_column for c in candidates}
        assert "customer_id" in source_cols

    def test_explicit_fk_removes_candidate(self) -> None:
        """Passing an exact join key into explicit_fk_keys removes that candidate."""
        tables = [
            _simple_table("orders", [_pk_col(), _fk_col("customer_id")]),
            _simple_table("customers", [_pk_col()]),
        ]
        snapshot = _make_snapshot(tables)
        scorer = self.make_scorer()
        all_candidates = scorer.score(snapshot, set())
        customer_id_candidates = [
            c for c in all_candidates if c.source_column == "customer_id"
        ]
        assert customer_id_candidates, "expected customer_id candidate before exclusion"
        c = customer_id_candidates[0]
        explicit: set[JoinKey] = {(c.source_node_id, c.target_node_id, c.source_column, c.target_column)}
        after = scorer.score(snapshot, explicit)
        excluded_keys = {
            (x.source_node_id, x.target_node_id, x.source_column, x.target_column)
            for x in after
        }
        assert (c.source_node_id, c.target_node_id, c.source_column, c.target_column) not in excluded_keys

    def test_no_self_join_candidates(self) -> None:
        """A candidate must not reference the same source and target node."""
        tables = [
            _simple_table("orders", [_pk_col(), _fk_col("customer_id")]),
            _simple_table("customers", [_pk_col()]),
        ]
        snapshot = _make_snapshot(tables)
        for c in self.make_scorer().score(snapshot, set()):
            assert c.source_node_id != c.target_node_id, (
                f"self-join candidate found: {c}"
            )
