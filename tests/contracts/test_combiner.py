"""Contract tests for ConfidenceCombiner extension point.

``CombinerContractTest`` verifies that any :class:`~pretensor.intelligence.combining.ConfidenceCombiner`
implementation is a valid drop-in replacement for the OSS default.  Cloud
implementations import this class and bind ``make_combiner`` to their own factory.
"""

from __future__ import annotations

import abc

from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.combining import ConfidenceCombiner, MaxScoreCombiner

# ---------------------------------------------------------------------------
# Shared builder helpers
# ---------------------------------------------------------------------------


def _candidate(
    candidate_id: str,
    src_node: str,
    tgt_node: str,
    src_col: str,
    tgt_col: str,
    confidence: float,
) -> RelationshipCandidate:
    return RelationshipCandidate(
        candidate_id=candidate_id,
        source_node_id=src_node,
        target_node_id=tgt_node,
        source_column=src_col,
        target_column=tgt_col,
        source="heuristic",
        reasoning="test",
        confidence=confidence,
        status="suggested",
    )


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


class CombinerContractTest(abc.ABC):
    """Reusable contract suite for :class:`ConfidenceCombiner` implementations.

    Subclass and implement :meth:`make_combiner` to verify any combiner.

    Example (Cloud)::

        class TestMyCloudCombiner(CombinerContractTest):
            def make_combiner(self) -> ConfidenceCombiner:
                return MyCloudCombiner()
    """

    @abc.abstractmethod
    def make_combiner(self) -> ConfidenceCombiner:
        """Return an instance of the combiner under test."""

    # -- Interface shape -------------------------------------------------------

    def test_is_confidence_combiner_instance(self) -> None:
        """Combiner must be a ConfidenceCombiner subclass."""
        combiner = self.make_combiner()
        assert isinstance(combiner, ConfidenceCombiner)

    # -- combine() return type -------------------------------------------------

    def test_combine_no_groups_returns_list(self) -> None:
        """combine() with zero groups must return a list."""
        combiner = self.make_combiner()
        result = combiner.combine()
        assert isinstance(result, list)

    def test_combine_empty_group_returns_list(self) -> None:
        """combine() with a single empty group must return a list."""
        result = self.make_combiner().combine([])
        assert isinstance(result, list)

    def test_combine_single_group_returns_candidates(self) -> None:
        """combine() with one group must return RelationshipCandidates."""
        c = _candidate("c1", "n1", "n2", "col_a", "id", 0.7)
        result = self.make_combiner().combine([c])
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, RelationshipCandidate)

    def test_combine_two_groups_returns_list(self) -> None:
        """combine() with two groups must return a list."""
        g1 = [_candidate("c1", "n1", "n2", "col_a", "id", 0.7)]
        g2 = [_candidate("c2", "n3", "n4", "col_b", "id", 0.5)]
        result = self.make_combiner().combine(g1, g2)
        assert isinstance(result, list)

    # -- combine() semantics ---------------------------------------------------

    def test_combine_deduplicates_identical_join_key(self) -> None:
        """Duplicates on the same join key must produce exactly one output candidate."""
        dup1 = _candidate("a1", "n1", "n2", "col_x", "id", 0.4)
        dup2 = _candidate("a2", "n1", "n2", "col_x", "id", 0.9)
        result = self.make_combiner().combine([dup1], [dup2])
        matching = [
            c
            for c in result
            if c.source_node_id == "n1"
            and c.target_node_id == "n2"
            and c.source_column == "col_x"
            and c.target_column == "id"
        ]
        assert len(matching) == 1, (
            f"Expected 1 deduplicated candidate, got {len(matching)}"
        )

    def test_combine_keeps_distinct_join_keys(self) -> None:
        """Candidates with different join keys must all appear in the output."""
        c1 = _candidate("x1", "n1", "n2", "col_a", "id", 0.6)
        c2 = _candidate("x2", "n3", "n4", "col_b", "id", 0.8)
        result = self.make_combiner().combine([c1, c2])
        assert len(result) >= 2

    def test_combine_preserves_candidate_fields(self) -> None:
        """Output candidates must have non-empty required fields."""
        c = _candidate("q1", "node_src", "node_tgt", "src_col", "tgt_col", 0.5)
        result = self.make_combiner().combine([c])
        assert result, "expected at least one output candidate"
        out = result[0]
        assert out.source_node_id
        assert out.target_node_id
        assert out.source_column
        assert out.target_column
        assert 0.0 <= out.confidence <= 1.0

    def test_combine_multiple_empty_groups_returns_empty(self) -> None:
        """Combining several empty groups must return an empty list."""
        result = self.make_combiner().combine([], [], [])
        assert result == []


# ---------------------------------------------------------------------------
# Concrete: MaxScoreCombiner
# ---------------------------------------------------------------------------


class TestMaxScoreCombinerContract(CombinerContractTest):
    """Run the full combiner contract against :class:`MaxScoreCombiner`."""

    def make_combiner(self) -> ConfidenceCombiner:
        return MaxScoreCombiner()

    # -- MaxScoreCombiner-specific behaviour -----------------------------------

    def test_keeps_highest_confidence_for_duplicate_key(self) -> None:
        """MaxScoreCombiner must keep the candidate with the highest confidence."""
        low = _candidate("low", "n1", "n2", "col_x", "id", 0.3)
        high = _candidate("high", "n1", "n2", "col_x", "id", 0.9)
        result = self.make_combiner().combine([low], [high])
        matching = [
            c
            for c in result
            if c.source_node_id == "n1"
            and c.target_node_id == "n2"
            and c.source_column == "col_x"
        ]
        assert len(matching) == 1
        assert matching[0].confidence == 0.9

    def test_single_candidate_passes_through(self) -> None:
        """A single candidate group with one item must produce that candidate."""
        c = _candidate("only", "n1", "n2", "col_a", "id", 0.55)
        result = self.make_combiner().combine([c])
        assert len(result) == 1
        assert result[0].confidence == 0.55

    def test_order_of_groups_does_not_affect_winner(self) -> None:
        """Whichever group is listed first, the highest confidence wins."""
        high = _candidate("h", "n1", "n2", "col_x", "id", 0.9)
        low = _candidate("l", "n1", "n2", "col_x", "id", 0.2)
        result_a = self.make_combiner().combine([high], [low])
        result_b = self.make_combiner().combine([low], [high])
        assert result_a[0].confidence == result_b[0].confidence == 0.9
