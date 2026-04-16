"""Confidence combiner contracts for relationship discovery."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.scoring import JoinKey

__all__ = ["ConfidenceCombiner", "MaxScoreCombiner"]


class ConfidenceCombiner(ABC):
    """Contract for merging relationship candidate groups into a single deduplicated list."""

    @abstractmethod
    def combine(self, *groups: list[RelationshipCandidate]) -> list[RelationshipCandidate]:
        """Merge one or more candidate groups into a single deduplicated list."""


class MaxScoreCombiner(ConfidenceCombiner):
    """Keeps the highest-confidence candidate per directed column pair."""

    def combine(self, *groups: list[RelationshipCandidate]) -> list[RelationshipCandidate]:
        """Return one candidate per JoinKey, keeping the one with the highest confidence."""
        best: dict[JoinKey, RelationshipCandidate] = {}
        for group in groups:
            for c in group:
                k: JoinKey = (c.source_node_id, c.target_node_id, c.source_column, c.target_column)
                if k not in best or c.confidence > best[k].confidence:
                    best[k] = c
        return list(best.values())
