"""Scoring contracts and registry for relationship discovery candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TypeAlias

from pretensor.connectors.models import SchemaSnapshot
from pretensor.graph_models.relationship import RelationshipCandidate

__all__ = ["JoinKey", "RelationshipScorer", "ScorerRegistry"]

JoinKey: TypeAlias = tuple[str, str, str, str]


class RelationshipScorer(ABC):
    """Contract for candidate generators used by relationship discovery."""

    @abstractmethod
    def name(self) -> str:
        """Stable scorer name for observability and deduping."""

    @abstractmethod
    def score(
        self,
        snapshot: SchemaSnapshot,
        explicit_fk_keys: set[JoinKey],
    ) -> list[RelationshipCandidate]:
        """Generate inferred relationship candidates for a snapshot."""


class ScorerRegistry:
    """Ordered registry for relationship scorers.

    The first scorer can emit broad candidates while later scorers can refine or
    add alternatives. Discovery remains deterministic because the insertion order
    is preserved.
    """

    def __init__(self, scorers: Iterable[RelationshipScorer] | None = None) -> None:
        self._scorers_by_name: dict[str, RelationshipScorer] = {}
        self._order: list[str] = []
        for scorer in scorers or ():
            self.register(scorer)

    def register(self, scorer: RelationshipScorer) -> None:
        """Register a scorer instance by unique name."""
        name = scorer.name()
        if name in self._scorers_by_name:
            msg = f"Relationship scorer already registered: {name}"
            raise ValueError(msg)
        self._scorers_by_name[name] = scorer
        self._order.append(name)

    def score_all(
        self,
        snapshot: SchemaSnapshot,
        explicit_fk_keys: set[JoinKey],
    ) -> list[RelationshipCandidate]:
        """Run all scorers in registration order and collect candidates."""
        out: list[RelationshipCandidate] = []
        for name in self._order:
            scorer = self._scorers_by_name[name]
            out.extend(scorer.score(snapshot, explicit_fk_keys))
        return out
