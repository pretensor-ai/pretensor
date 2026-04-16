"""Orchestrate heuristic → statistical relationship discovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pretensor.connectors.models import SchemaSnapshot
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.combining import ConfidenceCombiner, MaxScoreCombiner
from pretensor.intelligence.heuristic import HeuristicScorer
from pretensor.intelligence.scoring import JoinKey, ScorerRegistry
from pretensor.intelligence.shadow_alias import get_shadow_alias_node_ids
from pretensor.intelligence.statistical import apply_statistical_adjustment

if TYPE_CHECKING:
    from pretensor.config import GraphConfig, PretensorConfig

logger = logging.getLogger(__name__)

__all__ = ["RelationshipDiscovery", "explicit_fk_join_keys"]


def explicit_fk_join_keys(snapshot: SchemaSnapshot) -> set[JoinKey]:
    """Set of (source_node_id, target_node_id, source_column, target_column) for declared FKs."""
    conn = snapshot.connection_name
    keys: set[JoinKey] = set()
    for table in snapshot.tables:
        for fk in table.foreign_keys:
            src_id = table_node_id(conn, fk.source_schema, fk.source_table)
            dst_id = table_node_id(conn, fk.target_schema, fk.target_table)
            keys.add((src_id, dst_id, fk.source_column, fk.target_column))
    return keys


def _join_key(c: RelationshipCandidate) -> JoinKey:
    return (
        c.source_node_id,
        c.target_node_id,
        c.source_column,
        c.target_column,
    )


class RelationshipDiscovery:
    """Runs discovery stages and writes ``INFERRED_JOIN`` edges to ``KuzuStore``."""

    def __init__(
        self,
        store: KuzuStore,
        *,
        scorers: ScorerRegistry | None = None,
        combiner: ConfidenceCombiner | None = None,
        config: PretensorConfig | None = None,
        graph_config: GraphConfig | None = None,
    ) -> None:
        self._store = store
        self._graph_config = graph_config
        if config is not None:
            self._scorers = scorers or config.scorer_registry
            self._combiner = combiner or config.combiner
            if graph_config is None:
                self._graph_config = config.graph
        else:
            self._scorers = scorers or ScorerRegistry([HeuristicScorer()])
            self._combiner = combiner or MaxScoreCombiner()

    def discover(
        self,
        snapshot: SchemaSnapshot,
        *,
        overlap_scores: dict[JoinKey, float] | None = None,
    ) -> list[RelationshipCandidate]:
        """Return merged inferred candidates (also written to the store)."""
        explicit = explicit_fk_join_keys(snapshot)

        scored = self._scorers.score_all(snapshot, explicit)

        merged = self._combiner.combine(scored)
        merged = [c for c in merged if _join_key(c) not in explicit]

        # Filter out candidates where source or target is a shadow alias.
        shadow_ids = get_shadow_alias_node_ids(
            self._store,
            snapshot.database,
            config=self._graph_config,
        )
        if shadow_ids:
            before = len(merged)
            merged = [
                c
                for c in merged
                if c.source_node_id not in shadow_ids
                and c.target_node_id not in shadow_ids
            ]
            dropped = before - len(merged)
            if dropped:
                logger.debug(
                    "Dropped %d INFERRED_JOIN candidates involving shadow aliases",
                    dropped,
                )

        overlap = overlap_scores or {}
        adjusted: list[RelationshipCandidate] = []
        for c in merged:
            adj = apply_statistical_adjustment(c, overlap.get(_join_key(c)))
            adjusted.append(adj)

        for c in adjusted:
            self._store.upsert_inferred_join(c)

        return adjusted
