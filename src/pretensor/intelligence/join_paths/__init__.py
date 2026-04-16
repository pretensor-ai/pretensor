"""Precompute and look up join paths between physical tables."""

from __future__ import annotations

import logging

from pretensor.config import GraphConfig
from pretensor.core.store import KuzuStore
from pretensor.intelligence.join_paths import on_demand as _od
from pretensor.intelligence.join_paths import storage as _storage

logger = logging.getLogger(__name__)

JoinStep = _od.JoinStep
StoredJoinPath = _od.StoredJoinPath


class JoinPathEngine:
    """BFS/DFS path finding and persistence as ``JoinPath`` nodes."""

    def __init__(self, store: KuzuStore) -> None:
        self._store = store

    def precompute(self, database_key: str, config: GraphConfig | None = None) -> int:
        """Store join paths for intra-cluster pairs and short cross-cluster pairs.

        Args:
            database_key: ``SchemaTable.database`` value.
            config: Depth limits; defaults to :class:`GraphConfig`.

        Returns:
            Number of ``JoinPath`` rows written.
        """
        cfg = config or GraphConfig()
        intra_max = cfg.join_path_max_depth
        # Cross-cluster pairs previously capped at depth 2 to bound precompute
        # cost on wide schemas. That cap hid the authoritative FK chain for
        # cross-domain pairs whose FK path is ≥3 hops (e.g. pagila
        # ``film → inventory → rental → customer``), letting a 2-hop inferred
        # shortcut win as the stored winner. Match ``intra_max`` so the cost
        # model (which already penalises inferred hops) decides the winner
        # rather than a depth cap. If a future benchmark shows precompute
        # regressions on 1000+ table schemas, introduce a dedicated
        # ``join_path_cross_max_depth`` config knob.
        cross_max = intra_max

        table_meta = _od.table_meta(self._store, database_key)
        if not table_meta:
            return 0
        adj = _od.build_adjacency(self._store, database_key)
        cluster_of = _od.table_to_cluster(self._store, database_key)

        written = 0
        ids = list(table_meta.keys())
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                ca = cluster_of.get(a)
                cb = cluster_of.get(b)
                max_d = intra_max if ca is not None and ca == cb else cross_max
                rec = _od.best_path(adj, table_meta, a, b, max_d)
                if rec is None:
                    continue
                _storage.persist_path(self._store, database_key, rec)
                written += 1
                rev = _od.reverse_stored_path(rec)
                _storage.persist_path(self._store, database_key, rev)
                written += 1
        logger.info(
            "Join path precompute for %s: %d paths (intra depth=%d, cross depth=%d)",
            database_key,
            written,
            intra_max,
            cross_max,
        )
        return written

    def load_stored_paths(
        self,
        database_key: str,
        from_table_id: str,
        to_table_id: str,
    ) -> list[StoredJoinPath]:
        """Return precomputed paths for an ordered table pair."""
        return _storage.load_stored_paths(
            self._store, database_key, from_table_id, to_table_id
        )

    def find_path_on_demand(
        self,
        database_key: str,
        from_table_id: str,
        to_table_id: str,
        max_depth: int,
    ) -> StoredJoinPath | None:
        """Compute a single best path without reading ``JoinPath`` nodes."""
        table_meta = _od.table_meta(self._store, database_key)
        if from_table_id not in table_meta or to_table_id not in table_meta:
            return None
        adj = _od.build_adjacency(self._store, database_key)
        return _od.best_path(adj, table_meta, from_table_id, to_table_id, max_depth)

    def find_paths_on_demand(
        self,
        database_key: str,
        from_table_id: str,
        to_table_id: str,
        max_depth: int,
        *,
        top_k: int = 3,
        edge_kinds: tuple[_od.EdgeKind, ...] | None = None,
        max_inferred_hops: int = 2,
    ) -> list[StoredJoinPath]:
        """Compute up to ``top_k`` shortest paths without reading ``JoinPath`` nodes."""
        table_meta = _od.table_meta(self._store, database_key)
        if from_table_id not in table_meta or to_table_id not in table_meta:
            return []
        adj = _od.build_adjacency(self._store, database_key)
        return _od.best_paths(
            adj,
            table_meta,
            from_table_id,
            to_table_id,
            max_depth,
            top_k=top_k,
            edge_kinds=edge_kinds,
            max_inferred_hops=max_inferred_hops,
        )


__all__ = ["JoinPathEngine", "JoinStep", "StoredJoinPath"]
