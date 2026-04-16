"""Leiden (or Louvain) community detection on the table graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import igraph as ig

from pretensor.config import GraphConfig

__all__ = ["Cluster", "ClusteringEngine"]

logger = logging.getLogger(__name__)

_LEIDEN_SEED = 42


@dataclass
class Cluster:
    """One detected community of physical tables."""

    table_ids: list[str]
    cohesion_score: float
    label: str | None = None
    singleton: bool = False


@dataclass
class ClusteringEngine:
    """Partition a weighted undirected graph into table clusters."""

    config: GraphConfig = field(default_factory=GraphConfig)

    def cluster(self, g: ig.Graph) -> list[Cluster]:
        """Run community detection and post-process tiny clusters.

        Returns:
            One :class:`Cluster` per community, after merging size-<
            ``min_cluster_size_merge`` groups into neighbors.
        """
        if g.vcount() == 0:
            return []
        if g.ecount() == 0:
            return [
                Cluster(
                    table_ids=[g.vs[i]["node_id"]],
                    cohesion_score=0.0,
                    singleton=True,
                )
                for i in range(g.vcount())
            ]

        membership = self._partition(g)
        groups: dict[int, list[int]] = {}
        for vi, part in enumerate(membership):
            groups.setdefault(int(part), []).append(vi)

        merged = self._merge_small_clusters(g, groups)
        clusters: list[Cluster] = []
        for _pid, members in sorted(merged.items(), key=lambda x: min(x[1])):
            table_ids = [g.vs[i]["node_id"] for i in sorted(members)]
            coh = _cohesion_score(g, members)
            clusters.append(
                Cluster(
                    table_ids=table_ids,
                    cohesion_score=coh,
                    singleton=len(members) == 1 and g.degree(members[0]) == 0,
                )
            )
        return clusters

    def _partition(self, g: ig.Graph) -> list[int]:
        resolution = (
            self.config.clustering_resolution_override
            if self.config.clustering_resolution_override is not None
            else _default_resolution(g.vcount())
        )
        try:
            import leidenalg  # type: ignore[import-untyped]

            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=_LEIDEN_SEED,
            )
            return list(partition.membership)
        except ImportError:
            logger.warning(
                "leidenalg not installed — falling back to igraph Louvain "
                "(no resolution tuning). Install the clustering extra for "
                "better community detection: pip install 'pretensor[clustering]'"
            )
            vc: Any = g.community_multilevel(weights="weight")
            return list(vc.membership)

    def _merge_small_clusters(
        self, g: ig.Graph, groups: dict[int, list[int]]
    ) -> dict[int, list[int]]:
        min_sz = self.config.min_cluster_size_merge
        if min_sz <= 1:
            return groups

        def cluster_weight_to(from_members: list[int], to_pid: int) -> float:
            to_members = groups[to_pid]
            total = 0.0
            for i in from_members:
                for j in to_members:
                    eid = g.get_eid(i, j, directed=False, error=False)
                    if eid >= 0:
                        total += float(g.es[eid]["weight"])
            return total

        changed = True
        while changed:
            changed = False
            small_pids = [pid for pid, mem in groups.items() if len(mem) < min_sz]
            if not small_pids:
                break
            pid = small_pids[0]
            members = groups[pid]
            if len(groups) <= 1:
                break
            candidates: list[tuple[float, int, int]] = []
            for other_pid, other_mem in groups.items():
                if other_pid == pid:
                    continue
                w = cluster_weight_to(members, other_pid)
                candidates.append((w, len(other_mem), other_pid))
            if not candidates:
                break
            candidates.sort(key=lambda t: (-t[0], -t[1], t[2]))
            best_pid = candidates[0][2]
            groups[best_pid].extend(members)
            del groups[pid]
            changed = True
        return groups


def _default_resolution(table_count: int) -> float:
    """Clustering heuristic: target medium-sized communities."""
    return 0.5 + (table_count / 200.0)


def _cohesion_score(g: ig.Graph, members: list[int]) -> float:
    """Ratio of realized internal weight to fully-connected upper bound."""
    k = len(members)
    if k < 2:
        return 0.0
    member_set = set(members)
    internal = 0.0
    for i in members:
        for j in g.neighbors(i):
            if j in member_set and i < j:
                eid = g.get_eid(i, j, directed=False, error=False)
                if eid >= 0:
                    internal += float(g.es[eid]["weight"])
    max_edges = k * (k - 1) / 2.0
    return internal / max_edges if max_edges else 0.0
