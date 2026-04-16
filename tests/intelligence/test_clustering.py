"""Unit tests for intelligence/clustering.py — ClusteringEngine and cohesion score."""

from __future__ import annotations

import igraph as ig
import pytest

from pretensor.config import GraphConfig
from pretensor.intelligence.clustering import ClusteringEngine, _cohesion_score


def _engine(*, resolution: float | None = None, min_merge: int = 1) -> ClusteringEngine:
    cfg = GraphConfig(
        clustering_resolution_override=resolution,
        min_cluster_size_merge=min_merge,
    )
    return ClusteringEngine(cfg)


def _graph_chain(n: int, weight: float = 1.0) -> ig.Graph:
    """Build a linear chain of n vertices: 0-1-2-...(n-1)."""
    g = ig.Graph(n)
    g.add_edges([(i, i + 1) for i in range(n - 1)])
    g.es["weight"] = [weight] * g.ecount()
    for i in range(n):
        g.vs[i]["node_id"] = f"t{i}"
    return g


def _graph_clique(n: int, weight: float = 1.0) -> ig.Graph:
    """Build a fully-connected graph of n vertices."""
    g = ig.Graph.Full(n)
    g.es["weight"] = [weight] * g.ecount()
    for i in range(n):
        g.vs[i]["node_id"] = f"t{i}"
    return g


def test_empty_graph_returns_empty_list() -> None:
    g = ig.Graph(0)
    clusters = _engine().cluster(g)
    assert clusters == []


def test_no_edges_returns_singletons() -> None:
    g = ig.Graph(3)
    for i in range(3):
        g.vs[i]["node_id"] = f"t{i}"
    clusters = _engine().cluster(g)
    assert len(clusters) == 3
    assert all(c.singleton for c in clusters)
    assert all(c.cohesion_score == 0.0 for c in clusters)


def test_cohesion_score_fully_connected_triangle() -> None:
    members = [0, 1, 2]
    g = _graph_clique(3)
    score = _cohesion_score(g, members)
    assert abs(score - 1.0) < 1e-9


def test_cohesion_score_singleton_is_zero() -> None:
    g = _graph_chain(3)
    assert _cohesion_score(g, [0]) == 0.0


def test_cohesion_score_range() -> None:
    g = _graph_chain(5)
    for i in range(5):
        score = _cohesion_score(g, list(range(i + 1)))
        assert 0.0 <= score <= 1.0


def test_single_cluster_from_chain() -> None:
    g = _graph_chain(4)
    clusters = _engine().cluster(g)
    # All nodes should be covered
    all_ids = {tid for c in clusters for tid in c.table_ids}
    assert all_ids == {"t0", "t1", "t2", "t3"}


def test_small_cluster_merge_absorbs_singleton() -> None:
    # Two groups: a clique of 3 and a single isolated node connected to the clique
    g = ig.Graph(4)
    for i in range(4):
        g.vs[i]["node_id"] = f"t{i}"
    # Clique 0-1-2, node 3 weakly connected to 0
    g.add_edges([(0, 1), (0, 2), (1, 2), (0, 3)])
    g.es["weight"] = [1.0, 1.0, 1.0, 0.1]
    clusters = _engine(min_merge=2).cluster(g)
    # With min_merge=2, singleton t3 should be merged into the bigger cluster
    assert all(len(c.table_ids) >= 2 for c in clusters)


def test_resolution_override_affects_partition() -> None:
    # Louvain fallback (no leidenalg) doesn't support resolution tuning — skip in that case.
    pytest.importorskip("leidenalg", reason="leidenalg required for resolution tuning")
    g = _graph_chain(8)
    low_res = _engine(resolution=0.01).cluster(g)
    high_res = _engine(resolution=10.0).cluster(g)
    assert len(high_res) >= len(low_res)
