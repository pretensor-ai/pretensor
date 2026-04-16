"""Unit tests for cluster schema pattern detection."""

from __future__ import annotations

from typing import cast

from pretensor.entities.classifier import TableClassification, TableRole
from pretensor.intelligence.clustering import Cluster
from pretensor.intelligence.schema_classification import compute_cluster_schema_patterns


def _roles(**kwargs: str) -> dict[str, TableClassification]:
    return {
        k: TableClassification(cast(TableRole, v), 0.9, ()) for k, v in kwargs.items()
    }


def test_star_direct_three_dimensions() -> None:
    f, d1, d2, d3 = "f", "d1", "d2", "d3"
    clusters = [Cluster(table_ids=[f, d1, d2, d3], cohesion_score=0.5)]
    fk = [(f, d1), (f, d2), (f, d3)]
    rb = _roles(f="fact", d1="dimension", d2="dimension", d3="dimension")
    p = compute_cluster_schema_patterns(clusters, rb, fk)
    assert p[0] == "star"


def test_constellation_shared_dimension_direct() -> None:
    f1, f2, d = "f1", "f2", "d"
    clusters = [Cluster(table_ids=[f1, f2, d], cohesion_score=0.5)]
    fk = [(f1, d), (f2, d)]
    rb = _roles(f1="fact", f2="fact", d="dimension")
    p = compute_cluster_schema_patterns(clusters, rb, fk)
    assert p[0] == "constellation"


def test_constellation_shared_dimension_via_bridge() -> None:
    f1, f2, b, d = "f1", "f2", "b", "d"
    clusters = [Cluster(table_ids=[f1, f2, b, d], cohesion_score=0.5)]
    fk = [(f1, b), (f2, b), (b, d)]
    rb = _roles(f1="fact", f2="fact", b="bridge", d="dimension")
    p = compute_cluster_schema_patterns(clusters, rb, fk)
    assert p[0] == "constellation"


def test_snowflake_dim_to_dim_chain_with_star_hub() -> None:
    f, d1, d2, d3 = "f", "d1", "d2", "d3"
    clusters = [Cluster(table_ids=[f, d1, d2, d3], cohesion_score=0.5)]
    fk = [(f, d1), (f, d2), (f, d3), (d1, d2)]
    rb = _roles(f="fact", d1="dimension", d2="dimension", d3="dimension")
    p = compute_cluster_schema_patterns(clusters, rb, fk)
    assert p[0] == "snowflake"


def test_role_only_star_when_no_fk() -> None:
    f, d1, d2, d3 = "f", "d1", "d2", "d3"
    clusters = [Cluster(table_ids=[f, d1, d2, d3], cohesion_score=0.5)]
    rb = _roles(f="fact", d1="dimension", d2="dimension", d3="dimension")
    p = compute_cluster_schema_patterns(clusters, rb, [])
    assert p[0] == "star"


def test_erd_two_connected_non_dimensional() -> None:
    a, b = "a", "b"
    clusters = [Cluster(table_ids=[a, b], cohesion_score=0.5)]
    fk = [(a, b)]
    rb = _roles(a="entity_candidate", b="entity_candidate")
    p = compute_cluster_schema_patterns(clusters, rb, fk)
    assert p[0] == "erd"
