"""Tests for graph Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pretensor.graph_models import GraphEdge, GraphNode, RelationshipCandidate


def test_graph_node_immutable() -> None:
    node = GraphNode(
        node_id="a::s::t",
        connection_name="a",
        database="db",
        schema_name="s",
        table_name="t",
    )
    with pytest.raises(ValidationError):
        node.connection_name = "b"  # type: ignore[misc]


def test_relationship_candidate_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        RelationshipCandidate(
            candidate_id="1",
            source_node_id="a",
            target_node_id="b",
            source_column="x",
            target_column="y",
            source="heuristic",
            confidence=2.0,
        )


def test_graph_edge_fields() -> None:
    e = GraphEdge(
        edge_id="e1",
        source_node_id="a",
        target_node_id="b",
        source_column="x",
        target_column="y",
    )
    assert e.edge_id == "e1"
