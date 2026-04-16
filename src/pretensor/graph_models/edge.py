"""Graph edge payloads for FK relationships and table-level lineage."""

from __future__ import annotations

from pydantic import Field

from pretensor.graph_models.base import GraphModel

__all__ = ["GraphEdge", "LineageEdge"]


class GraphEdge(GraphModel):
    """A directed foreign-key edge between two table nodes."""

    edge_id: str = Field(
        description="Stable primary key in the Kuzu FK edge rel table."
    )
    source_node_id: str
    target_node_id: str
    source_column: str
    target_column: str
    constraint_name: str | None = None


class LineageEdge(GraphModel):
    """Directed lineage from a source table to a derived or dependent table."""

    edge_id: str = Field(description="Stable primary key in the Kuzu LINEAGE rel table.")
    source_node_id: str
    target_node_id: str
    source: str = Field(
        description="Provenance label (e.g. connector + object name), not SQL text."
    )
    lineage_type: str
    confidence: float = 1.0
