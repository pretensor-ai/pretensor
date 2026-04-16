"""Relationship discovery candidates (heuristic, LLM, statistical, explicit FK)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pretensor.graph_models.base import GraphModel

__all__ = ["RelationshipCandidate", "RelationshipSource"]

RelationshipSource = Literal["explicit_fk", "heuristic", "llm_inferred", "statistical"]


class RelationshipCandidate(GraphModel):
    """A directed join hypothesis between two ``SchemaTable`` nodes."""

    candidate_id: str
    source_node_id: str
    target_node_id: str
    source_column: str
    target_column: str
    source: RelationshipSource
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    status: Literal["suggested", "confirmed", "rejected"] = "suggested"
