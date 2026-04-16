"""Business entity node payload for Kuzu."""

from __future__ import annotations

from pydantic import Field

from pretensor.graph_models.base import GraphModel

__all__ = ["EntityNode"]


class EntityNode(GraphModel):
    """A named business entity linked to one or more ``SchemaTable`` nodes."""

    node_id: str = Field(
        description="Stable primary key in the Kuzu Entity node table."
    )
    connection_name: str
    database: str
    name: str
    description: str = ""
