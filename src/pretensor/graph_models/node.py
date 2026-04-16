"""Graph node payload used when writing to Kuzu."""

from __future__ import annotations

from pydantic import Field

from pretensor.graph_models.base import GraphModel

__all__ = ["GraphNode"]


class GraphNode(GraphModel):
    """A table (or future entity type) represented as a node in the graph."""

    node_id: str = Field(
        description="Stable primary key in the Kuzu SchemaTable node table."
    )
    connection_name: str
    database: str
    schema_name: str
    table_name: str
    row_count: int | None = None
    row_count_source: str | None = Field(
        default=None,
        description=(
            "Provenance: 'stat', 'view_count', or 'view_timeout' (row_count = -1)."
        ),
    )
    comment: str | None = None
    description: str | None = Field(
        default=None,
        description="Human-authored description (e.g. dbt); distinct from DB comment.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Tag names merged onto the graph node (e.g. dbt tags).",
    )
    table_type: str | None = None
    entity_type: str | None = Field(
        default=None,
        description="Business entity label from extraction (e.g. Customer).",
    )
    seq_scan_count: int | None = None
    idx_scan_count: int | None = None
    insert_count: int | None = None
    update_count: int | None = None
    delete_count: int | None = None
    is_partitioned: bool | None = None
    partition_key: str | None = None
    grants_json: str | None = None
    access_read_count: int | None = None
    access_write_count: int | None = None
    days_since_last_access: int | None = None
    potentially_unused: bool | None = None
    table_bytes: int | None = None
    clustering_key: str | None = None
    role: str | None = Field(
        default=None,
        description="Data modeling role from schema classifier (fact, dimension, …).",
    )
    role_confidence: float | None = Field(
        default=None, description="Classifier confidence in ``[0, 1]``."
    )
    classification_signals: str | None = Field(
        default=None,
        description="JSON array string of human-readable classification signals.",
    )
    has_external_consumers: bool | None = Field(
        default=None,
        description="True when dbt lineage marks downstream exposure outside the project.",
    )
    test_count: int | None = Field(
        default=None,
        description="Number of dbt tests attached to this model (when known).",
    )
