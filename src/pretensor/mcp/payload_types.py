"""Typed payloads and small helpers for MCP tool responses."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict

from pretensor.config import GraphConfig
from pretensor.mcp.service_context import get_effective_graph_config

DB_TYPE_POSTGRES = "postgresql"

DIALECT_TO_DB_TYPE: dict[str, str] = {
    "postgres": "postgresql",
    "mysql": "mysql",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
}


class DatabaseListItem(TypedDict, total=False):
    """One row for ``list_databases`` and the databases resource."""

    name: str
    db_type: str
    table_count: int
    column_count: int
    row_count: int
    last_indexed: str
    is_stale: bool
    staleness_days: int
    stale_warning: str
    graph_path: str
    database: str
    schemas: list[str]
    # Trinary capability state: "not_attempted" | "empty" | "present".
    # "not_attempted" means the indexer never ran this pass, so absence of signal
    # is not authoritative. "empty" means the pass ran but found nothing.
    has_dbt_manifest: str
    has_llm_enrichment: str
    has_external_consumers: str
    # Unified capability list derived from the trinary flags plus always-on
    # features. Additive convenience over the individual ``has_*`` fields;
    # scan this to decide whether a graph supports a feature at a glance.
    capabilities: list[str]


class QueryHit(TypedDict, total=False):
    """One search hit for the ``query`` tool."""

    node_type: str
    name: str
    database_name: str
    connection_name: str
    description: str
    snippet: str
    score: float


class ColumnInfo(TypedDict, total=False):
    """Column metadata for ``context`` (from ``SchemaColumn`` nodes when present).

    ``column_name`` is the canonical field.  ``name`` is kept for backward
    compatibility and always carries the same value — it will be removed in a
    future version once callers have migrated to ``column_name``.
    """

    column_name: str
    name: str  # Deprecated: use column_name instead
    data_type: str
    nullable: bool
    is_primary_key: bool
    is_foreign_key: bool
    description: str
    is_indexed: bool
    check_constraints: list[str]
    most_common_values: list[str]
    histogram_bounds: list[str]
    stats_correlation: float


class RelationshipInfo(TypedDict, total=False):
    """FK or inferred join from a table to another table.

    For composite FKs, ``source_columns`` / ``target_columns`` carry the
    ordered column lists and ``constraint_name`` identifies the constraint.
    Single-column FKs still set ``source_column`` / ``target_column`` for
    backward compatibility.
    """

    target_table: str
    rel_type: str
    source_column: str
    target_column: str
    source_columns: list[str]
    target_columns: list[str]
    constraint_name: str | None
    source: str | None
    confidence: float | None
    reasoning: str | None


class LineageRef(TypedDict, total=False):
    """One table-level LINEAGE hop for ``context``."""

    table: str
    lineage_type: str
    confidence: float
    source: str


class ClusterInfo(TypedDict, total=False):
    """Domain cluster assignment for ``context``."""

    cluster_id: str
    label: str
    description: str
    cohesion_score: float
    schema_pattern: str
    stale: bool
    stale_warning: str


class ContextPayload(TypedDict, total=False):
    """Structured ``context`` tool response."""

    connection_name: str
    database: str
    schema_name: str
    table_name: str
    table_type: str | None
    qualified_name: str
    description: str
    tags: list[str]
    has_external_consumers: bool
    test_count: int
    staleness_status: str
    staleness_as_of: str
    row_count: int | None
    entity_type: str | None
    entity_name: str | None
    entity_description: str | None
    role: str | None
    role_confidence: float | None
    classification_signals: list[str]
    classification_summary: str
    cluster: ClusterInfo | None
    columns: list[ColumnInfo]
    relationships: list[RelationshipInfo]
    lineage_in: list[LineageRef]
    lineage_out: list[LineageRef]
    lineage_markdown: str
    aliases: list[str]
    shadow_of: str | None
    deprecation_signal: str | None
    staleness_warning: str | None
    detail: str
    usage_stats: dict[str, int]
    partition: dict[str, object]
    grants: list[dict[str, str]]
    access_patterns: dict[str, object]


class TraverseStepPayload(TypedDict, total=False):
    """One hop in ``traverse`` output."""

    from_table: str
    to_table: str
    from_column: str
    to_column: str
    edge_type: str
    from_db: str
    to_db: str
    via: str
    join_columns: str | None
    constraint_name: str | None


class TraversePathPayload(TypedDict, total=False):
    """One ranked path in ``traverse`` output."""

    confidence: float
    ambiguous: bool
    semantic_label: str
    steps: list[TraverseStepPayload]
    sql_hint: str
    stale: bool
    stale_warning: str


class ImpactItemPayload(TypedDict, total=False):
    """Single dependent in ``impact`` output."""

    type: str
    name: str
    via: str
    confidence: float
    hop: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_format(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def stale_threshold_days(config: GraphConfig | None = None) -> int:
    return get_effective_graph_config(config).stale_index_warning_days


def staleness_days(indexed_at: datetime, now: datetime | None = None) -> int:
    reference = now or utc_now()
    if indexed_at.tzinfo is None:
        indexed_at = indexed_at.replace(tzinfo=timezone.utc)
    delta = reference - indexed_at.astimezone(timezone.utc)
    return max(0, int(delta.total_seconds() // 86400))


def snippet(text: str, max_len: int = 200) -> str:
    t = text.strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


__all__ = [
    "ClusterInfo",
    "ColumnInfo",
    "ContextPayload",
    "DatabaseListItem",
    "ImpactItemPayload",
    "LineageRef",
    "QueryHit",
    "RelationshipInfo",
    "TraversePathPayload",
    "TraverseStepPayload",
    "DB_TYPE_POSTGRES",
    "DIALECT_TO_DB_TYPE",
    "iso_format",
    "snippet",
    "staleness_days",
    "stale_threshold_days",
    "utc_now",
]
