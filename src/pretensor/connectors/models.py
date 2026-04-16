"""Schema and connection DTOs for database introspection (graph layer)."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from typing import Any

from pydantic import Field
from ruamel.yaml import YAML

from pretensor.introspection.models.base import PretensorModel
from pretensor.introspection.models.config import ConnectionConfig

__all__ = [
    "Column",
    "ConnectionConfig",
    "ForeignKey",
    "SchemaSnapshot",
    "Table",
    "ViewDependency",
]


class Column(PretensorModel):
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    default_value: str | None = None
    comment: str | None = None
    is_indexed: bool = False
    parent_column: str | None = None
    is_array: bool = False
    check_constraints: list[str] = Field(default_factory=list)
    ordinal_position: int | None = None

    distinct_count: int | None = None
    min_value: str | None = None
    max_value: str | None = None
    null_percentage: float | None = None
    sample_distinct_values: list[str] | None = None
    most_common_values: list[str] | None = None
    histogram_bounds: list[str] | None = None
    stats_correlation: float | None = None
    column_cardinality: int | None = None
    index_type: str | None = None
    index_is_unique: bool | None = None


class ForeignKey(PretensorModel):
    constraint_name: str | None = None
    source_schema: str
    source_table: str
    source_column: str
    target_schema: str
    target_table: str
    target_column: str


class ViewDependency(PretensorModel):
    """Table-level lineage: source table feeds into target (e.g. view reads source)."""

    source_schema: str
    source_table: str
    target_schema: str
    target_table: str
    lineage_type: str
    object_name: str
    confidence: float = 1.0


class Table(PretensorModel):
    name: str
    schema_name: str
    table_type: str | None = None
    columns: list[Column] = Field(default_factory=list)
    row_count: int | None = None
    row_count_source: str | None = None
    comment: str | None = None
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    seq_scan_count: int | None = None
    idx_scan_count: int | None = None
    insert_count: int | None = None
    update_count: int | None = None
    delete_count: int | None = None
    is_partitioned: bool | None = None
    partition_key: str | None = None
    grants: list[dict[str, Any]] | None = None
    access_read_count: int | None = None
    access_write_count: int | None = None
    days_since_last_access: int | None = None
    potentially_unused: bool | None = None
    table_bytes: int | None = None
    clustering_key: str | None = None


class SchemaSnapshot(PretensorModel):
    connection_name: str
    database: str
    schemas: list[str]
    tables: list[Table]
    introspected_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    view_dependencies: list[ViewDependency] = Field(default_factory=list)

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json")
        yaml = YAML()
        yaml.default_flow_style = False
        buf = StringIO()
        yaml.dump(data, buf)
        return buf.getvalue()

    @classmethod
    def from_yaml(cls, yaml_str: str) -> SchemaSnapshot:
        yaml = YAML()
        data = yaml.load(yaml_str)
        return cls.model_validate(data)

    @classmethod
    def empty(cls, connection_name: str, database: str) -> SchemaSnapshot:
        return cls(
            connection_name=connection_name,
            database=database,
            schemas=[],
            tables=[],
            introspected_at=datetime.now(timezone.utc),
        )
