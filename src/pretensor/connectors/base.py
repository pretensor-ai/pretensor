"""Abstract connector interface for database introspection.

Every database connector implements BaseConnector. All database-specific
imports and logic live inside the concrete connector files — nothing
above the connector layer knows which database produced the data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from pretensor.connectors.models import ViewDependency
from pretensor.introspection.models.base import PretensorModel
from pretensor.introspection.models.config import ConnectionConfig, SchemaFilter


class TableInfo(PretensorModel):
    """Table metadata returned by a connector (no column details)."""

    name: str
    schema_name: str
    row_count: int | None = None
    row_count_source: str | None = Field(
        default=None,
        description=(
            "Provenance of `row_count`: 'stat' (catalog statistic), "
            "'view_count' (live SELECT COUNT(*) on a view), "
            "'view_timeout' (view count exceeded the connector timeout — "
            "row_count = -1)."
        ),
    )
    comment: str | None = None
    table_type: str | None = None


class ColumnInfo(PretensorModel):
    """Column metadata returned by a connector (no stats)."""

    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    default_value: str | None = None
    comment: str | None = None
    is_indexed: bool = False
    check_constraints: list[str] = Field(default_factory=list)
    ordinal_position: int | None = None
    column_cardinality: int | None = None
    index_type: str | None = None
    index_is_unique: bool | None = None
    parent_column: str | None = None
    is_array: bool = False


class ForeignKeyInfo(PretensorModel):
    """Foreign key relationship returned by a connector."""

    constraint_name: str | None = None
    source_schema: str
    source_table: str
    source_column: str
    target_schema: str
    target_table: str
    target_column: str


class ColumnStats(PretensorModel):
    """Statistical metadata for a single column."""

    distinct_count: int | None = None
    min_value: str | None = None
    max_value: str | None = None
    null_percentage: float | None = None
    sample_distinct_values: list[str] | None = None


class TableGrant(PretensorModel):
    """SELECT grant on a table for an effective grantee (role or login)."""

    grantee: str
    schema_name: str
    table_name: str


class BaseConnector(ABC):
    """Abstract interface that every database connector must implement.

    Usage as a context manager is preferred::

        with PostgresConnector(config) as conn:
            tables = conn.get_tables()
    """

    def __init__(self, config: ConnectionConfig) -> None:
        self.config = config

    def __enter__(self) -> BaseConnector:
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disconnect()

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_tables(
        self, schema_filter: SchemaFilter | None = None
    ) -> list[TableInfo]: ...

    @abstractmethod
    def get_columns(self, table_name: str, schema_name: str) -> list[ColumnInfo]: ...

    @abstractmethod
    def get_foreign_keys(self) -> list[ForeignKeyInfo]: ...

    @abstractmethod
    def get_table_row_count(self, table_name: str, schema_name: str) -> int: ...

    @abstractmethod
    def get_column_stats(
        self, table_name: str, column_name: str, schema_name: str
    ) -> ColumnStats: ...

    @abstractmethod
    def execute_query(self, sql: str) -> list[dict[str, Any]]: ...

    def load_deep_catalog(
        self, schema_filter: SchemaFilter
    ) -> tuple[
        dict[tuple[str, str], dict[str, Any]],
        dict[tuple[str, str, str], dict[str, Any]],
    ]:
        """Optional bulk catalog enrichment (usage, stats, grants). Default: none.

        Callers treat this as best-effort: failures must not block indexing (log and
        continue without enrichment).
        """
        _ = schema_filter
        return {}, {}

    def load_view_dependencies(self, schema_filter: SchemaFilter) -> list[ViewDependency]:
        """Table-level lineage from views, triggers, or warehouse objects. Default: none."""
        _ = schema_filter
        return []

    def get_table_grants(
        self, schema_filter: SchemaFilter | None = None
    ) -> list[TableGrant]:
        """SELECT table grants, including connector-specific expansion (e.g. role inheritance).

        Default: no grant introspection (empty list).

        Callers treat this as best-effort: failures must not block indexing (log and
        continue without grants), same contract as :meth:`load_deep_catalog`.
        """
        _ = schema_filter
        return []
