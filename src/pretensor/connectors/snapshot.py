"""Snapshot diff — compare two schema snapshots for drift detection."""

from __future__ import annotations

from enum import StrEnum

from pretensor.connectors.models import Column, SchemaSnapshot, Table, ViewDependency
from pretensor.introspection.models.base import PretensorModel

__all__ = [
    "ChangeTarget",
    "ChangeType",
    "SchemaChange",
    "diff_snapshots",
]


class ChangeType(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class ChangeTarget(StrEnum):
    TABLE = "table"
    COLUMN = "column"
    LINEAGE = "lineage"


class SchemaChange(PretensorModel):
    """A single difference between two schema snapshots."""

    change_type: ChangeType
    target: ChangeTarget
    table_name: str
    schema_name: str
    column_name: str | None = None
    details: str = ""


def diff_snapshots(old: SchemaSnapshot, new: SchemaSnapshot) -> list[SchemaChange]:
    """Compare two snapshots and return a list of changes.

    Detects added/removed tables and added/removed/modified columns.
    """
    changes: list[SchemaChange] = []

    old_tables = {(t.schema_name, t.name): t for t in old.tables}
    new_tables = {(t.schema_name, t.name): t for t in new.tables}

    old_keys = set(old_tables.keys())
    new_keys = set(new_tables.keys())

    for schema_name, table_name in sorted(new_keys - old_keys):
        changes.append(
            SchemaChange(
                change_type=ChangeType.ADDED,
                target=ChangeTarget.TABLE,
                table_name=table_name,
                schema_name=schema_name,
                details=f"New table with {len(new_tables[(schema_name, table_name)].columns)} columns",
            )
        )

    for schema_name, table_name in sorted(old_keys - new_keys):
        changes.append(
            SchemaChange(
                change_type=ChangeType.REMOVED,
                target=ChangeTarget.TABLE,
                table_name=table_name,
                schema_name=schema_name,
            )
        )

    for key in sorted(old_keys & new_keys):
        schema_name, table_name = key
        old_table = old_tables[key]
        new_table = new_tables[key]

        old_cols = {c.name: c for c in old_table.columns}
        new_cols = {c.name: c for c in new_table.columns}

        for col_name in sorted(set(new_cols) - set(old_cols)):
            col = new_cols[col_name]
            changes.append(
                SchemaChange(
                    change_type=ChangeType.ADDED,
                    target=ChangeTarget.COLUMN,
                    table_name=table_name,
                    schema_name=schema_name,
                    column_name=col_name,
                    details=f"type={col.data_type}, nullable={col.nullable}",
                )
            )

        for col_name in sorted(set(old_cols) - set(new_cols)):
            changes.append(
                SchemaChange(
                    change_type=ChangeType.REMOVED,
                    target=ChangeTarget.COLUMN,
                    table_name=table_name,
                    schema_name=schema_name,
                    column_name=col_name,
                )
            )

        for col_name in sorted(set(old_cols) & set(new_cols)):
            old_col = old_cols[col_name]
            new_col = new_cols[col_name]
            diffs = _column_diffs(old_col, new_col)
            if diffs:
                changes.append(
                    SchemaChange(
                        change_type=ChangeType.MODIFIED,
                        target=ChangeTarget.COLUMN,
                        table_name=table_name,
                        schema_name=schema_name,
                        column_name=col_name,
                        details=", ".join(diffs),
                    )
                )

        table_diffs = _table_diffs(old_table, new_table)
        if table_diffs:
            changes.append(
                SchemaChange(
                    change_type=ChangeType.MODIFIED,
                    target=ChangeTarget.TABLE,
                    table_name=table_name,
                    schema_name=schema_name,
                    details=", ".join(table_diffs),
                )
            )

    changes.extend(_lineage_diffs(old.view_dependencies, new.view_dependencies))

    return changes


def _lineage_key(dep: ViewDependency) -> tuple[str, str, str, str, str]:
    return (
        dep.source_schema,
        dep.source_table,
        dep.target_schema,
        dep.target_table,
        dep.lineage_type,
    )


def _lineage_diffs(
    old_deps: list[ViewDependency], new_deps: list[ViewDependency]
) -> list[SchemaChange]:
    """Detect added and removed lineage edges between two snapshots."""
    old_map = {_lineage_key(d): d for d in old_deps}
    new_map = {_lineage_key(d): d for d in new_deps}
    changes: list[SchemaChange] = []
    for key in sorted(set(new_map) - set(old_map)):
        d = new_map[key]
        changes.append(
            SchemaChange(
                change_type=ChangeType.ADDED,
                target=ChangeTarget.LINEAGE,
                schema_name=d.source_schema,
                table_name=d.source_table,
                details=(
                    f"{d.lineage_type}: {d.source_schema}.{d.source_table}"
                    f" → {d.target_schema}.{d.target_table}"
                ),
            )
        )
    for key in sorted(set(old_map) - set(new_map)):
        d = old_map[key]
        changes.append(
            SchemaChange(
                change_type=ChangeType.REMOVED,
                target=ChangeTarget.LINEAGE,
                schema_name=d.source_schema,
                table_name=d.source_table,
                details=(
                    f"{d.lineage_type}: {d.source_schema}.{d.source_table}"
                    f" → {d.target_schema}.{d.target_table}"
                ),
            )
        )
    return changes


def _column_diffs(old: Column, new: Column) -> list[str]:
    """Compare two Column objects on structural and catalog-stat fields.

    ``parent_column`` and ``is_array`` are intentionally excluded: they are
    set once during initial introspection (e.g. BigQuery nested fields) and
    are not expected to change across re-indexes of the same table.
    """
    structural_fields = (
        "data_type",
        "nullable",
        "is_primary_key",
        "is_foreign_key",
        "default_value",
        "comment",
        "is_indexed",
        "check_constraints",
        "ordinal_position",
        "most_common_values",
        "histogram_bounds",
        "stats_correlation",
    )
    diffs: list[str] = []
    for field in structural_fields:
        old_val = getattr(old, field)
        new_val = getattr(new, field)
        if old_val != new_val:
            diffs.append(f"{field}: {old_val!r} → {new_val!r}")
    return diffs


def _table_diffs(old: Table, new: Table) -> list[str]:
    """Compare table-level metadata (not column sets)."""
    diffs: list[str] = []
    for field in (
        "comment",
        "table_type",
        "row_count",
        "seq_scan_count",
        "idx_scan_count",
        "insert_count",
        "update_count",
        "delete_count",
        "is_partitioned",
        "partition_key",
        "grants",
        "access_read_count",
        "access_write_count",
        "days_since_last_access",
        "potentially_unused",
        "table_bytes",
        "clustering_key",
    ):
        old_val = getattr(old, field)
        new_val = getattr(new, field)
        if old_val != new_val:
            diffs.append(f"{field}: {old_val!r} → {new_val!r}")
    return diffs
