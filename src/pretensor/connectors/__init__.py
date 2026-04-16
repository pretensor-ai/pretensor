"""Connector-layer schema types, introspection, and snapshot diffing."""

from pretensor.connectors.inspect import inspect
from pretensor.connectors.models import (
    Column,
    ConnectionConfig,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.connectors.snapshot import (
    ChangeTarget,
    ChangeType,
    SchemaChange,
    diff_snapshots,
)

__all__ = [
    "ChangeTarget",
    "ChangeType",
    "Column",
    "ConnectionConfig",
    "ForeignKey",
    "SchemaChange",
    "SchemaSnapshot",
    "Table",
    "diff_snapshots",
    "inspect",
]
