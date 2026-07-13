"""Tests for GraphSchemaManager (schema creation and idempotent column migration)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pretensor.core.graph_schema_manager import GraphSchemaManager
from pretensor.core.store import KuzuStore


def test_ensure_schema_fresh_succeeds(tmp_path: Path) -> None:
    """A fresh KuzuStore.ensure_schema() completes without error."""
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
    finally:
        store.close()


def test_ensure_schema_idempotent(tmp_path: Path) -> None:
    """Calling ensure_schema() twice must not raise (exercises the 'already has' path)."""
    store = KuzuStore(tmp_path / "g2.kuzu")
    try:
        store.ensure_schema()
        store.ensure_schema()
    finally:
        store.close()


def test_add_column_if_missing_swallows_already_has(tmp_path: Path) -> None:
    """_add_column_if_missing silently ignores the 'already has' RuntimeError."""
    store = KuzuStore(tmp_path / "g3.kuzu")
    try:
        store.ensure_schema()
        mgr = store._schema
        mgr._add_column_if_missing("SchemaTable", "entity_type", "STRING")
    finally:
        store.close()


def test_add_column_if_missing_logs_unexpected_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_add_column_if_missing logs at WARNING for unexpected RuntimeErrors."""
    conn = MagicMock()
    conn.execute.side_effect = RuntimeError("some unexpected error from kuzu")
    mgr = GraphSchemaManager(conn)
    with caplog.at_level("WARNING", logger="pretensor.core.graph_schema_manager"):
        mgr._add_column_if_missing("SomeTable", "some_col", "STRING")
    assert "SomeTable" in caplog.text or "some_col" in caplog.text


def test_drop_column_if_present_swallows_not_present(tmp_path: Path) -> None:
    """_drop_column_if_present silently ignores missing columns."""
    store = KuzuStore(tmp_path / "g4.kuzu")
    try:
        store.ensure_schema()
        mgr = store._schema
        mgr._drop_column_if_present("SchemaTable", "nonexistent_column_xyz")
    finally:
        store.close()
