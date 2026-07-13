"""Smoke tests for QueryRunner – direct-access round-trips."""

from __future__ import annotations

from pathlib import Path

from pretensor.core.query_runner import QueryRunner
from pretensor.core.store import KuzuStore


def test_runner_is_composed(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "test.kuzu")
    try:
        assert isinstance(store._runner, QueryRunner)
    finally:
        store.close()


def test_runner_query_all_rows_empty(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "test.kuzu")
    store.ensure_schema()
    try:
        rows = store._runner.query_all_rows("MATCH (t:SchemaTable) RETURN t.node_id")
        assert rows == []
    finally:
        store.close()


def test_runner_execute_write_no_error(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "test.kuzu")
    store.ensure_schema()
    try:
        store._runner.execute_write("MATCH (t:SchemaTable) WHERE false DELETE t")
    finally:
        store.close()


def test_facade_query_all_rows_delegates_to_runner(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "test.kuzu")
    store.ensure_schema()
    try:
        facade_rows = store.query_all_rows("MATCH (t:SchemaTable) RETURN t.node_id")
        runner_rows = store._runner.query_all_rows(
            "MATCH (t:SchemaTable) RETURN t.node_id"
        )
        assert facade_rows == runner_rows
    finally:
        store.close()
