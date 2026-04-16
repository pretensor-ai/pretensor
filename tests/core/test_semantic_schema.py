"""Tests for reserved semantic DDL in KuzuStore.ensure_schema()."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from tests.query_helpers import first_cell, single_query_result

from pretensor.core.store import KuzuStore

_RESERVED_NODE_TABLES = ["Metric", "Dimension", "BusinessRule"]
_RESERVED_EDGE_TABLES = [
    "METRIC_DEPENDS_ON",
    "DIMENSION_LEVEL",
    "RULE_APPLIES_TO",
]


@pytest.fixture()
def store(tmp_path: Path) -> Generator[KuzuStore, None, None]:
    s = KuzuStore(tmp_path / "test.kuzu")
    s.ensure_schema()
    try:
        yield s
    finally:
        s.close()


@pytest.mark.parametrize("table", _RESERVED_NODE_TABLES)
def test_reserved_node_table_exists_and_is_empty(store: KuzuStore, table: str) -> None:
    result = single_query_result(store, f"MATCH (n:{table}) RETURN count(n)")
    count = first_cell(result)
    assert int(count) == 0, f"{table} should be empty in OSS"


@pytest.mark.parametrize("table", _RESERVED_EDGE_TABLES)
def test_reserved_edge_table_exists_and_is_empty(store: KuzuStore, table: str) -> None:
    result = single_query_result(
        store, f"MATCH ()-[r:{table}]->() RETURN count(r)"
    )
    count = first_cell(result)
    assert int(count) == 0, f"{table} edge type should be empty in OSS"


def test_ensure_schema_idempotent(tmp_path: Path) -> None:
    """Calling ensure_schema() twice must not raise."""
    s = KuzuStore(tmp_path / "idem.kuzu")
    try:
        s.ensure_schema()
        s.ensure_schema()
    finally:
        s.close()


def test_existing_graph_operations_unaffected(store: KuzuStore) -> None:
    """Core SchemaTable node creation still works after semantic DDL is added."""
    result = single_query_result(store, "MATCH (t:SchemaTable) RETURN count(t)")
    count = first_cell(result)
    assert int(count) == 0
