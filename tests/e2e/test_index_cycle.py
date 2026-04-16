"""E2E tests — registry, graph file, and Kuzu node/edge counts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pretensor.cli.constants import REGISTRY_FILENAME
from pretensor.core.store import KuzuStore
from pretensor.staleness.snapshot_store import SnapshotStore
from tests.query_helpers import first_cell, single_query_result

if not os.getenv("PRETENSOR_E2E"):
    pytest.skip("set PRETENSOR_E2E=1", allow_module_level=True)

pytestmark = pytest.mark.e2e


def test_registry_has_pagila_entry(indexed_state: Path) -> None:
    reg_path = indexed_state / REGISTRY_FILENAME
    assert reg_path.exists(), "registry.json not found"
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    entries = data.get("entries", {})
    assert "pagila" in entries, f"'pagila' not in registry entries: {list(entries)}"
    entry = entries["pagila"]
    assert entry["table_count"] >= 6, f"Expected ≥6 tables, got {entry['table_count']}"
    assert entry["dialect"] == "postgres", f"Expected dialect=postgres, got {entry['dialect']}"
    last_indexed = entry.get("last_indexed_at", "")
    assert "+" in last_indexed or "Z" in last_indexed, (
        f"last_indexed_at should be timezone-aware: {last_indexed!r}"
    )


def test_graph_file_exists(indexed_state: Path) -> None:
    graph_path = indexed_state / "graphs" / "pagila.kuzu"
    assert graph_path.exists(), f"Graph file missing: {graph_path}"
    assert graph_path.stat().st_size > 0, "Graph file is empty"


def test_kuzu_has_schema_tables(indexed_state: Path) -> None:
    store = KuzuStore(indexed_state / "graphs" / "pagila.kuzu")
    try:
        result = single_query_result(store, "MATCH (t:SchemaTable) RETURN count(*) AS n")
        n = first_cell(result)
    finally:
        store.close()
    assert n >= 6, f"Expected ≥6 SchemaTable nodes, got {n}"


def test_kuzu_has_fk_edges(indexed_state: Path) -> None:
    store = KuzuStore(indexed_state / "graphs" / "pagila.kuzu")
    try:
        result = single_query_result(store, "MATCH ()-[r:FK_REFERENCES]->() RETURN count(*) AS n")
        n = first_cell(result)
    finally:
        store.close()
    assert n >= 4, f"Expected ≥4 FK_REFERENCES edges, got {n}"


def test_kuzu_has_columns(indexed_state: Path) -> None:
    store = KuzuStore(indexed_state / "graphs" / "pagila.kuzu")
    try:
        result = single_query_result(store, "MATCH (c:SchemaColumn) RETURN count(*) AS n")
        n = first_cell(result)
    finally:
        store.close()
    assert n >= 20, f"Expected ≥20 SchemaColumn nodes, got {n}"


def test_snapshot_saved(indexed_state: Path) -> None:
    snapshot = SnapshotStore(indexed_state).load("pagila")
    assert snapshot is not None, "Snapshot for 'pagila' not found"
    assert len(snapshot.tables) >= 6, (
        f"Expected ≥6 tables in snapshot, got {len(snapshot.tables)}"
    )


def test_snapshot_has_known_tables(indexed_state: Path) -> None:
    snapshot = SnapshotStore(indexed_state).load("pagila")
    assert snapshot is not None
    table_names = {t.name for t in snapshot.tables}
    for expected in ("film", "rental", "customer"):
        assert expected in table_names, (
            f"Table '{expected}' missing from snapshot; found: {sorted(table_names)}"
        )
