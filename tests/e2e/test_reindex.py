"""E2E tests — schema drift detection (ALTER TABLE / CREATE TABLE)."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
import pytest

from pretensor.cli.constants import REGISTRY_FILENAME
from pretensor.connectors.inspect import inspect
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.introspection.models.dsn import connection_config_from_url
from pretensor.mcp.tools.detect_changes import detect_changes_payload
from pretensor.staleness.snapshot_store import SnapshotStore

if not os.getenv("PRETENSOR_E2E"):
    pytest.skip("set PRETENSOR_E2E=1", allow_module_level=True)

pytestmark = pytest.mark.e2e


def _build_isolated_index(pagila_dsn: str, state_dir: Path, connection_name: str) -> None:
    """Build a fresh index in state_dir using the given connection_name."""
    cfg = connection_config_from_url(pagila_dsn, connection_name)
    snapshot = inspect(cfg)
    SnapshotStore(state_dir).save(connection_name, snapshot)
    graph_path = state_dir / "graphs" / f"{connection_name}.kuzu"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph_path)
    GraphBuilder().build(snapshot, store, run_relationship_discovery=False)
    store.close()
    registry = GraphRegistry(state_dir / REGISTRY_FILENAME).load()
    registry.upsert(
        connection_name=connection_name,
        database=snapshot.database,
        dsn=pagila_dsn,
        graph_path=graph_path,
        table_count=len(snapshot.tables),
    )
    registry.save()


def test_detect_changes_finds_added_column(pagila_dsn: str, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _build_isolated_index(pagila_dsn, state_dir, "pagila_drift_col")

    conn = psycopg2.connect(pagila_dsn)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute(
            "ALTER TABLE film ADD COLUMN e2e_test_marker VARCHAR(64) DEFAULT NULL"
        )
        result = detect_changes_payload(state_dir, database="pagila_drift_col")
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        changes = result.get("changes", [])
        added = [
            c for c in changes
            if c.get("column") == "e2e_test_marker" or "e2e_test_marker" in str(c)
        ]
        assert len(added) >= 1, (
            f"Expected change for 'e2e_test_marker'; got changes: {changes}"
        )
    finally:
        cur.execute("ALTER TABLE film DROP COLUMN IF EXISTS e2e_test_marker")
        cur.close()
        conn.close()


def test_detect_changes_finds_added_table(pagila_dsn: str, tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _build_isolated_index(pagila_dsn, state_dir, "pagila_drift_tbl")

    conn = psycopg2.connect(pagila_dsn)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute(
            "CREATE TABLE public.e2e_test_events "
            "(id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
        )
        result = detect_changes_payload(state_dir, database="pagila_drift_tbl")
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        changes = result.get("changes", [])
        added = [
            c for c in changes
            if "e2e_test_events" in str(c)
        ]
        assert len(added) >= 1, (
            f"Expected change for 'e2e_test_events'; got changes: {changes}"
        )
    finally:
        cur.execute("DROP TABLE IF EXISTS public.e2e_test_events")
        cur.close()
        conn.close()
