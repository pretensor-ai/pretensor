"""Tests for structured ``connection_unavailable`` envelopes in ``detect_changes``.

First-impression OSS users hit ``detect_changes`` on a box that doesn't have
Snowflake / Postgres creds wired up; the previous payload surfaced the raw
exception string and nothing else, so MCP clients couldn't branch on reason.
These tests assert the envelope carries ``category`` (machine-readable) and
``hint`` (human remediation).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from pretensor.cli.constants import REGISTRY_FILENAME
from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.mcp.tools import detect_changes as dc_mod
from pretensor.mcp.tools.detect_changes import (
    _remediation_hint,
    detect_changes_payload,
)
from pretensor.staleness.snapshot_store import SnapshotStore


def _prep_indexed_state(
    tmp_path: Path,
    *,
    connection_name: str,
    dialect: str,
) -> Path:
    """Create a minimal graph + snapshot + registry entry so detect_changes runs.

    Returns the state_dir. The actual introspection is monkeypatched per test.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    tables = [
        Table(
            name="t",
            schema_name="public",
            columns=[Column(name="id", data_type="int", is_primary_key=True)],
        )
    ]
    snapshot = SchemaSnapshot(
        connection_name=connection_name,
        database=connection_name,
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    SnapshotStore(state_dir).save(connection_name, snapshot)
    graph_path = state_dir / "graphs" / f"{connection_name}.kuzu"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph_path)
    GraphBuilder().build(snapshot, store, run_relationship_discovery=False)
    store.close()
    reg = GraphRegistry(state_dir / REGISTRY_FILENAME).load()
    reg.upsert(
        connection_name=connection_name,
        database=connection_name,
        dsn=f"{dialect}://u@h/{connection_name}",
        graph_path=graph_path,
        dialect=dialect,  # type: ignore[arg-type]
        table_count=len(snapshot.tables),
    )
    reg.save()
    return state_dir


def _force_inspect_failure(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def _boom(_cfg: Any) -> Any:
        raise exc

    monkeypatch.setattr(dc_mod, "inspect", _boom)


def test_oserror_returns_structured_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _prep_indexed_state(tmp_path, connection_name="pg_demo", dialect="postgres")
    _force_inspect_failure(monkeypatch, OSError("connection refused"))

    result = detect_changes_payload(state_dir, database="pg_demo")

    assert result["status"] == "connection_unavailable"
    assert result["category"] == "connection_unavailable"
    assert result["database"] == "pg_demo"
    assert "connection refused" in result["message"]
    # Hint must be actionable — point at the CLI verifier.
    assert "pretensor connect --dry-run" in result["hint"]
    # Provenance so clients can surface "checked at / last indexed".
    assert "last_indexed" in result
    assert "checked_at" in result


def test_generic_exception_wraps_with_introspection_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _prep_indexed_state(tmp_path, connection_name="pg_demo2", dialect="postgres")
    _force_inspect_failure(monkeypatch, RuntimeError("schema fetch blew up"))

    result = detect_changes_payload(state_dir, database="pg_demo2")

    assert result["category"] == "connection_unavailable"
    assert result["message"].startswith("Introspection failed: ")
    assert "schema fetch blew up" in result["message"]
    assert "pretensor connect --dry-run" in result["hint"]


def test_snowflake_hint_mentions_snowflake_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _prep_indexed_state(
        tmp_path, connection_name="sf_demo", dialect="snowflake"
    )
    _force_inspect_failure(monkeypatch, OSError("Snowflake account is required"))

    result = detect_changes_payload(state_dir, database="sf_demo")

    assert result["category"] == "connection_unavailable"
    # The Snowflake-specific hint must name the env vars the OSS user can set.
    hint = result["hint"]
    assert "SNOWFLAKE_ACCOUNT" in hint
    assert "SNOWFLAKE_USER" in hint
    assert "SNOWFLAKE_PASSWORD" in hint


def test_remediation_hint_unknown_dialect_is_generic() -> None:
    hint = _remediation_hint(None)
    assert "pretensor connect --dry-run" in hint
    # No dialect-specific env vars leak into the generic hint.
    assert "SNOWFLAKE" not in hint
