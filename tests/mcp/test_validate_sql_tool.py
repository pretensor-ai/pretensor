"""Tests for the MCP ``validate_sql`` payload."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.mcp.tools.validate_sql import validate_sql_payload


def _write_registry(tmp_path: Path) -> None:
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[
            Table(
                name="orders",
                schema_name="public",
                columns=[
                    Column(name="id", data_type="int", is_primary_key=True),
                    Column(name="user_id", data_type="int", is_foreign_key=True),
                ],
                foreign_keys=[
                    ForeignKey(
                        source_schema="public",
                        source_table="orders",
                        source_column="user_id",
                        target_schema="public",
                        target_table="users",
                        target_column="id",
                    )
                ],
            ),
            Table(
                name="users",
                schema_name="public",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            ),
        ],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://localhost/demo",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()


def test_validate_sql_valid_sql(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    out = validate_sql_payload(
        tmp_path,
        sql=(
            "SELECT o.id FROM public.orders o "
            "JOIN public.users u ON o.user_id = u.id"
        ),
        database="demo",
    )
    assert out["valid"] is True
    assert out["missing_tables"] == []
    assert out["invalid_joins"] == []


def test_validate_sql_missing_table(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    out = validate_sql_payload(
        tmp_path,
        sql="SELECT * FROM public.nope",
        database="demo",
    )
    assert out["valid"] is False
    assert "public.nope" in out["missing_tables"]


def test_validate_sql_syntax_error(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    out = validate_sql_payload(
        tmp_path,
        sql="SELECT FROM",
        database="demo",
    )
    assert out["valid"] is False
    assert out["syntax_errors"]


def test_validate_sql_unknown_database(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    out = validate_sql_payload(
        tmp_path,
        sql="SELECT 1",
        database="nope",
    )
    assert "error" in out


def test_validate_sql_missing_sql(tmp_path: Path) -> None:
    out = validate_sql_payload(
        tmp_path,
        sql="",
        database="demo",
    )
    assert out.get("error", "").startswith("Missing")
