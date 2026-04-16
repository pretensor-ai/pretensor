"""CLI tests for ``pretensor validate``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from pretensor.cli.main import app
from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore


def _setup(tmp_path: Path) -> None:
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
                columns=[
                    Column(name="id", data_type="int", is_primary_key=True)
                ],
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


def test_validate_cli_valid(tmp_path: Path) -> None:
    _setup(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "SELECT o.id FROM public.orders o "
            "JOIN public.users u ON o.user_id = u.id",
            "--db",
            "demo",
            "--state-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["valid"] is True


def test_validate_cli_invalid_exit_code(tmp_path: Path) -> None:
    _setup(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "SELECT * FROM public.nope",
            "--db",
            "demo",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "public.nope" in result.stdout


def test_validate_cli_reads_file(tmp_path: Path) -> None:
    _setup(tmp_path)
    sql_path = tmp_path / "q.sql"
    sql_path.write_text(
        "SELECT o.id FROM public.orders o "
        "JOIN public.users u ON o.user_id = u.id",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        [
            "validate",
            str(sql_path),
            "--db",
            "demo",
            "--state-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["valid"] is True


def test_validate_cli_unknown_db(tmp_path: Path) -> None:
    _setup(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "SELECT 1",
            "--db",
            "nope",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2


def test_validate_cli_help() -> None:
    result = CliRunner().invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
