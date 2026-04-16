"""CLI tests for ``pretensor semantic compile``."""

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


def _setup(tmp_path: Path) -> Path:
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
                    Column(name="amount", data_type="numeric"),
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
    yaml_path = tmp_path / "orders.yaml"
    yaml_path.write_text(
        """
connection_name: demo
domains:
  - name: sales
    description: sales
    entities:
      - name: orders
        description: orders
        source_table: public.orders
        attributes:
          - name: id
            description: pk
            role: identifier
            source_column: id
        metrics:
          - name: total_revenue
            description: sum
            type: sum
            field: amount
""".lstrip(),
        encoding="utf-8",
    )
    return yaml_path


def test_semantic_compile_rich_output(tmp_path: Path) -> None:
    yaml_path = _setup(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "semantic",
            "compile",
            str(yaml_path),
            "--metric",
            "total_revenue",
            "--db",
            "demo",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "total_revenue" in result.stdout
    assert "valid" in result.stdout.lower()


def test_semantic_compile_json_output(tmp_path: Path) -> None:
    yaml_path = _setup(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "semantic",
            "compile",
            str(yaml_path),
            "--metric",
            "total_revenue",
            "--db",
            "demo",
            "--state-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["metric"] == "total_revenue"
    assert payload["valid"] is True
    assert 'SUM("amount")' in payload["sql"]


def test_semantic_compile_unknown_metric(tmp_path: Path) -> None:
    yaml_path = _setup(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "semantic",
            "compile",
            str(yaml_path),
            "--metric",
            "ghost",
            "--db",
            "demo",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "not found" in result.stdout.lower()


def test_semantic_compile_unknown_db(tmp_path: Path) -> None:
    yaml_path = _setup(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "semantic",
            "compile",
            str(yaml_path),
            "--metric",
            "total_revenue",
            "--db",
            "nope",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "no registry entry" in result.stdout.lower()


def test_semantic_compile_help() -> None:
    result = CliRunner().invoke(app, ["semantic", "compile", "--help"])
    assert result.exit_code == 0
