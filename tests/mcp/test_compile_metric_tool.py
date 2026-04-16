"""Tests for the MCP ``compile_metric`` payload."""

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
from pretensor.mcp.tools.compile_metric import compile_metric_payload


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


_YAML = """
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
""".lstrip()


def test_compile_metric_success(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    out = compile_metric_payload(
        tmp_path,
        semantic_yaml=_YAML,
        metric="total_revenue",
        database="demo",
    )
    assert "error" not in out
    assert out["metric"] == "total_revenue"
    assert out["entity"] == "orders"
    assert 'SUM("amount")' in out["sql"]
    assert out["valid"] is True
    assert out["missing_tables"] == []


def test_compile_metric_missing_yaml(tmp_path: Path) -> None:
    out = compile_metric_payload(
        tmp_path,
        semantic_yaml="",
        metric="x",
        database="demo",
    )
    assert out.get("error", "").startswith("Missing")


def test_compile_metric_unknown_database(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    out = compile_metric_payload(
        tmp_path,
        semantic_yaml=_YAML,
        metric="total_revenue",
        database="nope",
    )
    assert "error" in out
    assert "registry" in out["error"].lower()


def test_compile_metric_missing_metric(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    out = compile_metric_payload(
        tmp_path,
        semantic_yaml=_YAML,
        metric="ghost",
        database="demo",
    )
    assert "error" in out
    assert "not found" in out["error"]


def test_compile_metric_bad_yaml(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    out = compile_metric_payload(
        tmp_path,
        semantic_yaml="not: [valid",
        metric="x",
        database="demo",
    )
    assert "error" in out
    assert "parse" in out["error"].lower()
