"""Tests for the MCP ``schema`` tool: node/edge catalog introspection."""

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
from pretensor.core.schema import format_catalog_summary
from pretensor.core.store import KuzuStore
from pretensor.mcp.service import schema_payload


def _write_minimal_registry(tmp_path: Path) -> None:
    t_orders = Table(
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
    )
    t_users = Table(
        name="users",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_orders, t_users],
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


def test_schema_returns_core_node_labels(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = schema_payload(tmp_path, database="demo")
    assert "error" not in out
    node_labels = {n["label"] for n in out["nodes"]}
    assert {"SchemaTable", "SchemaColumn", "Cluster", "JoinPath"} <= node_labels


def test_schema_returns_core_edge_types(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = schema_payload(tmp_path, database="demo")
    edge_types = {e["type"] for e in out["edges"]}
    assert {"FK_REFERENCES", "INFERRED_JOIN", "HAS_COLUMN"} <= edge_types


def test_schema_node_includes_properties(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = schema_payload(tmp_path, database="demo")
    table_node = next(n for n in out["nodes"] if n["label"] == "SchemaTable")
    prop_names = {p["name"] for p in table_node["properties"]}
    assert {"node_id", "schema_name", "table_name", "row_count"} <= prop_names


def test_schema_label_filter(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = schema_payload(tmp_path, database="demo", label="SchemaTable")
    assert len(out["nodes"]) == 1
    assert out["nodes"][0]["label"] == "SchemaTable"
    assert out["edges"] == []


def test_schema_unknown_database(tmp_path: Path) -> None:
    _write_minimal_registry(tmp_path)
    out = schema_payload(tmp_path, database="nope")
    assert out.get("error", "").startswith("Unknown database")


def test_schema_missing_database_arg(tmp_path: Path) -> None:
    out = schema_payload(tmp_path, database="")
    assert out.get("error") == "Missing `database`"


def test_format_catalog_summary_lists_known_labels() -> None:
    summary = format_catalog_summary()
    assert "SchemaTable" in summary
    assert "FK_REFERENCES" in summary
    assert "INFERRED_JOIN" in summary
