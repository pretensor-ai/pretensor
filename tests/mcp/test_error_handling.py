"""Tests for MCP graceful error handling.

Verifies that:
- MCP tool handlers return structured JSON errors, not exceptions.
- The McpToolRegistry catches unhandled exceptions and returns {"error": ...}.
- load_registry tolerates corrupt files (returns empty registry).
- open_store_for_entry raises RuntimeError with a friendly message on corrupt graph.
- MCP tool payloads return {"error": ...} for missing/bad graph files.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from pretensor.mcp.tool_registry import McpTool, McpToolRegistry

# ---------------------------------------------------------------------------
# McpToolRegistry — handler exception is caught and returned as error dict
# ---------------------------------------------------------------------------


def test_tool_registry_catches_handler_exception() -> None:
    async def _bad_handler(args: dict) -> dict:
        raise RuntimeError("Graph engine exploded")

    reg = McpToolRegistry()
    reg.register(
        McpTool(
            name="exploder",
            description="always raises",
            input_schema={"type": "object"},
            handler=_bad_handler,
        )
    )
    result = asyncio.run(reg.call_tool("exploder", {}))
    assert "error" in result
    assert "exploded" in result["error"]
    assert result.get("tool") == "exploder"


def test_tool_registry_exception_includes_traceback() -> None:
    async def _raises(args: dict) -> dict:
        raise ValueError("something went wrong")

    reg = McpToolRegistry()
    reg.register(
        McpTool(
            name="bad",
            description="",
            input_schema={},
            handler=_raises,
        )
    )
    result = asyncio.run(reg.call_tool("bad", {}))
    assert "error" in result
    assert "traceback" in result


# ---------------------------------------------------------------------------
# load_registry — tolerates corrupt JSON
# ---------------------------------------------------------------------------


def test_load_registry_corrupt_json_returns_empty(tmp_path: Path) -> None:
    from pretensor.mcp.service_registry import load_registry

    reg_path = tmp_path / "registry.json"
    reg_path.write_text("{ this is not valid json }", encoding="utf-8")
    reg = load_registry(tmp_path)
    assert reg.list_entries() == []


def test_load_registry_missing_file_returns_empty(tmp_path: Path) -> None:
    from pretensor.mcp.service_registry import load_registry

    reg = load_registry(tmp_path)
    assert reg.list_entries() == []


# ---------------------------------------------------------------------------
# open_store_for_entry — raises RuntimeError with friendly message on corrupt
# ---------------------------------------------------------------------------


def test_open_store_for_entry_raises_on_corrupt_graph(tmp_path: Path) -> None:
    from pretensor.core.registry import RegistryEntry
    from pretensor.mcp.service_registry import open_store_for_entry

    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True)

    entry = RegistryEntry(
        connection_name="demo",
        database="demo",
        dsn="postgresql://u@h/demo",
        graph_path=str(graph),
        last_indexed_at=datetime.now(timezone.utc),
    )

    with patch(
        "pretensor.mcp.service_registry.KuzuStore",
        side_effect=RuntimeError("Cannot open DB"),
    ):
        with pytest.raises(RuntimeError, match="Cannot open graph file"):
            open_store_for_entry(entry)


# ---------------------------------------------------------------------------
# MCP tool payloads — return {"error": ...} for unknown database
# ---------------------------------------------------------------------------


def test_context_payload_unknown_database(tmp_path: Path) -> None:
    from pretensor.mcp.tools.context import context_payload

    result = context_payload(tmp_path, table="orders", db="nonexistent")
    assert "error" in result
    assert "nonexistent" in result["error"].lower() or "unknown" in result["error"].lower()


def test_cypher_payload_unknown_database(tmp_path: Path) -> None:
    from pretensor.mcp.tools.cypher import cypher_payload

    result = cypher_payload(
        tmp_path,
        query="MATCH (t:SchemaTable) RETURN t.table_name",
        database="nonexistent",
    )
    assert "error" in result
    assert "nonexistent" in result["error"].lower() or "unknown" in result["error"].lower()


def test_impact_payload_unknown_database(tmp_path: Path) -> None:
    from pretensor.mcp.tools.impact import impact_payload

    result = impact_payload(tmp_path, table="orders", database="nonexistent")
    assert "error" in result


def test_traverse_payload_unknown_database(tmp_path: Path) -> None:
    from pretensor.mcp.tools.traverse import traverse_payload

    result = traverse_payload(
        tmp_path,
        from_table="orders",
        to_table="users",
        database="nonexistent",
    )
    assert "error" in result


def test_detect_changes_payload_unknown_database(tmp_path: Path) -> None:
    from pretensor.mcp.tools.detect_changes import detect_changes_payload

    result = detect_changes_payload(tmp_path, database="nonexistent")
    assert "error" in result


# ---------------------------------------------------------------------------
# MCP tool payloads — return {"error": ...} for missing graph file
# ---------------------------------------------------------------------------


def _write_registry_only(tmp_path: Path) -> None:
    """Write a registry entry pointing at a non-existent graph file."""
    from pretensor.core.registry import GraphRegistry

    graph = tmp_path / "graphs" / "demo.kuzu"
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://u@localhost/demo",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()


def test_context_payload_missing_graph(tmp_path: Path) -> None:
    from pretensor.mcp.tools.context import context_payload

    _write_registry_only(tmp_path)
    result = context_payload(tmp_path, table="orders", db="demo")
    assert "error" in result
    assert "missing" in result["error"].lower() or "graph" in result["error"].lower()


def test_cypher_payload_missing_graph(tmp_path: Path) -> None:
    from pretensor.mcp.tools.cypher import cypher_payload

    _write_registry_only(tmp_path)
    result = cypher_payload(
        tmp_path,
        query="MATCH (t:SchemaTable) RETURN t",
        database="demo",
    )
    assert "error" in result


def test_traverse_payload_missing_graph(tmp_path: Path) -> None:
    from pretensor.mcp.tools.traverse import traverse_payload

    _write_registry_only(tmp_path)
    result = traverse_payload(
        tmp_path,
        from_table="orders",
        to_table="users",
        database="demo",
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# MCP tool payload — bad input (empty query, etc.)
# ---------------------------------------------------------------------------


def test_cypher_payload_empty_query(tmp_path: Path) -> None:
    from pretensor.mcp.tools.cypher import cypher_payload

    result = cypher_payload(tmp_path, query="", database="demo")
    assert "error" in result
    assert "empty" in result["error"].lower() or "missing" in result["error"].lower()


def test_cypher_payload_mutating_query(tmp_path: Path) -> None:
    from pretensor.mcp.tools.cypher import cypher_payload

    _write_registry_only(tmp_path)
    result = cypher_payload(tmp_path, query="CREATE (n:Node)", database="demo")
    assert "error" in result
    assert "read" in result["error"].lower() or "not permitted" in result["error"].lower()


def test_cypher_payload_invalid_timeout(tmp_path: Path) -> None:
    from pretensor.mcp.tools.cypher import cypher_payload

    result = cypher_payload(
        tmp_path,
        query="MATCH (t) RETURN t",
        database="demo",
        timeout_seconds=-1,
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# McpToolRegistry — open_store_for_entry RuntimeError propagates as error dict
# ---------------------------------------------------------------------------


def test_tool_registry_corrupt_graph_returns_error_dict(tmp_path: Path) -> None:
    """If open_store_for_entry raises, the registry must catch it in call_tool."""
    from datetime import datetime, timezone

    from pretensor.connectors.models import Column, SchemaSnapshot, Table
    from pretensor.core.builder import GraphBuilder
    from pretensor.core.registry import GraphRegistry
    from pretensor.core.store import KuzuStore
    from pretensor.mcp.server import _build_oss_registry

    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True)
    store = KuzuStore(graph)
    try:
        snap = SchemaSnapshot(
            connection_name="demo",
            database="demo",
            schemas=["public"],
            tables=[
                Table(
                    name="orders",
                    schema_name="public",
                    columns=[Column(name="id", data_type="int")],
                )
            ],
            introspected_at=datetime.now(timezone.utc),
        )
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://u@localhost/demo",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()

    registry = _build_oss_registry(tmp_path)

    with patch(
        "pretensor.mcp.service_registry.KuzuStore",
        side_effect=RuntimeError("Corrupt graph"),
    ):
        result = asyncio.run(
            registry.call_tool("context", {"table": "orders", "db": "demo"})
        )

    assert "error" in result
