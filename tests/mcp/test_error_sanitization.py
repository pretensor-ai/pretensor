"""Regression tests: MCP error envelopes must not leak internals to clients.

The MCP client is semi-trusted. No tool result or resource body may contain a
Python traceback, a raw exception string, or an absolute filesystem path —
even when a handler raises (e.g. a corrupt/missing graph file).
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.mcp.server import _build_oss_registry

_ABS_PATH = re.compile(r"(?:/[^\s'\"]+){2,}")


def _assert_no_leak(blob: str) -> None:
    assert "traceback" not in blob.lower()
    assert "Traceback (most recent call last)" not in blob
    # No absolute path-looking token (the on-disk graph path is the main risk).
    assert not _ABS_PATH.search(blob), f"absolute path leaked: {blob!r}"


def _build_registry(tmp_path: Path) -> None:
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
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


def test_forced_handler_exception_envelope_has_no_traceback_or_path(
    tmp_path: Path,
) -> None:
    _build_registry(tmp_path)
    registry = _build_oss_registry(tmp_path)
    # Force every store open to raise with a path-bearing message.
    with patch(
        "pretensor.mcp.service_registry.KuzuStore",
        side_effect=RuntimeError("boom at /var/lib/pretensor/graphs/demo.kuzu"),
    ):
        result = asyncio.run(
            registry.call_tool("context", {"table": "orders", "db": "demo"})
        )
    assert "error" in result
    assert "traceback" not in result
    _assert_no_leak(str(result))


def test_cypher_engine_error_is_path_scrubbed(tmp_path: Path) -> None:
    """A real engine error on the fall-through path must not echo the graph path."""
    from pretensor.mcp.tools.cypher import cypher_payload

    _build_registry(tmp_path)
    # Valid leading clause (passes the allowlist) but malformed → engine error.
    result = cypher_payload(
        tmp_path,
        query="MATCH (t:SchemaTable) RETURN",
        database="demo",
    )
    assert "error" in result
    _assert_no_leak(str(result))


def test_read_resource_error_path_is_opaque(tmp_path: Path) -> None:
    """A failing resource handler returns a correlation id, not the exception."""
    from pretensor.mcp import server as server_mod

    _build_registry(tmp_path)
    with patch.object(
        server_mod,
        "databases_resource_markdown",
        side_effect=RuntimeError("kaboom at /var/lib/pretensor/graphs/demo.kuzu"),
    ):
        out = server_mod.render_resource_markdown(tmp_path, "pretensor://databases")
    assert "Correlation ID" in out
    assert "kaboom" not in out
    _assert_no_leak(out)


def test_validate_sql_sanitizes_store_open_failure(tmp_path: Path) -> None:
    from pretensor.mcp.tools.validate_sql import validate_sql_payload

    _build_registry(tmp_path)
    with patch(
        "pretensor.mcp.service_registry.KuzuStore",
        side_effect=RuntimeError("corrupt at /var/lib/pretensor/graphs/demo.kuzu"),
    ):
        result = validate_sql_payload(
            tmp_path,
            sql="SELECT 1",
            database="demo",
        )
    assert "error" in result
    _assert_no_leak(str(result))
