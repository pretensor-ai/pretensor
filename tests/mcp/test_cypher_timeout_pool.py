"""Regression tests: cypher timeouts use one bounded, shared worker pool.

Previously every ``cypher_payload`` call created its own
``ThreadPoolExecutor(max_workers=1)`` inside a ``with`` block. That leaked one
unkillable worker thread per timed-out query (unbounded under load), and the
``with`` block's ``shutdown(wait=True)`` blocked the timeout response itself
until the runaway query finished.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.mcp.service import cypher_payload
from pretensor.mcp.tools import cypher as cypher_mod


def _build_graph(tmp_path: Path) -> None:
    users = Table(
        name="users",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[users],
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


def test_pool_is_shared_across_calls() -> None:
    assert cypher_mod._get_query_pool() is cypher_mod._get_query_pool()


def test_timeout_returns_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out query must not block on the still-running worker."""
    _build_graph(tmp_path)

    def _slow(graph_path: Path, q: str) -> list:
        time.sleep(1.5)
        return []

    monkeypatch.setattr(cypher_mod, "_materialize_read_only_rows", _slow)
    started = time.perf_counter()
    out = cypher_payload(
        tmp_path, query="MATCH (t:SchemaTable) RETURN t.node_id", timeout_seconds=0.1
    )
    elapsed = time.perf_counter() - started
    assert "timed out" in out["error"]
    assert elapsed < 1.0  # the old per-call `with` block waited out the full sleep


def test_wedged_workers_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """More timed-out queries than pool slots must not grow the thread count."""
    _build_graph(tmp_path)

    def _slow(graph_path: Path, q: str) -> list:
        time.sleep(1.5)
        return []

    monkeypatch.setattr(cypher_mod, "_materialize_read_only_rows", _slow)
    for _ in range(cypher_mod._QUERY_POOL_MAX_WORKERS + 3):
        out = cypher_payload(
            tmp_path,
            query="MATCH (t:SchemaTable) RETURN t.node_id",
            timeout_seconds=0.05,
        )
        assert "timed out" in out["error"]

    pool_threads = [
        t for t in threading.enumerate() if t.name.startswith("cypher-query")
    ]
    assert len(pool_threads) <= cypher_mod._QUERY_POOL_MAX_WORKERS
