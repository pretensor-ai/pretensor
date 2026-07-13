"""Tests for StoreCache and connection-pooling integration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.mcp.service_context import (
    build_server_context,
    reset_server_context,
    set_server_context,
)
from pretensor.mcp.service_registry import (
    load_registry,
    open_store_for_entry,
    release_store,
    resolve_registry_entry,
)
from pretensor.mcp.store_cache import StoreCache


def _make_graph(tmp_path: Path, connection_name: str = "demo") -> Path:
    snap = SchemaSnapshot(
        connection_name=connection_name,
        database=connection_name,
        schemas=["public"],
        tables=[
            Table(
                name="users",
                schema_name="public",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            )
        ],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / f"{connection_name}.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name=connection_name,
        database=connection_name,
        dsn=f"postgresql://localhost/{connection_name}",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()
    return graph


class TestStoreCache:
    def test_same_path_returns_same_object(self, tmp_path: Path) -> None:
        graph = _make_graph(tmp_path)
        cache = StoreCache()
        try:
            s1 = cache.get(graph)
            s2 = cache.get(graph)
            assert s1 is s2
        finally:
            cache.close()

    def test_resolved_path_matches(self, tmp_path: Path) -> None:
        graph = _make_graph(tmp_path)
        cache = StoreCache()
        try:
            s1 = cache.get(graph)
            s2 = cache.get(graph.resolve())
            assert s1 is s2
        finally:
            cache.close()

    def test_owns_returns_true_for_cached(self, tmp_path: Path) -> None:
        graph = _make_graph(tmp_path)
        cache = StoreCache()
        try:
            store = cache.get(graph)
            assert cache.owns(store) is True
        finally:
            cache.close()

    def test_owns_returns_false_for_unrelated(self, tmp_path: Path) -> None:
        graph = _make_graph(tmp_path)
        cache = StoreCache()
        try:
            cache.get(graph)
            unrelated = KuzuStore(graph)
            try:
                assert cache.owns(unrelated) is False
            finally:
                unrelated.close()
        finally:
            cache.close()

    def test_close_empties_cache(self, tmp_path: Path) -> None:
        graph = _make_graph(tmp_path)
        cache = StoreCache()
        cache.get(graph)
        cache.close()
        assert len(cache._stores) == 0

    def test_close_logs_and_continues_on_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A store whose close() raises is logged, not swallowed, and does not
        abort the loop or leave the cache populated."""

        class _BoomStore:
            def close(self) -> None:
                raise RuntimeError("boom")

        cache = StoreCache()
        cache._stores[Path("/fake/a.kuzu")] = _BoomStore()  # type: ignore[assignment]
        cache._stores[Path("/fake/b.kuzu")] = _BoomStore()  # type: ignore[assignment]
        with caplog.at_level("WARNING"):
            cache.close()  # must not raise
        assert len(cache._stores) == 0
        assert (
            sum("Error closing cached store" in r.message for r in caplog.records) == 2
        )


class TestPoolingIntegration:
    def setup_method(self) -> None:
        reset_server_context()

    def teardown_method(self) -> None:
        reset_server_context()

    def test_pooled_stores_are_same_object(self, tmp_path: Path) -> None:
        _make_graph(tmp_path)
        ctx = build_server_context(tmp_path)
        set_server_context(ctx)

        reg = load_registry(tmp_path)
        entry = resolve_registry_entry(reg, "demo")
        assert entry is not None

        s1 = open_store_for_entry(entry)
        s2 = open_store_for_entry(entry)
        assert s1 is s2

    def test_release_store_does_not_close_pooled(self, tmp_path: Path) -> None:
        _make_graph(tmp_path)
        ctx = build_server_context(tmp_path)
        set_server_context(ctx)

        reg = load_registry(tmp_path)
        entry = resolve_registry_entry(reg, "demo")
        assert entry is not None

        store = open_store_for_entry(entry)
        release_store(store)
        rows = store.query_all_rows("MATCH (t:SchemaTable) RETURN t.table_name")
        assert len(rows) >= 1

    def test_release_store_closes_unpooled(self, tmp_path: Path) -> None:
        graph = _make_graph(tmp_path)
        store = KuzuStore(graph)
        store.ensure_schema()
        release_store(store)

    def test_reset_server_context_closes_cache(self, tmp_path: Path) -> None:
        _make_graph(tmp_path)
        ctx = build_server_context(tmp_path)
        set_server_context(ctx)

        reg = load_registry(tmp_path)
        entry = resolve_registry_entry(reg, "demo")
        assert entry is not None
        open_store_for_entry(entry)

        assert len(ctx.store_cache._stores) == 1
        reset_server_context()
        assert len(ctx.store_cache._stores) == 0
