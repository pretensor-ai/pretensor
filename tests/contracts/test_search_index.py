"""Contract tests for BaseSearchIndex extension point.

``SearchIndexContractTest`` verifies that any :class:`~pretensor.search.base.BaseSearchIndex`
implementation satisfies the interface.  Cloud implementations import this class
and bind ``make_index`` to their own factory.

The concrete test for :class:`~pretensor.search.index.KeywordSearchIndex` is at
the bottom of this file; it uses a real SQLite FTS5 index built from a minimal
in-memory graph.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.search.base import BaseSearchIndex, SearchResult
from pretensor.search.index import KeywordSearchIndex

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _build_keyword_index(tmp_path: Path) -> KeywordSearchIndex:
    """Build and return a KeywordSearchIndex populated with two tables."""
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    snapshot = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[
            Table(
                name="customers",
                schema_name="public",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                comment="Customer accounts",
            ),
            Table(
                name="orders",
                schema_name="public",
                columns=[
                    Column(name="id", data_type="int", is_primary_key=True),
                    Column(name="customer_id", data_type="int"),
                ],
                comment="Purchase orders",
            ),
        ],
        introspected_at=datetime.now(timezone.utc),
    )
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snapshot, store, run_relationship_discovery=False)
    finally:
        store.close()

    reg_path = tmp_path / "registry.json"
    reg = GraphRegistry(reg_path).load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://x",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()

    idx_path = KeywordSearchIndex.default_path(tmp_path)
    return KeywordSearchIndex.build(reg.load(), idx_path)


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


class SearchIndexContractTest(abc.ABC):
    """Reusable contract suite for :class:`BaseSearchIndex` implementations.

    Subclass and implement :meth:`make_index` to verify any index.

    Example (Cloud)::

        class TestHybridSearchIndex(SearchIndexContractTest):
            def make_index(self, tmp_path: Path) -> BaseSearchIndex:
                return HybridSearchIndex(tmp_path / "hybrid.idx")
    """

    @abc.abstractmethod
    def make_index(self, tmp_path: Path) -> BaseSearchIndex:
        """Return a ready-to-query index instance.

        Args:
            tmp_path: Pytest-provided temporary directory for on-disk artifacts.
        """

    # -- Interface shape -------------------------------------------------------

    def test_is_base_search_index(self, tmp_path: Path) -> None:
        """Index must be a BaseSearchIndex subclass."""
        idx = self.make_index(tmp_path)
        assert isinstance(idx, BaseSearchIndex)

    # -- search() return type --------------------------------------------------

    def test_search_returns_list(self, tmp_path: Path) -> None:
        """search() must return a list."""
        idx = self.make_index(tmp_path)
        result = idx.search("customer")
        assert isinstance(result, list)

    def test_search_returns_search_results(self, tmp_path: Path) -> None:
        """Every hit must be a SearchResult."""
        idx = self.make_index(tmp_path)
        for hit in idx.search("customer", limit=10):
            assert isinstance(hit, SearchResult), (
                f"Expected SearchResult, got {type(hit)}"
            )

    def test_search_empty_query_returns_list(self, tmp_path: Path) -> None:
        """search() with an empty query must return a list without raising."""
        idx = self.make_index(tmp_path)
        result = idx.search("")
        assert isinstance(result, list)

    def test_search_limit_respected(self, tmp_path: Path) -> None:
        """search() must return at most ``limit`` results."""
        idx = self.make_index(tmp_path)
        limit = 1
        result = idx.search("a", limit=limit)
        assert len(result) <= limit

    def test_search_result_fields_non_empty(self, tmp_path: Path) -> None:
        """Non-description string fields must not be empty on returned hits."""
        idx = self.make_index(tmp_path)
        for hit in idx.search("customer", limit=5):
            assert hit.node_type, "node_type must not be empty"
            assert hit.name, "name must not be empty"
            assert hit.connection_name, "connection_name must not be empty"

    # -- similar() return type -------------------------------------------------

    def test_similar_returns_list(self, tmp_path: Path) -> None:
        """similar() must return a list."""
        idx = self.make_index(tmp_path)
        result = idx.similar("public.customers")
        assert isinstance(result, list)

    def test_similar_excludes_exact_name(self, tmp_path: Path) -> None:
        """similar() must not return the node itself by exact name match."""
        idx = self.make_index(tmp_path)
        results = idx.similar("public.customers", limit=10)
        names = {r.name for r in results}
        assert "public.customers" not in names

    def test_similar_limit_respected(self, tmp_path: Path) -> None:
        """similar() must return at most ``limit`` results."""
        idx = self.make_index(tmp_path)
        limit = 1
        result = idx.similar("public.orders", limit=limit)
        assert len(result) <= limit

    # -- index_graph() ---------------------------------------------------------

    def test_index_graph_accepts_registry(self, tmp_path: Path) -> None:
        """index_graph() must accept a GraphRegistry without raising."""
        idx = self.make_index(tmp_path)
        reg_path = tmp_path / "registry2.json"
        reg = GraphRegistry(reg_path).load()
        idx.index_graph(reg)


# ---------------------------------------------------------------------------
# Concrete: KeywordSearchIndex
# ---------------------------------------------------------------------------


class TestKeywordSearchIndexContract(SearchIndexContractTest):
    """Run the full search index contract against :class:`KeywordSearchIndex`."""

    def make_index(self, tmp_path: Path) -> BaseSearchIndex:
        return _build_keyword_index(tmp_path)

    # -- KeywordSearchIndex-specific behaviour ---------------------------------

    def test_is_subclass_of_base(self) -> None:
        assert issubclass(KeywordSearchIndex, BaseSearchIndex)

    def test_base_is_abstract(self) -> None:
        """BaseSearchIndex must not be directly instantiable."""
        with pytest.raises(TypeError):
            BaseSearchIndex()  # type: ignore[abstract]

    def test_search_hits_customers_table(self, tmp_path: Path) -> None:
        """Querying 'customer' must return the customers table."""
        idx = self.make_index(tmp_path)
        hits = idx.search("customer", limit=5)
        assert hits, "expected at least one hit"
        names = {h.name for h in hits}
        assert any("customers" in n for n in names)

    def test_default_path_is_under_graph_dir(self, tmp_path: Path) -> None:
        """default_path() must return a path under the provided directory."""
        p = KeywordSearchIndex.default_path(tmp_path)
        assert str(tmp_path) in str(p)

    def test_needs_rebuild_when_index_missing(self, tmp_path: Path) -> None:
        """needs_rebuild() must return True when the index file does not exist."""
        reg = GraphRegistry(tmp_path / "registry.json").load()
        idx_path = tmp_path / "missing.sqlite"
        assert KeywordSearchIndex.needs_rebuild(reg, idx_path)
