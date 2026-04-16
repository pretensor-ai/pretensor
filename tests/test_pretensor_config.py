"""Unit tests for PretensorConfig — central pluggable configuration dataclass."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from pretensor.config import GraphConfig, PretensorConfig
from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.graph_models.node import GraphNode
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.combining import ConfidenceCombiner, MaxScoreCombiner
from pretensor.intelligence.discovery import RelationshipDiscovery
from pretensor.intelligence.scoring import RelationshipScorer, ScorerRegistry
from pretensor.mcp.service_context import (
    build_server_context,
    get_effective_search_index_cls,
    reset_server_context,
    set_server_context,
)
from pretensor.mcp.tools.search import query_payload
from pretensor.search.index import KeywordSearchIndex
from pretensor.semantic.base import NullSemanticLayer


def test_default_config_has_sensible_oss_defaults() -> None:
    cfg = PretensorConfig()
    assert isinstance(cfg.graph, GraphConfig)
    assert isinstance(cfg.scorer_registry, ScorerRegistry)
    assert isinstance(cfg.combiner, MaxScoreCombiner)
    assert cfg.search_index_cls is KeywordSearchIndex
    assert isinstance(cfg.semantic_layer, NullSemanticLayer)


def test_graph_config_fields_preserved() -> None:
    cfg = PretensorConfig()
    assert cfg.graph.stale_index_warning_days == 7
    assert cfg.graph.join_path_max_depth == 4
    assert cfg.graph.min_cluster_size_merge == 3


def test_custom_graph_config() -> None:
    graph = GraphConfig(stale_index_warning_days=14, join_path_max_depth=8)
    cfg = PretensorConfig(graph=graph)
    assert cfg.graph.stale_index_warning_days == 14
    assert cfg.graph.join_path_max_depth == 8


def test_custom_scorer_registry() -> None:
    class _FixedScorer(RelationshipScorer):
        def name(self) -> str:
            return "fixed"

        def score(
            self,
            snapshot: SchemaSnapshot,
            explicit_fk_keys: set[tuple[str, str, str, str]],
        ) -> list[RelationshipCandidate]:
            _ = (snapshot, explicit_fk_keys)
            return []

    registry = ScorerRegistry([_FixedScorer()])
    cfg = PretensorConfig(scorer_registry=registry)
    assert cfg.scorer_registry is registry


def test_custom_combiner() -> None:
    class _PassthroughCombiner(ConfidenceCombiner):
        def combine(
            self, *groups: list[RelationshipCandidate]
        ) -> list[RelationshipCandidate]:
            return [c for group in groups for c in group]

    combiner = _PassthroughCombiner()
    cfg = PretensorConfig(combiner=combiner)
    assert cfg.combiner is combiner


def test_custom_search_index_cls() -> None:
    from pretensor.core.registry import GraphRegistry
    from pretensor.search.base import BaseSearchIndex, SearchResult

    class _CustomIndex(BaseSearchIndex):
        def search(
            self, q: str, *, db: str | None = None, limit: int = 10
        ) -> list[SearchResult]:
            return []

        def similar(
            self, name: str, *, db: str | None = None, limit: int = 10
        ) -> list[SearchResult]:
            return []

        def index_graph(self, registry: GraphRegistry) -> None:
            pass

    cfg = PretensorConfig(search_index_cls=_CustomIndex)
    assert cfg.search_index_cls is _CustomIndex


def test_cloud_subclass_override() -> None:
    """Cloud pattern: subclass PretensorConfig to override defaults."""

    class _CloudConfig(PretensorConfig):
        pass

    cloud_cfg = _CloudConfig()
    assert isinstance(cloud_cfg, PretensorConfig)
    assert cloud_cfg.search_index_cls is KeywordSearchIndex


def test_server_context_uses_configured_search_index_cls(tmp_path: Path) -> None:
    from pretensor.core.registry import GraphRegistry
    from pretensor.search.base import BaseSearchIndex, SearchResult

    class _CustomIndex(BaseSearchIndex):
        def __init__(self, index_path: Path) -> None:
            self._index_path = index_path

        def search(
            self, q: str, *, db: str | None = None, limit: int = 10
        ) -> list[SearchResult]:
            _ = (q, db, limit)
            return []

        def similar(
            self, name: str, *, db: str | None = None, limit: int = 10
        ) -> list[SearchResult]:
            _ = (name, db, limit)
            return []

        def index_graph(self, registry: GraphRegistry) -> None:
            _ = registry

    ctx = build_server_context(
        tmp_path, config=PretensorConfig(search_index_cls=_CustomIndex)
    )
    assert ctx.search_index_cls is _CustomIndex


def test_query_payload_uses_context_search_index_cls(tmp_path: Path) -> None:
    from pretensor.core.registry import GraphRegistry
    from pretensor.search.base import BaseSearchIndex, SearchResult

    calls: dict[str, object] = {}

    class _TrackingIndex(BaseSearchIndex):
        def __init__(self, index_path: Path) -> None:
            calls["ctor_path"] = index_path

        @staticmethod
        def default_path(graph_dir: Path) -> Path:
            calls["default_path_graph_dir"] = graph_dir
            return graph_dir / "custom-search-index.bin"

        @classmethod
        def load_or_build(cls, registry: GraphRegistry, index_path: Path) -> "_TrackingIndex":
            calls["load_or_build_registry"] = registry
            calls["load_or_build_path"] = index_path
            return cls(index_path)

        def search(
            self, q: str, *, db: str | None = None, limit: int = 10
        ) -> list[SearchResult]:
            calls["search_q"] = q
            calls["search_db"] = db
            calls["search_limit"] = limit
            return []

        def similar(
            self, name: str, *, db: str | None = None, limit: int = 10
        ) -> list[SearchResult]:
            _ = (name, db, limit)
            return []

        def index_graph(self, registry: GraphRegistry) -> None:
            _ = registry

    reset_server_context()
    ctx = build_server_context(
        tmp_path, config=PretensorConfig(search_index_cls=_TrackingIndex)
    )
    set_server_context(ctx)
    try:
        out = query_payload(tmp_path, q="customers", db="demo", limit=5)
    finally:
        reset_server_context()

    assert out["results"] == []
    assert calls["default_path_graph_dir"] == tmp_path
    assert calls["load_or_build_path"] == tmp_path / "custom-search-index.bin"
    assert calls["search_q"] == "customers"
    assert calls["search_db"] == "demo"
    assert calls["search_limit"] == 20


def test_query_payload_defaults_to_keyword_search_index_without_context() -> None:
    from pretensor.search.base import SearchResult

    reset_server_context()
    assert get_effective_search_index_cls() is KeywordSearchIndex

    class _StubKeywordIndex(KeywordSearchIndex):
        def __init__(self) -> None:
            self._path = Path("/tmp/stub")

        def search(
            self, q: str, *, db: str | None = None, limit: int = 10
        ) -> list[SearchResult]:
            _ = (q, db, limit)
            return []

    with patch(
        "pretensor.mcp.tools.search.KeywordSearchIndex.load_or_build",
        return_value=_StubKeywordIndex(),
    ) as mocked_load:
        out = query_payload(Path("/tmp/nonexistent-graph"), q="x", limit=1)
    assert out["results"] == []
    mocked_load.assert_called_once()


def _two_table_snap() -> SchemaSnapshot:
    customers = Table(
        name="customers",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="customer_id", data_type="int"),
        ],
        foreign_keys=[],
    )
    return SchemaSnapshot(
        connection_name="demo",
        database="db",
        schemas=["public"],
        tables=[customers, orders],
        introspected_at=datetime.now(timezone.utc),
    )


def test_builder_uses_config_scorer_and_combiner(tmp_path) -> None:
    """PretensorConfig passed to GraphBuilder threads scorers and combiner into discovery."""
    snap = _two_table_snap()
    combine_called = False

    class _TrackingCombiner(ConfidenceCombiner):
        def combine(
            self, *groups: list[RelationshipCandidate]
        ) -> list[RelationshipCandidate]:
            nonlocal combine_called
            combine_called = True
            return MaxScoreCombiner().combine(*groups)

    cfg = PretensorConfig(combiner=_TrackingCombiner())
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        GraphBuilder().build(snap, store, config=cfg)
        assert combine_called, "combiner from PretensorConfig should be used by builder"
    finally:
        store.close()


def test_builder_noop_scorer_produces_no_inferred_joins(tmp_path) -> None:
    """Custom no-op scorer via config should produce zero inferred joins."""
    from tests.query_helpers import first_cell, single_query_result

    class _NoopScorer(RelationshipScorer):
        def name(self) -> str:
            return "noop"

        def score(
            self,
            snapshot: SchemaSnapshot,
            explicit_fk_keys: set[tuple[str, str, str, str]],
        ) -> list[RelationshipCandidate]:
            return []

    cfg = PretensorConfig(scorer_registry=ScorerRegistry([_NoopScorer()]))
    snap = _two_table_snap()
    store = KuzuStore(tmp_path / "g2.kuzu")
    try:
        GraphBuilder().build(snap, store, config=cfg)
        result = single_query_result(
            store, "MATCH ()-[e:INFERRED_JOIN]->() RETURN count(*) AS c"
        )
        assert first_cell(result) == 0
    finally:
        store.close()


def test_discovery_uses_config_directly(tmp_path) -> None:
    """RelationshipDiscovery accepts PretensorConfig via its config param."""
    snap = _two_table_snap()
    combine_called = False

    class _TrackingCombiner(ConfidenceCombiner):
        def combine(
            self, *groups: list[RelationshipCandidate]
        ) -> list[RelationshipCandidate]:
            nonlocal combine_called
            combine_called = True
            return MaxScoreCombiner().combine(*groups)

    cfg = PretensorConfig(combiner=_TrackingCombiner())
    store = KuzuStore(tmp_path / "g3.kuzu")
    try:
        store.ensure_schema()
        store.clear_graph()
        for t in snap.tables:
            store.upsert_table(
                GraphNode(
                    node_id=table_node_id(snap.connection_name, t.schema_name, t.name),
                    connection_name=snap.connection_name,
                    database=snap.database,
                    schema_name=t.schema_name,
                    table_name=t.name,
                    row_count=t.row_count,
                    comment=t.comment,
                )
            )
        RelationshipDiscovery(store, config=cfg).discover(snap)
        assert combine_called
    finally:
        store.close()


def test_discovery_explicit_params_override_config(tmp_path) -> None:
    """scorers/combiner passed explicitly take priority over config."""
    snap = _two_table_snap()
    explicit_combiner_called = False

    class _ExplicitCombiner(ConfidenceCombiner):
        def combine(
            self, *groups: list[RelationshipCandidate]
        ) -> list[RelationshipCandidate]:
            nonlocal explicit_combiner_called
            explicit_combiner_called = True
            return MaxScoreCombiner().combine(*groups)

    class _NeverCombiner(ConfidenceCombiner):
        def combine(
            self, *groups: list[RelationshipCandidate]
        ) -> list[RelationshipCandidate]:
            raise AssertionError("config combiner should not be called")

    cfg = PretensorConfig(combiner=_NeverCombiner())
    store = KuzuStore(tmp_path / "g4.kuzu")
    try:
        store.ensure_schema()
        store.clear_graph()
        for t in snap.tables:
            store.upsert_table(
                GraphNode(
                    node_id=table_node_id(snap.connection_name, t.schema_name, t.name),
                    connection_name=snap.connection_name,
                    database=snap.database,
                    schema_name=t.schema_name,
                    table_name=t.name,
                    row_count=t.row_count,
                    comment=t.comment,
                )
            )
        RelationshipDiscovery(
            store, combiner=_ExplicitCombiner(), config=cfg
        ).discover(snap)
        assert explicit_combiner_called
    finally:
        store.close()
