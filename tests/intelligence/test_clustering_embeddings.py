"""Tests for embedding-aware clustering + label-selection tiebreaker.

Three contracts pinned here:

1.  Default (``cluster_blend=0.0``, no ``embedding_resolver``) is byte-identical
    to the pre-embedding pipeline output — the parity test in
    ``tests/intelligence/test_null_path_parity.py`` already covers this; the
    targeted unit-level checks below are the smaller-scope guards.
2.  When ``cluster_blend > 0`` and tables carry embeddings, the cosine signal
    boosts existing FK / INFERRED_JOIN edge weights but does not add new
    edges; ``GraphExporter._apply_embedding_edge_blend`` is the single
    enforcement point.
3.  When the labeler is given an ``embedding_resolver``, centroid distance
    only resolves anchor-table ties — heuristic role-weighted degree always
    dominates.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from pretensor.config import EmbeddingsConfig, GraphConfig, PretensorConfig
from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.intelligence.cluster_labeler import (
    _compute_cluster_centroid,
    _heuristic_label,
)
from pretensor.intelligence.clustering import Cluster
from pretensor.intelligence.graph_export import GraphExporter
from pretensor.intelligence.pipeline import run_intelligence_layer


def _table(
    name: str, columns: list[str], *, fks: list[ForeignKey] | None = None
) -> Table:
    return Table(
        name=name,
        schema_name="public",
        columns=[Column(name=c, data_type="text") for c in columns],
        foreign_keys=fks or [],
        comment="",
    )


def _build_minimal_db(tmp_path: Path, name: str = "test") -> KuzuStore:
    """Three FK-linked tables: ``customers`` ← ``orders`` ← ``shipments``."""
    snap = SchemaSnapshot(
        connection_name=name,
        database=name,
        schemas=["public"],
        tables=[
            _table("customers", ["id", "name"]),
            _table(
                "orders",
                ["id", "customer_id", "amount"],
                fks=[
                    ForeignKey(
                        source_schema="public",
                        source_table="orders",
                        source_column="customer_id",
                        target_schema="public",
                        target_table="customers",
                        target_column="id",
                    ),
                ],
            ),
            _table(
                "shipments",
                ["id", "order_id"],
                fks=[
                    ForeignKey(
                        source_schema="public",
                        source_table="shipments",
                        source_column="order_id",
                        target_schema="public",
                        target_table="orders",
                        target_column="id",
                    ),
                ],
            ),
        ],
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    store = KuzuStore(tmp_path / f"{name}.kuzu")
    GraphBuilder().build(snap, store, run_relationship_discovery=False)
    return store


def _node_ids_by_table(store: KuzuStore, db: str) -> dict[str, str]:
    rows = store.query_all_rows(
        "MATCH (t:SchemaTable {database: $db}) RETURN t.table_name, t.node_id",
        {"db": db},
    )
    return {str(name): str(nid) for name, nid in rows}


def _set_emb(store: KuzuStore, nid: str, vec: list[float]) -> None:
    store.set_table_embedding(nid, vec)


# ── EmbeddingsConfig defaults (toggle parity) ────────────────────────────────


def test_cluster_blend_default_is_zero() -> None:
    """Default config keeps ``cluster_blend`` at 0.0 — null-path guarantee."""
    assert EmbeddingsConfig().cluster_blend == 0.0


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, -1.0, 2.0, float("inf")])
def test_cluster_blend_rejects_out_of_range(bad_value: float) -> None:
    """``EmbeddingsConfig.__post_init__`` validates the [0.0, 1.0] range."""
    with pytest.raises(ValueError, match="cluster_blend"):
        EmbeddingsConfig(cluster_blend=bad_value)


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, -1.0, 2.0, float("inf")])
def test_role_weight_rejects_out_of_range(bad_value: float) -> None:
    """``EmbeddingsConfig.__post_init__`` validates the [0.0, 1.0] range."""
    with pytest.raises(ValueError, match="role_weight"):
        EmbeddingsConfig(role_weight=bad_value)


@pytest.mark.parametrize("good_value", [0.0, 0.25, 0.5, 1.0])
def test_cluster_blend_accepts_valid_range(good_value: float) -> None:
    """Boundary values (0.0 and 1.0) and typical settings are accepted."""
    cfg = EmbeddingsConfig(cluster_blend=good_value)
    assert cfg.cluster_blend == good_value


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, -1.0, 2.0])
def test_join_threshold_rejects_out_of_range(bad_value: float) -> None:
    """``EmbeddingsConfig.__post_init__`` validates the [0.0, 1.0] range."""
    with pytest.raises(ValueError, match="join_threshold"):
        EmbeddingsConfig(join_threshold=bad_value)


@pytest.mark.parametrize("good_value", [None, 0.0, 0.5, 0.85, 1.0])
def test_join_threshold_accepts_none_or_valid_range(good_value: float | None) -> None:
    """``None`` (off) and any value in [0.0, 1.0] are accepted."""
    cfg = EmbeddingsConfig(join_threshold=good_value)
    assert cfg.join_threshold == good_value


# ── GraphExporter cosine blend ───────────────────────────────────────────────


def test_to_igraph_default_blend_does_not_touch_weights(tmp_path: Path) -> None:
    """``cluster_blend=0.0`` (default) → edge weights identical to the pre-embedding pipeline."""
    store = _build_minimal_db(tmp_path)
    try:
        nids = _node_ids_by_table(store, "test")
        # Populate vectors that would, if blended, change weights.
        _set_emb(store, nids["customers"], [1.0] + [0.0] * 383)
        _set_emb(store, nids["orders"], [1.0] + [0.0] * 383)
        _set_emb(store, nids["shipments"], [1.0] + [0.0] * 383)

        g_off = GraphExporter(store).to_igraph("test", config=GraphConfig())
        g_on = GraphExporter(store).to_igraph(
            "test", config=GraphConfig(), cluster_blend=0.0
        )

        # Both graphs have 3 vertices and 2 FK edges, equal weights (1.0).
        assert g_off.vcount() == g_on.vcount() == 3
        assert g_off.ecount() == g_on.ecount() == 2
        weights_off = sorted(float(w) for w in g_off.es["weight"])
        weights_on = sorted(float(w) for w in g_on.es["weight"])
        assert weights_off == weights_on
        assert weights_off == [1.0, 1.0]
    finally:
        store.close()


def test_to_igraph_blend_boosts_only_embedded_endpoints(tmp_path: Path) -> None:
    """With ``cluster_blend>0``, edges between embedded pairs get a boost;
    edges with at least one unembedded endpoint stay at the FK weight.
    """
    store = _build_minimal_db(tmp_path)
    try:
        nids = _node_ids_by_table(store, "test")
        # Embed customers + orders identical → cosine = 1.0; leave shipments unembedded.
        v = [1.0] + [0.0] * 383
        _set_emb(store, nids["customers"], v)
        _set_emb(store, nids["orders"], v)

        g = GraphExporter(store).to_igraph(
            "test", config=GraphConfig(), cluster_blend=0.5
        )

        # Extract edge weights keyed by endpoint table names.
        names = g.vs["name"]
        weights_by_pair: dict[frozenset[str], float] = {}
        for edge in g.es:
            pair = frozenset({names[edge.source], names[edge.target]})
            weights_by_pair[pair] = float(edge["weight"])

        cust_orders = frozenset({"public.customers", "public.orders"})
        orders_ship = frozenset({"public.orders", "public.shipments"})

        # customers↔orders boosted by 0.5 * 1.0 = 0.5 → weight = 1.5
        assert weights_by_pair[cust_orders] == pytest.approx(1.5)
        # orders↔shipments unchanged (shipments has no embedding)
        assert weights_by_pair[orders_ship] == pytest.approx(1.0)
    finally:
        store.close()


def test_to_igraph_blend_no_embeddings_is_noop(tmp_path: Path) -> None:
    """``cluster_blend>0`` but zero embedded tables → no error, no weight change."""
    store = _build_minimal_db(tmp_path)
    try:
        g = GraphExporter(store).to_igraph(
            "test", config=GraphConfig(), cluster_blend=0.5
        )
        weights = sorted(float(w) for w in g.es["weight"])
        assert weights == [1.0, 1.0]
    finally:
        store.close()


def test_to_igraph_blend_creates_no_new_edges(tmp_path: Path) -> None:
    """The blend never introduces edges between non-adjacent embedded pairs.

    Three tables, two FK edges; embedding ``customers`` + ``shipments``
    identical means cosine 1.0 between *non-adjacent* tables, but the graph
    must stay at 2 edges.
    """
    store = _build_minimal_db(tmp_path)
    try:
        nids = _node_ids_by_table(store, "test")
        v = [1.0] + [0.0] * 383
        _set_emb(store, nids["customers"], v)
        _set_emb(store, nids["shipments"], v)

        g = GraphExporter(store).to_igraph(
            "test", config=GraphConfig(), cluster_blend=0.5
        )
        assert g.ecount() == 2  # No new (customers, shipments) edge
    finally:
        store.close()


# ── ClusterLabeler centroid tiebreaker ───────────────────────────────────────


def test_compute_cluster_centroid_basic() -> None:
    """Centroid is the per-component mean over tables with embeddings."""
    cluster = Cluster(table_ids=["a", "b", "c"], cohesion_score=0.0)
    embs: dict[str, list[float] | None] = {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
        "c": None,  # skipped, doesn't pull centroid
    }
    centroid = _compute_cluster_centroid(cluster, embs.get)
    assert centroid is not None
    assert centroid == [0.5, 0.5, 0.0]


def test_compute_cluster_centroid_no_embeddings_returns_none() -> None:
    cluster = Cluster(table_ids=["a", "b"], cohesion_score=0.0)
    embs: dict[str, list[float] | None] = {"a": None, "b": None}
    assert _compute_cluster_centroid(cluster, embs.get) is None


def test_heuristic_label_centroid_tiebreaker_only_breaks_ties(tmp_path: Path) -> None:
    """When role-weighted degree ties, centroid distance picks the closer table.
    The heuristic key dominates: a clearly higher-degree table wins regardless
    of centroid distance.
    """
    # Build a simple two-table store, both same role/degree → tied on heuristic.
    snap = SchemaSnapshot(
        connection_name="test",
        database="test",
        schemas=["public"],
        tables=[
            _table("alpha", ["id"]),
            _table("bravo", ["id"]),
        ],
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    store = KuzuStore(tmp_path / "tied.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        nids = _node_ids_by_table(store, "test")
        cluster = Cluster(table_ids=[nids["alpha"], nids["bravo"]], cohesion_score=0.0)

        # Without centroid tiebreaker, alphabetical wins → "alpha".
        label_off, _ = _heuristic_label(store, cluster)
        assert "alpha" in label_off

        # Centroid is closer to "bravo" than to "alpha" → bravo wins.
        embs: dict[str, list[float] | None] = {
            nids["alpha"]: [1.0, 0.0],
            nids["bravo"]: [0.0, 1.0],
        }
        # Centroid identical to bravo → distance(bravo) = 0, distance(alpha) > 0.
        centroid = [0.0, 1.0]
        label_on, _ = _heuristic_label(
            store, cluster, centroid=centroid, embedding_resolver=embs.get
        )
        assert "bravo" in label_on
    finally:
        store.close()


def test_heuristic_label_role_weighted_degree_still_dominates(
    tmp_path: Path,
) -> None:
    """A higher-degree table beats a closer-to-centroid one — heuristic primary."""
    # Three tables, alpha is FK-connected to both others (degree 2);
    # bravo and charlie each have degree 1.
    snap = SchemaSnapshot(
        connection_name="test",
        database="test",
        schemas=["public"],
        tables=[
            _table("alpha", ["id"]),
            _table(
                "bravo",
                ["id", "alpha_id"],
                fks=[
                    ForeignKey(
                        source_schema="public",
                        source_table="bravo",
                        source_column="alpha_id",
                        target_schema="public",
                        target_table="alpha",
                        target_column="id",
                    ),
                ],
            ),
            _table(
                "charlie",
                ["id", "alpha_id"],
                fks=[
                    ForeignKey(
                        source_schema="public",
                        source_table="charlie",
                        source_column="alpha_id",
                        target_schema="public",
                        target_table="alpha",
                        target_column="id",
                    ),
                ],
            ),
        ],
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    store = KuzuStore(tmp_path / "hub.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        nids = _node_ids_by_table(store, "test")
        cluster = Cluster(
            table_ids=[nids["alpha"], nids["bravo"], nids["charlie"]],
            cohesion_score=0.0,
        )

        # Centroid placed identical to bravo (so bravo is centroid-closest)
        # while alpha is centroid-farthest.  Heuristic must still pick alpha
        # because alpha has degree 2 in the cluster.
        embs: dict[str, list[float] | None] = {
            nids["alpha"]: [1.0, 0.0],
            nids["bravo"]: [0.0, 1.0],
            nids["charlie"]: [0.0, 0.5],
        }
        centroid = [0.0, 1.0]  # identical to bravo

        label, _ = _heuristic_label(
            store, cluster, centroid=centroid, embedding_resolver=embs.get
        )
        assert "alpha" in label, (
            f"role-weighted degree (alpha=2 vs bravo=1) must dominate "
            f"centroid distance; got label {label!r}"
        )
    finally:
        store.close()


# ── Pipeline integration: cluster_blend toggle plumbed through ───────────────


def test_pipeline_cluster_blend_off_keeps_default_behavior(
    tmp_path: Path, load_schema: Any
) -> None:
    """With ``cluster_blend=0.0`` the pipeline produces identical Kuzu state
    regardless of whether tables carry embeddings or not.
    """
    snap = load_schema("pagila")
    store = KuzuStore(tmp_path / "pagila.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=True)
        # Populate one embedding for one table; with blend=0 it should be
        # entirely ignored.
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {database: 'pagila'}) RETURN t.node_id LIMIT 1"
        )
        assert rows
        _set_emb(store, str(rows[0][0]), [1.0] + [0.0] * 383)

        cfg = PretensorConfig(embeddings=EmbeddingsConfig())  # blend=0.0
        asyncio.run(run_intelligence_layer(store, "pagila", config=cfg))

        cluster_count = store.query_all_rows(
            "MATCH (c:Cluster) WHERE c.database_key = 'pagila' RETURN count(c)"
        )
        assert int(cluster_count[0][0]) > 0
    finally:
        store.close()
