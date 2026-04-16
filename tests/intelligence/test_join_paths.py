"""Unit tests for intelligence/join_paths/on_demand.py."""

from __future__ import annotations

import datetime
from pathlib import Path

from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.intelligence.join_paths.on_demand import (
    _FK_CONFIDENCE,
    AdjEdge,
    JoinStep,
    best_path,
    best_paths,
    build_adjacency,
    edge_cost,
    parse_steps_json,
    steps_to_json_payload,
    yen_k_shortest,
)


def _ts() -> datetime.datetime:
    return datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)


def _snap_with_fk(tmp_path: Path) -> tuple[KuzuStore, str, str]:
    """Build a two-table graph (customers → orders via FK) and return (store, cid, oid)."""
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
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="orders",
                source_column="customer_id",
                target_schema="public",
                target_table="customers",
                target_column="id",
            )
        ],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="db",
        schemas=["public"],
        tables=[customers, orders],
        introspected_at=_ts(),
    )
    store = KuzuStore(tmp_path / "jp.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    except Exception:
        store.close()
        raise
    cid = table_node_id("demo", "public", "customers")
    oid = table_node_id("demo", "public", "orders")
    return store, cid, oid


def test_build_adjacency_bidirectional(tmp_path: Path) -> None:
    store, cid, oid = _snap_with_fk(tmp_path)
    try:
        adj = build_adjacency(store, "db")
    finally:
        store.close()

    # Both directions should exist
    assert any(e.to_id == cid for e in adj.get(oid, []))
    assert any(e.to_id == oid for e in adj.get(cid, []))


def test_build_adjacency_fk_confidence_is_one(tmp_path: Path) -> None:
    store, cid, oid = _snap_with_fk(tmp_path)
    try:
        adj = build_adjacency(store, "db")
    finally:
        store.close()

    all_edges = adj.get(oid, []) + adj.get(cid, [])
    fk_edges = [e for e in all_edges if e.kind == "fk"]
    assert fk_edges, "expected FK edges in adjacency"
    assert all(e.confidence == _FK_CONFIDENCE for e in fk_edges)


# ── Pure function tests (no store needed) ─────────────────────────────────────


def _adj_chain(n: int) -> tuple[dict[str, list[AdjEdge]], dict[str, tuple[str, str]]]:
    """Build a linear chain t0→t1→t2→...→t(n-1) with confidence 1.0."""
    adj: dict[str, list[AdjEdge]] = {}
    meta: dict[str, tuple[str, str]] = {}
    for i in range(n):
        tid = f"t{i}"
        meta[tid] = ("public", f"table{i}")
        if i + 1 < n:
            nxt = f"t{i + 1}"
            adj.setdefault(tid, []).append(AdjEdge(nxt, "id", "ref_id", "fk", 1.0))
            adj.setdefault(nxt, []).append(AdjEdge(tid, "ref_id", "id", "fk", 1.0))
    return adj, meta


def test_best_path_direct_hop() -> None:
    adj, meta = _adj_chain(3)
    path = best_path(adj, meta, "t0", "t1", max_depth=3)
    assert path is not None
    assert path.depth == 1
    assert path.from_table_id == "t0"
    assert path.to_table_id == "t1"


def test_best_path_two_hops() -> None:
    adj, meta = _adj_chain(3)
    path = best_path(adj, meta, "t0", "t2", max_depth=3)
    assert path is not None
    assert path.depth == 2


def test_best_path_none_when_disconnected() -> None:
    adj: dict[str, list[AdjEdge]] = {"t0": [], "t1": []}
    meta = {"t0": ("public", "a"), "t1": ("public", "b")}
    assert best_path(adj, meta, "t0", "t1", max_depth=3) is None


def test_best_path_same_node_returns_none() -> None:
    adj, meta = _adj_chain(2)
    assert best_path(adj, meta, "t0", "t0", max_depth=3) is None


def test_best_path_ambiguity_two_equal_paths() -> None:
    # Diamond: t0→t1→t3 and t0→t2→t3, same confidence
    adj: dict[str, list[AdjEdge]] = {
        "t0": [AdjEdge("t1", "id", "fk", "fk", 1.0), AdjEdge("t2", "id", "fk", "fk", 1.0)],
        "t1": [AdjEdge("t0", "fk", "id", "fk", 1.0), AdjEdge("t3", "id", "fk2", "fk", 1.0)],
        "t2": [AdjEdge("t0", "fk", "id", "fk", 1.0), AdjEdge("t3", "id", "fk2", "fk", 1.0)],
        "t3": [AdjEdge("t1", "fk2", "id", "fk", 1.0), AdjEdge("t2", "fk2", "id", "fk", 1.0)],
    }
    meta = {f"t{i}": ("public", f"t{i}") for i in range(4)}
    tied = best_paths(adj, meta, "t0", "t3", max_depth=3, top_k=3)
    assert len(tied) >= 2
    twohop = [p for p in tied if p.depth == 2]
    assert len(twohop) == 2
    intermediates = {p.steps[0].to_table for p in twohop}
    assert intermediates == {"t1", "t2"}


def test_best_paths_single_winner() -> None:
    # Straight chain t0→t1→t2: single path, no tie.
    adj, meta = _adj_chain(3)
    tied = best_paths(adj, meta, "t0", "t2", max_depth=3)
    assert len(tied) == 1
    assert tied[0].ambiguous is False


def test_yen_picks_3hop_fk_over_2hop_inferred() -> None:
    """Regression: low-confidence inferred shortcuts must lose to longer FK chains."""
    adj: dict[str, list[AdjEdge]] = {
        "t0": [
            AdjEdge("t1", "id", "ref", "fk", 1.0),
            AdjEdge("tx", "id", "ref", "inferred", 0.4),
        ],
        "t1": [AdjEdge("t2", "id", "ref", "fk", 1.0)],
        "t2": [AdjEdge("t3", "id", "ref", "fk", 1.0)],
        "tx": [AdjEdge("t3", "id", "ref", "inferred", 0.4)],
    }
    meta = {tid: ("public", tid) for tid in ("t0", "t1", "t2", "t3", "tx")}
    paths = best_paths(adj, meta, "t0", "t3", max_depth=4, top_k=2)
    assert paths
    winner = paths[0]
    assert winner.depth == 3
    assert all(s.edge_type == "fk" for s in winner.steps)


def test_edge_kinds_filter_excludes_inferred() -> None:
    """``edge_kinds=("fk",)`` must drop inferred-only paths entirely."""
    adj: dict[str, list[AdjEdge]] = {
        "t0": [AdjEdge("tx", "id", "ref", "inferred", 0.9)],
        "tx": [AdjEdge("t1", "id", "ref", "inferred", 0.9)],
    }
    meta = {tid: ("public", tid) for tid in ("t0", "tx", "t1")}
    assert best_paths(adj, meta, "t0", "t1", max_depth=4, edge_kinds=("fk",)) == []


def test_max_inferred_hops_cap() -> None:
    """Setting ``max_inferred_hops=0`` rejects any path that needs an inferred edge."""
    adj: dict[str, list[AdjEdge]] = {
        "t0": [AdjEdge("t1", "id", "ref", "inferred", 0.9)],
        "t1": [AdjEdge("t2", "id", "ref", "fk", 1.0)],
    }
    meta = {tid: ("public", tid) for tid in ("t0", "t1", "t2")}
    assert best_paths(adj, meta, "t0", "t2", max_depth=4, max_inferred_hops=0) == []


def test_yen_returns_distinct_alternatives() -> None:
    """Yen's K must return K distinct paths when alternatives exist."""
    adj: dict[str, list[AdjEdge]] = {
        "s": [
            AdjEdge("a", "id", "ref", "fk", 1.0),
            AdjEdge("b", "id", "ref", "fk", 1.0),
            AdjEdge("c", "id", "ref", "fk", 1.0),
        ],
        "a": [AdjEdge("g", "id", "ref", "fk", 1.0)],
        "b": [AdjEdge("g", "id", "ref", "fk", 1.0)],
        "c": [AdjEdge("g", "id", "ref", "fk", 1.0)],
    }
    paths = yen_k_shortest(adj, "s", "g", max_depth=3, k=3)
    assert len(paths) == 3
    intermediates = {edges[0].to_id for _, edges in paths}
    assert intermediates == {"a", "b", "c"}


def test_best_paths_no_duplicate_path_ids() -> None:
    """Parallel FK + INFERRED edges on the same columns must not yield duplicate
    StoredJoinPath rows. Each returned path must have a unique ``path_id``."""
    adj: dict[str, list[AdjEdge]] = {
        "t0": [
            AdjEdge("t1", "id", "ref", "fk", 1.0),
            AdjEdge("t1", "id", "ref", "inferred", 1.0),
        ],
        "t1": [
            AdjEdge("t0", "ref", "id", "fk", 1.0),
            AdjEdge("t0", "ref", "id", "inferred", 1.0),
            AdjEdge("t2", "id", "ref", "fk", 1.0),
            AdjEdge("t2", "id", "ref", "inferred", 1.0),
        ],
        "t2": [
            AdjEdge("t1", "ref", "id", "fk", 1.0),
            AdjEdge("t1", "ref", "id", "inferred", 1.0),
        ],
    }
    meta = {tid: ("public", tid) for tid in ("t0", "t1", "t2")}
    paths = best_paths(adj, meta, "t0", "t2", max_depth=3, top_k=5)
    path_ids = [p.path_id for p in paths]
    assert len(path_ids) == len(set(path_ids)), f"duplicate path_ids: {path_ids}"


def test_edge_cost_formula() -> None:
    fk = AdjEdge("x", "a", "b", "fk", 1.0)
    inf_strong = AdjEdge("x", "a", "b", "inferred", 0.9)
    inf_weak = AdjEdge("x", "a", "b", "inferred", 0.4)
    assert edge_cost(fk) == 1.0
    # 1.0 base + 1.5 inferred = 2.5
    assert abs(edge_cost(inf_strong) - 2.5) < 1e-9
    # 1.0 base + 1.5 inferred + (0.7 - 0.4) low-conf = 2.8
    assert abs(edge_cost(inf_weak) - 2.8) < 1e-9


def test_three_hop_fk_beats_two_hop_with_strong_inferred() -> None:
    """Regression: 2-hop with one conf-1.0 inferred must not beat 3-hop all-FK.

    Prior cost (inferred_penalty=0.5) tied 2-hop-inferred at 2.5 vs 3-hop-FK at
    3.0 and the inferred path won; pagila film→customer and AW
    product→salesorder were the visible regressions.
    """
    adj: dict[str, list[AdjEdge]] = {
        "t0": [
            AdjEdge("t1", "id", "ref", "fk", 1.0),
            AdjEdge("tx", "id", "ref", "inferred", 1.0),
        ],
        "t1": [AdjEdge("t2", "id", "ref", "fk", 1.0)],
        "t2": [AdjEdge("t3", "id", "ref", "fk", 1.0)],
        "tx": [AdjEdge("t3", "id", "ref", "inferred", 1.0)],
    }
    meta = {tid: ("public", tid) for tid in ("t0", "t1", "t2", "t3", "tx")}
    paths = best_paths(adj, meta, "t0", "t3", max_depth=4, top_k=2)
    assert paths
    winner = paths[0]
    assert winner.depth == 3
    assert all(s.edge_type == "fk" for s in winner.steps)


def test_steps_json_roundtrip() -> None:
    import json

    steps = (
        JoinStep(
            from_schema="public",
            from_table="orders",
            to_schema="public",
            to_table="customers",
            from_column="customer_id",
            to_column="id",
            edge_type="fk",
            confidence=1.0,
        ),
    )
    payload = steps_to_json_payload(steps)
    recovered = parse_steps_json(json.dumps(payload))
    assert len(recovered) == 1
    assert recovered[0].from_table == "orders"
    assert recovered[0].to_table == "customers"
    assert recovered[0].edge_type == "fk"
    assert recovered[0].confidence == 1.0
