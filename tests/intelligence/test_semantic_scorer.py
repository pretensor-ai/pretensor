"""Tests for ``EmbeddingRelationshipScorer``.

Pinned contracts:

1.  Null path: ``EmbeddingsConfig.join_threshold=None`` (default) means the
    scorer is never even instantiated; the wider parity test in
    ``tests/intelligence/test_null_path_parity.py`` already enforces no
    ``source='embedding'`` rows in that mode.
2.  On path: pairs with cosine ≥ threshold AND type-family-compatible
    join columns produce ``RelationshipCandidate`` rows with
    ``source="embedding"``, ``status="suggested"``.
3.  Type-family veto: high cosine but incompatible column types → no
    candidate (reuses the heuristic's ``types_compatible``).
4.  Explicit-FK skip: pairs already in ``explicit_fk_keys`` are not
    re-emitted.
5.  No embedded tables → empty candidate list.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.intelligence.discovery import explicit_fk_join_keys
from pretensor.intelligence.scoring import ScorerRegistry
from pretensor.intelligence.semantic import (
    EmbeddingRelationshipScorer,
    _pick_join_columns,
    extend_with_embedding_scorer,
)


def _table(
    name: str,
    columns: list[tuple[str, str]],
    *,
    fks: list[ForeignKey] | None = None,
) -> Table:
    return Table(
        name=name,
        schema_name="public",
        columns=[Column(name=n, data_type=t) for n, t in columns],
        foreign_keys=fks or [],
        comment="",
    )


def _build_two_table_db(
    tmp_path: Path, *, src_id_type: str = "int", dst_id_type: str = "int"
) -> tuple[KuzuStore, str, dict[str, str]]:
    """``customers(id) ← orders(customer_id, ...)``; returns (store, db, nid_by_table)."""
    snap = SchemaSnapshot(
        connection_name="t",
        database="t",
        schemas=["public"],
        tables=[
            _table("customers", [("id", dst_id_type), ("name", "text")]),
            _table("orders", [("id", "int"), ("customer_id", src_id_type)]),
        ],
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    store = KuzuStore(tmp_path / "t.kuzu")
    GraphBuilder().build(snap, store, run_relationship_discovery=False)
    return (
        store,
        "t",
        {
            "customers": table_node_id("t", "public", "customers"),
            "orders": table_node_id("t", "public", "orders"),
        },
    )


def _set_emb(store: KuzuStore, nid: str, vec: list[float]) -> None:
    store.set_table_embedding(nid, vec)


# ── _pick_join_columns ───────────────────────────────────────────────────────


def test_pick_join_columns_canonical_shape() -> None:
    """``orders.customer_id`` → ``customers.id`` is the preferred match."""
    src = _table("orders", [("id", "int"), ("customer_id", "int")])
    dst = _table("customers", [("id", "int"), ("name", "text")])
    assert _pick_join_columns(src, dst) == ("customer_id", "id")


def test_pick_join_columns_type_mismatch_vetoed() -> None:
    """Numeric FK column vs varchar PK → no pair returned."""
    src = _table("orders", [("id", "int"), ("customer_id", "int")])
    dst = _table("customers", [("id", "varchar(36)"), ("name", "text")])
    assert _pick_join_columns(src, dst) is None


def test_pick_join_columns_no_pk_like_returns_none() -> None:
    src = _table("orders", [("id", "int"), ("customer_id", "int")])
    dst = _table("customers", [("name", "text"), ("created_at", "timestamp")])
    assert _pick_join_columns(src, dst) is None


# ── EmbeddingRelationshipScorer ──────────────────────────────────────────────


def test_scorer_emits_candidate_for_high_cosine_pair(tmp_path: Path) -> None:
    store, db, nids = _build_two_table_db(tmp_path)
    try:
        v = [1.0] + [0.0] * 383
        _set_emb(store, nids["customers"], v)
        _set_emb(store, nids["orders"], v)  # cosine = 1.0 > threshold

        scorer = EmbeddingRelationshipScorer(
            store=store, database_key=db, threshold=0.85
        )
        snap = _snapshot_for_db(db)
        candidates = scorer.score(snap, explicit_fk_join_keys(snap))

        # Two directions emitted; only the canonical orders→customers pair has
        # type-compat columns.
        directional = [
            c
            for c in candidates
            if c.source_node_id == nids["orders"]
            and c.target_node_id == nids["customers"]
        ]
        assert len(directional) == 1
        c = directional[0]
        assert c.source == "embedding"
        assert c.status == "suggested"
        assert c.source_column == "customer_id"
        assert c.target_column == "id"
        assert c.confidence > 0.99
    finally:
        store.close()


def test_scorer_skips_pair_below_threshold(tmp_path: Path) -> None:
    store, db, nids = _build_two_table_db(tmp_path)
    try:
        # Orthogonal vectors → cosine ~ 0.
        e1 = [1.0] + [0.0] * 383
        e2 = [0.0, 1.0] + [0.0] * 382
        _set_emb(store, nids["customers"], e1)
        _set_emb(store, nids["orders"], e2)

        scorer = EmbeddingRelationshipScorer(
            store=store, database_key=db, threshold=0.85
        )
        snap = _snapshot_for_db(db)
        assert scorer.score(snap, explicit_fk_join_keys(snap)) == []
    finally:
        store.close()


def test_scorer_no_embeddings_returns_empty(tmp_path: Path) -> None:
    store, db, _ = _build_two_table_db(tmp_path)
    try:
        scorer = EmbeddingRelationshipScorer(
            store=store, database_key=db, threshold=0.85
        )
        snap = _snapshot_for_db(db)
        assert scorer.score(snap, explicit_fk_join_keys(snap)) == []
    finally:
        store.close()


def test_scorer_type_incompat_vetoed(tmp_path: Path) -> None:
    """High cosine but mismatched column types → no candidate emitted."""
    store, db, nids = _build_two_table_db(
        tmp_path, src_id_type="int", dst_id_type="varchar(36)"
    )
    try:
        v = [1.0] + [0.0] * 383
        _set_emb(store, nids["customers"], v)
        _set_emb(store, nids["orders"], v)

        scorer = EmbeddingRelationshipScorer(
            store=store, database_key=db, threshold=0.85
        )
        # Build a matching snapshot — must reflect the actual mixed types so
        # ``_pick_join_columns`` sees the type incompat.
        snap = SchemaSnapshot(
            connection_name="t",
            database="t",
            schemas=["public"],
            tables=[
                _table("customers", [("id", "varchar(36)"), ("name", "text")]),
                _table("orders", [("id", "int"), ("customer_id", "int")]),
            ],
            introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        # No type-compat (FK-like, PK-like) pair → empty.
        assert scorer.score(snap, explicit_fk_join_keys(snap)) == []
    finally:
        store.close()


def test_scorer_skips_explicit_fk_pair(tmp_path: Path) -> None:
    """When a JoinKey is already in the explicit-FK set, scorer doesn't re-emit it."""
    snap = SchemaSnapshot(
        connection_name="t",
        database="t",
        schemas=["public"],
        tables=[
            _table("customers", [("id", "int"), ("name", "text")]),
            _table(
                "orders",
                [("id", "int"), ("customer_id", "int")],
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
        ],
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    store = KuzuStore(tmp_path / "t.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        nids = {
            "customers": table_node_id("t", "public", "customers"),
            "orders": table_node_id("t", "public", "orders"),
        }
        v = [1.0] + [0.0] * 383
        _set_emb(store, nids["customers"], v)
        _set_emb(store, nids["orders"], v)

        scorer = EmbeddingRelationshipScorer(
            store=store, database_key="t", threshold=0.85
        )
        explicit = explicit_fk_join_keys(snap)
        candidates = scorer.score(snap, explicit)

        # No candidate proposes the same JoinKey as the explicit FK.
        for c in candidates:
            jk = (
                c.source_node_id,
                c.target_node_id,
                c.source_column,
                c.target_column,
            )
            assert jk not in explicit
    finally:
        store.close()


# ── threshold validation ─────────────────────────────────────────────────────


def test_scorer_rejects_threshold_below_zero(tmp_path: Path) -> None:
    """Construction with ``threshold < 0.0`` raises ``ValueError``."""
    store, db, _ = _build_two_table_db(tmp_path)
    try:
        with pytest.raises(ValueError, match="threshold"):
            EmbeddingRelationshipScorer(store=store, database_key=db, threshold=-0.1)
    finally:
        store.close()


def test_scorer_rejects_threshold_above_one(tmp_path: Path) -> None:
    """Construction with ``threshold > 1.0`` raises ``ValueError``."""
    store, db, _ = _build_two_table_db(tmp_path)
    try:
        with pytest.raises(ValueError, match="threshold"):
            EmbeddingRelationshipScorer(store=store, database_key=db, threshold=1.01)
    finally:
        store.close()


@pytest.mark.parametrize("good_threshold", [0.0, 0.5, 1.0])
def test_scorer_accepts_valid_threshold(tmp_path: Path, good_threshold: float) -> None:
    """Boundary values (0.0 and 1.0) and typical settings are accepted."""
    store, db, _ = _build_two_table_db(tmp_path)
    try:
        scorer = EmbeddingRelationshipScorer(
            store=store, database_key=db, threshold=good_threshold
        )
        assert scorer.name() == "embedding"
    finally:
        store.close()


# ── extend_with_embedding_scorer ─────────────────────────────────────────────


def test_extend_with_embedding_scorer_threshold_none_returns_base(
    tmp_path: Path,
) -> None:
    """``threshold=None`` is the null-path knob — base registry passes through."""
    store, db, _ = _build_two_table_db(tmp_path)
    try:
        from pretensor.intelligence.heuristic import HeuristicScorer

        base = ScorerRegistry([HeuristicScorer()])
        out = extend_with_embedding_scorer(
            base, store=store, database_key=db, threshold=None
        )
        assert out is base
    finally:
        store.close()


def test_extend_with_embedding_scorer_appends_after_base(tmp_path: Path) -> None:
    """``threshold=0.85`` builds a fresh registry: heuristic first, embedding last."""
    store, db, _ = _build_two_table_db(tmp_path)
    try:
        from pretensor.intelligence.heuristic import HeuristicScorer

        base = ScorerRegistry([HeuristicScorer()])
        out = extend_with_embedding_scorer(
            base, store=store, database_key=db, threshold=0.85
        )
        names = [s.name() for s in out]
        assert names == ["heuristic", "embedding"]
    finally:
        store.close()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _snapshot_for_db(db: str) -> SchemaSnapshot:
    """Reconstruct the canonical two-table snapshot used in this module."""
    return SchemaSnapshot(
        connection_name=db,
        database=db,
        schemas=["public"],
        tables=[
            _table("customers", [("id", "int"), ("name", "text")]),
            _table("orders", [("id", "int"), ("customer_id", "int")]),
        ],
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
