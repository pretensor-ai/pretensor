"""Cross-cutting null-path parity tests for the embedding extra.

Every embedding-aware pass (clustering blend, semantic scorer, role
exemplar NN, within-DB dedup) adds a new toggle to ``EmbeddingsConfig``
and a new code path that reads ``SchemaTable.embedding``. This module
pins down two contracts those passes must keep holding:

1.  **Cross-config equivalence.**  Building pagila with two equivalent default
    configs (``PretensorConfig()`` vs ``PretensorConfig(embeddings=EmbeddingsConfig())``)
    must produce byte-identical intelligence-layer state.  Catches any toggle
    whose default drifts to ON.
2.  **No embedding-derived data on null path.**  When every toggle is off,
    no ``SchemaTable.embedding`` is populated and no ``INFERRED_JOIN`` /
    ``SAME_ENTITY`` row carries ``source='embedding'``.  This is the load-bearing
    determinism guarantee from the determinism contract.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pretensor.config import EmbeddingsConfig, PretensorConfig
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.intelligence.pipeline import run_intelligence_layer


def _capture_intelligence_state(
    store: KuzuStore, database_key: str
) -> dict[str, list[tuple[Any, ...]]]:
    """Dump the intelligence-layer state to a canonicalized, sortable shape."""
    embeddings = sorted(
        (str(row[0]), row[1])
        for row in store.query_all_rows(
            "MATCH (t:SchemaTable {database: $db}) RETURN t.node_id, t.embedding",
            {"db": database_key},
        )
    )

    cluster_membership = sorted(
        (str(row[0]), str(row[1]))
        for row in store.query_all_rows(
            "MATCH (t:SchemaTable {database: $db})-[:IN_CLUSTER]->(c:Cluster) "
            "WHERE c.database_key = $db "
            "RETURN t.node_id, c.node_id",
            {"db": database_key},
        )
    )

    inferred_joins = sorted(
        (
            str(row[0]),
            str(row[1]),
            str(row[2] or ""),
            round(float(row[3] or 0.0), 6),
        )
        for row in store.query_all_rows(
            "MATCH (a:SchemaTable {database: $db})-[r:INFERRED_JOIN]->"
            "(b:SchemaTable {database: $db}) "
            "RETURN a.node_id, b.node_id, r.source, r.confidence",
            {"db": database_key},
        )
    )

    role_classifications = sorted(
        (str(row[0]), str(row[1] or ""))
        for row in store.query_all_rows(
            "MATCH (t:SchemaTable {database: $db}) RETURN t.node_id, t.role",
            {"db": database_key},
        )
    )

    # SAME_ENTITY is between Entity nodes. Within a single-DB pagila fixture
    # there should be no SAME_ENTITY edges on the null path (cross-DB
    # resolution is the only producer today; within-DB dedup will become
    # the second producer once it lands and is opted in).
    same_entity = sorted(
        (
            str(row[0]),
            str(row[1]),
            str(row[2] or ""),
            round(float(row[3] or 0.0), 6),
            str(row[4] or ""),
        )
        for row in store.query_all_rows(
            "MATCH (a:Entity {database: $db})-[r:SAME_ENTITY]->(b:Entity) "
            "RETURN a.node_id, b.node_id, r.status, r.score, r.reasoning",
            {"db": database_key},
        )
    )

    return {
        "embeddings": list(embeddings),
        "cluster_membership": list(cluster_membership),
        "inferred_joins": list(inferred_joins),
        "role_classifications": list(role_classifications),
        "same_entity": list(same_entity),
    }


def _build_with_config(tmp_path: Path, name: str, snap: Any, cfg: Any) -> KuzuStore:
    store = KuzuStore(tmp_path / f"{name}.kuzu")
    GraphBuilder().build(snap, store, run_relationship_discovery=True)
    asyncio.run(run_intelligence_layer(store, "pagila", config=cfg))
    return store


def test_null_path_cross_config_equivalence(tmp_path: Path, load_schema: Any) -> None:
    """``PretensorConfig()`` and ``PretensorConfig(embeddings=EmbeddingsConfig())``
    must produce byte-identical intelligence-layer state on pagila.

    Every embedding-related change that adds a toggle to ``EmbeddingsConfig`` must
    keep this green: the default value of the new toggle must produce no
    behavior change versus the prior epoch.
    """
    snap = load_schema("pagila")

    cfg_a = PretensorConfig()  # implicit defaults
    cfg_b = PretensorConfig(embeddings=EmbeddingsConfig())  # explicit defaults

    store_a = _build_with_config(tmp_path, "implicit", snap, cfg_a)
    try:
        snap_a = _capture_intelligence_state(store_a, "pagila")
    finally:
        store_a.close()

    store_b = _build_with_config(tmp_path, "explicit", snap, cfg_b)
    try:
        snap_b = _capture_intelligence_state(store_b, "pagila")
    finally:
        store_b.close()

    for key in snap_a:
        assert snap_a[key] == snap_b[key], (
            f"null-path divergence in {key}: a={snap_a[key]!r} != b={snap_b[key]!r}"
        )


def test_null_path_emits_no_embedding_derived_data(null_embedded_store) -> None:
    """No ``SchemaTable.embedding`` populated, no edge with source='embedding'.

    Forward-compatible guard: the semantic scorer and within-DB dedup paths
    will start emitting ``source='embedding'`` rows when their toggles are on.
    With every toggle off, those rows must not appear.
    """
    state = _capture_intelligence_state(null_embedded_store, "pagila")

    assert state["embeddings"], "expected pagila tables to be present"
    for nid, vec in state["embeddings"]:
        assert vec is None, (
            f"expected embedding=None on null path for {nid}, got a vector"
        )

    embedding_joins = [j for j in state["inferred_joins"] if j[2] == "embedding"]
    assert not embedding_joins, (
        f"unexpected INFERRED_JOIN rows with source='embedding' on null path: "
        f"{embedding_joins}"
    )

    # within-DB dedup will mark its emitted edges with a reasoning that
    # contains "within_db_embedding".  None should appear on the null path.
    embedding_se = [
        e for e in state["same_entity"] if "embedding" in (e[4] or "").lower()
    ]
    assert not embedding_se, (
        f"unexpected SAME_ENTITY rows with embedding-derived reasoning on null path: "
        f"{embedding_se}"
    )


def test_null_path_intelligence_layer_still_produces_output(
    null_embedded_store,
) -> None:
    """Sanity: the rest of the intelligence layer ran on the null path.

    This guards against a regression where disabling embeddings accidentally
    short-circuits classify/cluster/label/join_paths.
    """
    state = _capture_intelligence_state(null_embedded_store, "pagila")

    assert len(state["cluster_membership"]) > 0, (
        "expected at least one cluster→table edge on null path"
    )
    assert len(state["role_classifications"]) > 0, (
        "expected role classifications to be persisted"
    )
    non_empty_roles = [r for _, r in state["role_classifications"] if r]
    assert non_empty_roles, "every table should carry a non-empty role label"


def test_embedded_store_populates_vectors(embedded_store) -> None:
    """Smoke: with the toggle on (and a stub embedder), every table has a vector.

    Mirrors ``test_embedding_index_step.test_on_path_populates_embeddings`` but
    via the shared ``embedded_store`` fixture so the rest of this module — and
    every embedding-related test — share one fixture surface.

    The ``embedded_store`` fixture skips when the ``[embeddings]`` extra is
    absent, so this test no-ops in that environment without an explicit
    ``skipif``.
    """
    rows = embedded_store.query_all_rows(
        "MATCH (t:SchemaTable {database: $db}) RETURN t.node_id, t.embedding",
        {"db": "pagila"},
    )
    assert rows, "expected pagila tables to be present"
    for row in rows:
        nid, emb = row
        assert emb is not None, f"expected embedding for {nid}, got None"
        assert len(emb) == 384, f"expected 384-dim vector for {nid}, got {len(emb)}"
