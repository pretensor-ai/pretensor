"""End-to-end ``traverse_payload`` tests against a pagila-shaped fixture.

Unit tests for ``dijkstra_join_path`` pass in isolation, but live pagila
``traverse(from_table="film", to_table="customer")`` regressed to the 2-hop
inferred shortcut because the full entry point reads precomputed ``JoinPath``
rows first. The cost model legitimately prefers a 1-hop inferred shortcut
(cost ≈ 2.95) over the 3-hop FK chain (cost 3.0) when the inferred edge's
confidence is ≥ ~0.25, so even with ``cross_max`` lifted to match ``intra_max``
the stored winner for pagila ``film → customer`` stays the 1-hop inferred.
The real rescue is the fk-only probe in ``traverse_payload`` that promotes a
pure-FK chain over an inferred stored winner.

These tests drive the full ``traverse_payload`` API with a pagila-shaped
fixture:
  * ``film → inventory → rental → customer`` FK chain (child → parent, so
    film is the FK parent — traversal must walk two FKs in reverse),
  * an ``INFERRED_JOIN`` shortcut ``film → customer`` at confidence 0.25,
  * two clusters so the precompute cross-cluster code path fires.

``GraphBuilder().build()`` already runs the intelligence pipeline (including
``JoinPathEngine.precompute``) once with only FK edges, so we then add the
inferred edge and clear+rerun precompute to mirror a single-shot real index
where the inferred shortcut is discovered before precompute evaluates pairs.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pretensor.cli.constants import REGISTRY_FILENAME
from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.cluster_labeler import ClusterLabeler
from pretensor.intelligence.clustering import Cluster
from pretensor.intelligence.join_paths import JoinPathEngine
from pretensor.mcp.tools.traverse import traverse_payload
from pretensor.staleness.snapshot_store import SnapshotStore

CONN = "pagila_demo"
SCHEMA = "public"


def _id(table: str) -> str:
    return table_node_id(CONN, SCHEMA, table)


def _cols(*names: str) -> list[Column]:
    """Return Column objects; the first is the primary key."""
    return [
        Column(name=n, data_type="int", is_primary_key=(i == 0))
        for i, n in enumerate(names)
    ]


def _fk(
    source_table: str, source_column: str, target_table: str, target_column: str
) -> ForeignKey:
    return ForeignKey(
        source_schema=SCHEMA,
        source_table=source_table,
        source_column=source_column,
        target_schema=SCHEMA,
        target_table=target_table,
        target_column=target_column,
        constraint_name=f"fk_{source_table}_{target_table}",
    )


def _build_pagila_shape(
    tmp_path: Path, *, include_rental_customer_fk: bool = True
) -> Path:
    """Build state_dir with the pagila join-graph shape.

    FK declarations (directional as Postgres has them — child → parent):
        inventory.film_id → film.film_id             (inventory → film)
        rental.inventory_id → inventory.inventory_id (rental → inventory)
        rental.customer_id → customer.customer_id    (rental → customer)

    So a ``film → customer`` join walks two FKs in reverse then one forward.

    Plus an INFERRED_JOIN shortcut ``film → customer`` (confidence 0.25)
    representing a column-name heuristic match that the cost model prefers
    over the 3-hop FK chain unless a fk-only probe intervenes.

    After ``GraphBuilder().build()`` runs the intelligence pipeline's first
    precompute pass (FK-only), we add the inferred edge, clear the stored
    ``JoinPath`` rows, and re-run precompute once so the persisted winner
    reflects the full adjacency (inferred shortcut included) — matching what
    a real ``pretensor index`` produces.
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    film = Table(name="film", schema_name=SCHEMA, columns=_cols("film_id", "title"))
    inventory = Table(
        name="inventory",
        schema_name=SCHEMA,
        columns=_cols("inventory_id", "film_id"),
        foreign_keys=[_fk("inventory", "film_id", "film", "film_id")],
    )
    rental_fks = [_fk("rental", "inventory_id", "inventory", "inventory_id")]
    if include_rental_customer_fk:
        rental_fks.append(_fk("rental", "customer_id", "customer", "customer_id"))
    rental = Table(
        name="rental",
        schema_name=SCHEMA,
        columns=_cols("rental_id", "inventory_id", "customer_id"),
        foreign_keys=rental_fks,
    )
    customer = Table(
        name="customer", schema_name=SCHEMA, columns=_cols("customer_id", "email")
    )

    snapshot = SchemaSnapshot(
        connection_name=CONN,
        database=CONN,
        schemas=[SCHEMA],
        tables=[film, inventory, rental, customer],
        introspected_at=datetime.now(timezone.utc),
    )
    SnapshotStore(state_dir).save(CONN, snapshot)

    graph_path = state_dir / "graphs" / f"{CONN}.kuzu"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph_path)
    try:
        GraphBuilder().build(snapshot, store, run_relationship_discovery=False)

        # Write the inferred shortcut that simulates the pagila column-name
        # heuristic match ``film.film_id ~ customer.customer_id``.
        store.upsert_inferred_join(
            RelationshipCandidate(
                candidate_id=str(uuid.uuid4()),
                source_node_id=_id("film"),
                target_node_id=_id("customer"),
                source_column="film_id",
                target_column="customer_id",
                source="heuristic",
                confidence=0.25,
                reasoning="test-fixture: simulate pagila film→customer shortcut",
            )
        )

        # Two clusters, mirroring the real pagila cluster split (media
        # vs. sales) that exposed the precompute cross-cluster depth cap.
        media_cluster = Cluster(
            table_ids=[_id("film"), _id("inventory")], cohesion_score=0.5
        )
        sales_cluster = Cluster(
            table_ids=[_id("rental"), _id("customer")], cohesion_score=0.5
        )
        asyncio.run(
            ClusterLabeler(store).label_and_persist(
                [media_cluster, sales_cluster], CONN
            )
        )

        # ``GraphBuilder().build()`` already ran precompute once (FK-only) via
        # ``run_intelligence_layer_sync``. Clear those rows and rerun so the
        # persisted winner reflects the full adjacency including the inferred
        # edge we just wrote — matching a single-shot ``pretensor index`` run.
        store._conn.execute("MATCH (p:JoinPath) DELETE p")
        JoinPathEngine(store).precompute(CONN)
    finally:
        store.close()

    reg = GraphRegistry(state_dir / REGISTRY_FILENAME).load()
    reg.upsert(
        connection_name=CONN,
        database=CONN,
        dsn=f"postgres://u@h/{CONN}",
        graph_path=graph_path,
        dialect="postgres",
        table_count=len(snapshot.tables),
    )
    reg.save()
    return state_dir


def _edge_types(path: dict) -> list[str]:
    return [str(s.get("edge_type", "")) for s in (path.get("steps") or [])]


def test_traverse_fk_probe_promotes_over_inferred_stored_winner(
    tmp_path: Path,
) -> None:
    """Real pagila repro: stored winner is the 1-hop inferred; fk-probe wins.

    The cost model prefers a 1-hop inferred shortcut (confidence 0.25 → cost
    2.95) over the 3-hop FK chain (cost 3.0) in Yen's K-shortest ranking, so
    precompute persists the inferred shortcut as the sole stored winner for
    ``film → customer``. Without the fk-probe safety net in
    ``traverse_payload``, the MCP client would see the 2-hop inferred as the
    top path — the first-impression OSS bug.

    The fk-probe must detect that every stored path contains an inferred
    edge, run a pure-FK dijkstra, and promote the 3-hop FK chain to the top
    slot with ``semantic_label="fk_chain_promoted"``.
    """
    state_dir = _build_pagila_shape(tmp_path)

    result = traverse_payload(
        state_dir,
        from_table=f"{SCHEMA}.film",
        to_table=f"{SCHEMA}.customer",
        database=CONN,
        max_depth=4,
    )

    assert "error" not in result, result
    paths = result.get("paths") or []
    assert paths, f"expected at least one path; got {result!r}"

    top = paths[0]
    assert _edge_types(top) == ["fk", "fk", "fk"], (
        f"fk-probe should have promoted the FK chain over the inferred "
        f"stored winner; got {top!r}"
    )
    assert top.get("semantic_label") == "fk_chain_promoted", (
        f"promoted FK chain must carry the fk_chain_promoted label so "
        f"clients can see the stored winner was bypassed; got {top!r}"
    )
    steps = top.get("steps") or []
    assert len(steps) == 3
    assert steps[0]["from_table"].endswith("film")
    assert steps[0]["to_table"].endswith("inventory")
    assert steps[1]["from_table"].endswith("inventory")
    assert steps[1]["to_table"].endswith("rental")
    assert steps[2]["from_table"].endswith("rental")
    assert steps[2]["to_table"].endswith("customer")


def test_traverse_surfaces_inferred_when_no_fk_chain(tmp_path: Path) -> None:
    """Sanity check: remove the rental→customer FK and the inferred shortcut wins.

    This guarantees the fix doesn't over-suppress inferred paths: when no
    pure-FK alternative exists within ``max_depth``, the inferred shortcut
    remains the legitimate answer. The fk-probe runs (stored winner is
    inferred) but finds no FK-only alternative, so no promotion happens.
    """
    state_dir = _build_pagila_shape(tmp_path, include_rental_customer_fk=False)

    result = traverse_payload(
        state_dir,
        from_table=f"{SCHEMA}.film",
        to_table=f"{SCHEMA}.customer",
        database=CONN,
        max_depth=4,
    )

    assert "error" not in result, result
    paths = result.get("paths") or []
    assert paths, f"expected at least one path; got {result!r}"

    top = paths[0]
    assert "inferred" in _edge_types(top), (
        f"expected inferred edge in steps when no FK chain exists; "
        f"got steps={top.get('steps')!r}"
    )
    assert top.get("semantic_label") != "fk_chain_promoted", (
        f"must not promote a non-existent FK chain; got {top!r}"
    )
