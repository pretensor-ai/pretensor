"""Corner-case tests enabled by the pagila_extended fixture.

Tests here exercise capabilities that stock pagila cannot trigger:
- FK chain at depth 4: film -> inventory -> rental -> payment -> payment_audit
- Cross-schema FK: reporting.customer_activity -> public.customer
- Within-DB near-duplicate tables: customer <-> customer_archive
- Multi-schema presence: public + reporting schemas
"""

from __future__ import annotations

from pathlib import Path

from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.intelligence.join_paths.on_demand import (
    best_path,
    build_adjacency,
    table_meta,
)


def _build_graph(tmp_path: Path, load_schema) -> KuzuStore:
    snap = load_schema("pagila_extended")
    store = KuzuStore(tmp_path / "pagila_ext.kuzu")
    GraphBuilder().build(snap, store, run_relationship_discovery=False)
    return store


def test_fixture_has_reporting_schema(load_schema) -> None:
    snap = load_schema("pagila_extended")
    assert "reporting" in snap.schemas
    reporting_tables = [t for t in snap.tables if t.schema_name == "reporting"]
    assert len(reporting_tables) >= 2


def test_fixture_has_near_duplicate_tables(load_schema) -> None:
    snap = load_schema("pagila_extended")
    names = {t.name for t in snap.tables}
    assert "customer" in names
    assert "customer_archive" in names
    archive = next(t for t in snap.tables if t.name == "customer_archive")
    col_names = {c.name for c in archive.columns}
    assert "archived_at" in col_names


def test_fk_chain_depth_four_reachable(tmp_path: Path, load_schema) -> None:
    """film -> inventory -> rental -> payment -> payment_audit must be a valid 4-hop FK path."""
    store = _build_graph(tmp_path, load_schema)
    try:
        adj = build_adjacency(store, "pagila_extended")
        meta = table_meta(store, "pagila_extended")
        film_id = table_node_id("pagila_extended", "public", "film")
        audit_id = table_node_id("pagila_extended", "public", "payment_audit")
        path = best_path(adj, meta, film_id, audit_id, max_depth=5)
        assert path is not None, "expected a join path from film to payment_audit"
        assert path.depth == 4, f"expected depth 4, got {path.depth}"
        assert all(s.edge_type == "fk" for s in path.steps), "expected all FK hops"
    finally:
        store.close()


def test_cross_schema_fk_is_reachable(tmp_path: Path, load_schema) -> None:
    """reporting.customer_activity -> public.customer must be a 1-hop FK path."""
    store = _build_graph(tmp_path, load_schema)
    try:
        adj = build_adjacency(store, "pagila_extended")
        meta = table_meta(store, "pagila_extended")
        activity_id = table_node_id("pagila_extended", "reporting", "customer_activity")
        customer_id = table_node_id("pagila_extended", "public", "customer")
        path = best_path(adj, meta, activity_id, customer_id, max_depth=2)
        assert path is not None, (
            "expected a join path from customer_activity to customer"
        )
        assert path.depth == 1, f"expected depth 1 (direct FK), got {path.depth}"
        assert path.steps[0].from_schema == "reporting"
        assert path.steps[0].to_schema == "public"
    finally:
        store.close()
