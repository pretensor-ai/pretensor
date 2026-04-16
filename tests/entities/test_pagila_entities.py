"""Integration-style test: Pagila-shaped snapshot + entity extraction (OSS = empty)."""

from __future__ import annotations

from datetime import datetime, timezone

from tests.query_helpers import first_cell, single_query_result

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore


def _pagila_snap() -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name="pagila",
        database="pagila",
        schemas=["public"],
        tables=[
            Table(
                name="film",
                schema_name="public",
                columns=[Column(name="film_id", data_type="int")],
            ),
            Table(
                name="customer",
                schema_name="public",
                columns=[Column(name="customer_id", data_type="int")],
            ),
            Table(
                name="rental",
                schema_name="public",
                columns=[
                    Column(name="rental_id", data_type="int"),
                    Column(name="inventory_id", data_type="int"),
                ],
            ),
            Table(
                name="payment",
                schema_name="public",
                columns=[Column(name="payment_id", data_type="int")],
            ),
            Table(
                name="stg_events",
                schema_name="public",
                columns=[Column(name="payload", data_type="text")],
            ),
        ],
        introspected_at=datetime.now(timezone.utc),
    )


def test_pagila_entity_extraction_returns_empty_in_oss(tmp_path) -> None:
    """OSS entity extraction always returns empty — no LLM wired."""
    snap = _pagila_snap()
    db_path = tmp_path / "pagila.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(
            snap,
            store,
            run_relationship_discovery=False,
        )
        r = single_query_result(store, "MATCH (e:Entity) RETURN count(*) AS c")
        assert first_cell(r) == 0
    finally:
        store.close()
