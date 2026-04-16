"""Tests for :class:`pretensor.entities.builder.EntityBuilder`."""

from __future__ import annotations

from datetime import datetime, timezone

from tests.query_helpers import first_cell, single_query_result

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import entity_node_id, table_node_id
from pretensor.core.store import KuzuStore
from pretensor.entities.builder import EntityBuilder
from pretensor.entities.llm_extract import ExtractedEntity


def test_entity_builder_writes_nodes_edges_and_entity_type(tmp_path) -> None:
    film = Table(
        name="film",
        schema_name="public",
        columns=[Column(name="film_id", data_type="int")],
    )
    customer = Table(
        name="customer",
        schema_name="public",
        columns=[Column(name="customer_id", data_type="int")],
    )
    snap = SchemaSnapshot(
        connection_name="pagila",
        database="pagila",
        schemas=["public"],
        tables=[film, customer],
        introspected_at=datetime.now(timezone.utc),
    )
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        entities = [
            ExtractedEntity(
                name="Film",
                description="Movies",
                tables=["public.film"],
            ),
            ExtractedEntity(
                name="Customer",
                tables=["customer"],
            ),
        ]
        EntityBuilder().build(entities, store, snap)

        ne = single_query_result(store, "MATCH (e:Entity) RETURN count(*) AS c")
        assert first_cell(ne) == 2

        re = single_query_result(
            store, "MATCH ()-[r:REPRESENTS]->() RETURN count(*) AS c"
        )
        assert first_cell(re) == 2

        tid = table_node_id("pagila", "public", "film")
        r = single_query_result(
            store,
            "MATCH (t:SchemaTable {node_id: $id}) RETURN t.entity_type AS et",
            {"id": tid},
        )
        assert first_cell(r) == "Film"

        eid = entity_node_id("pagila", "Film")
        r2 = single_query_result(
            store,
            "MATCH (e:Entity {node_id: $id}) RETURN e.name AS n",
            {"id": eid},
        )
        assert first_cell(r2) == "Film"
    finally:
        store.close()
