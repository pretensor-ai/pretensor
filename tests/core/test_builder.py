"""Tests for GraphBuilder."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from tests.query_helpers import first_cell, single_query_result

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.core.builder import GraphBuilder, fk_edge_id, table_node_id
from pretensor.core.store import KuzuStore


def _minimal_snapshot() -> SchemaSnapshot:
    t_orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="user_id", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="orders",
                source_column="user_id",
                target_schema="public",
                target_table="users",
                target_column="id",
            )
        ],
    )
    t_users = Table(
        name="users",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    return SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_orders, t_users],
        introspected_at=datetime.now(timezone.utc),
    )


def test_builder_writes_nodes_and_fk_edges(tmp_path) -> None:
    snap = _minimal_snapshot()
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store)
        r = single_query_result(store, "MATCH (t:SchemaTable) RETURN count(*) AS c")
        assert first_cell(r) == 2

        r2 = single_query_result(
            store, "MATCH ()-[e:FK_REFERENCES]->() RETURN count(*) AS c"
        )
        assert first_cell(r2) == 1

        r3 = single_query_result(
            store, "MATCH ()-[e:INFERRED_JOIN]->() RETURN count(*) AS c"
        )
        assert first_cell(r3) == 0
    finally:
        store.close()


def test_table_node_id_stable() -> None:
    assert table_node_id("c", "s", "t") == "c::s::t"


def test_fk_edge_id_stable() -> None:
    fk = ForeignKey(
        source_schema="public",
        source_table="orders",
        source_column="user_id",
        target_schema="public",
        target_table="users",
        target_column="id",
    )
    assert fk_edge_id(fk) == "public.orders.user_id->public.users.id"


def test_fk_constraint_name_preserved(tmp_path) -> None:
    """constraint_name on ForeignKey flows through the builder to FK_REFERENCES edges."""
    t_lineitem = Table(
        name="lineitem",
        schema_name="public",
        columns=[
            Column(name="l_orderkey", data_type="int", is_foreign_key=True),
            Column(name="l_partkey", data_type="int", is_foreign_key=True),
            Column(name="l_suppkey", data_type="int", is_foreign_key=True),
        ],
        foreign_keys=[
            ForeignKey(
                constraint_name="fk_lineitem_orders",
                source_schema="public",
                source_table="lineitem",
                source_column="l_orderkey",
                target_schema="public",
                target_table="orders",
                target_column="o_orderkey",
            ),
            ForeignKey(
                constraint_name="fk_lineitem_partsupp",
                source_schema="public",
                source_table="lineitem",
                source_column="l_partkey",
                target_schema="public",
                target_table="partsupp",
                target_column="ps_partkey",
            ),
            ForeignKey(
                constraint_name="fk_lineitem_partsupp",
                source_schema="public",
                source_table="lineitem",
                source_column="l_suppkey",
                target_schema="public",
                target_table="partsupp",
                target_column="ps_suppkey",
            ),
        ],
    )
    t_orders = Table(
        name="orders",
        schema_name="public",
        columns=[Column(name="o_orderkey", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    t_partsupp = Table(
        name="partsupp",
        schema_name="public",
        columns=[
            Column(name="ps_partkey", data_type="int", is_primary_key=True),
            Column(name="ps_suppkey", data_type="int", is_primary_key=True),
        ],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_lineitem, t_orders, t_partsupp],
        introspected_at=datetime.now(timezone.utc),
    )
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store)
        r = single_query_result(
            store, "MATCH ()-[e:FK_REFERENCES]->() RETURN count(*) AS c"
        )
        assert first_cell(r) == 3

        # Verify constraint_name is persisted in the graph
        rows = store.query_all_rows(
            "MATCH ()-[e:FK_REFERENCES]->() "
            "WHERE e.constraint_name = $cn "
            "RETURN e.source_column ORDER BY e.source_column",
            {"cn": "fk_lineitem_partsupp"},
        )
        assert len(rows) == 2
        assert rows[0][0] == "l_partkey"
        assert rows[1][0] == "l_suppkey"
    finally:
        store.close()


def test_builder_stores_catalog_enrichment_fields(tmp_path) -> None:
    """column_cardinality, index_type, index_is_unique are persisted to Kuzu."""
    t = Table(
        name="products",
        schema_name="public",
        columns=[
            Column(
                name="id",
                data_type="int",
                is_primary_key=True,
                column_cardinality=1000,
                index_type="btree",
                index_is_unique=True,
            ),
            Column(
                name="category",
                data_type="varchar",
                column_cardinality=12,
                index_type="gin",
                index_is_unique=False,
            ),
            Column(
                name="description",
                data_type="text",
                # no catalog enrichment — all None
            ),
        ],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t],
        introspected_at=datetime.now(timezone.utc),
    )
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store)

        r = single_query_result(
            store,
            "MATCH (col:SchemaColumn {column_name: 'id'}) "
            "RETURN col.column_cardinality, col.index_type, col.index_is_unique",
        )
        row = list(r.get_next())
        assert row[0] == 1000
        assert row[1] == "btree"
        assert row[2] is True

        r2 = single_query_result(
            store,
            "MATCH (col:SchemaColumn {column_name: 'category'}) "
            "RETURN col.column_cardinality, col.index_type, col.index_is_unique",
        )
        row2 = list(r2.get_next())
        assert row2[0] == 12
        assert row2[1] == "gin"
        assert row2[2] is False

        r3 = single_query_result(
            store,
            "MATCH (col:SchemaColumn {column_name: 'description'}) "
            "RETURN col.column_cardinality, col.index_type, col.index_is_unique",
        )
        row3 = list(r3.get_next())
        assert row3[0] is None
        assert row3[1] is None
        assert row3[2] is None
    finally:
        store.close()


def test_builder_handles_nested_columns_out_of_order(tmp_path) -> None:
    """Child columns arriving before their parent must still be written via topo-sort."""
    t = Table(
        name="events",
        schema_name="public",
        columns=[
            # child arrives before parent — exercises the deferred-pass loop
            Column(name="event.id", data_type="INT64", parent_column="event"),
            Column(name="event", data_type="RECORD"),
        ],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="bq",
        database="bq/ds",
        schemas=["public"],
        tables=[t],
        introspected_at=datetime.now(timezone.utc),
    )
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        GraphBuilder().build(snap, store)
        r = single_query_result(store, "MATCH (col:SchemaColumn) RETURN count(*) AS c")
        assert first_cell(r) == 2
        r2 = single_query_result(
            store, "MATCH ()-[e:HAS_SUBCOLUMN]->() RETURN count(*) AS c"
        )
        assert first_cell(r2) == 1
    finally:
        store.close()


def test_builder_warns_on_unresolvable_parent_column(tmp_path, caplog) -> None:
    """Columns referencing a missing parent_column are warned and skipped."""
    t = Table(
        name="events",
        schema_name="public",
        columns=[
            Column(name="orphan.x", data_type="STRING", parent_column="nonexistent"),
        ],
        foreign_keys=[],
    )
    snap = SchemaSnapshot(
        connection_name="bq",
        database="bq/ds",
        schemas=["public"],
        tables=[t],
        introspected_at=datetime.now(timezone.utc),
    )
    db_path = tmp_path / "g.kuzu"
    store = KuzuStore(db_path)
    try:
        with caplog.at_level(logging.WARNING, logger="pretensor.core.builder"):
            GraphBuilder().build(snap, store)
        assert any("unresolvable" in rec.message for rec in caplog.records)
        r = single_query_result(store, "MATCH (col:SchemaColumn) RETURN count(*) AS c")
        assert first_cell(r) == 0
    finally:
        store.close()
