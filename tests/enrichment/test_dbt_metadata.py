"""Tests for dbt manifest metadata → ``SchemaTable`` / ``SchemaColumn`` enrichment."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pretensor.core.ids import column_node_id, table_node_id
from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.manifest import DbtManifest, DbtModel, DbtSource
from pretensor.enrichment.dbt.metadata import write_dbt_metadata
from pretensor.graph_models.node import GraphNode


def _minimal_table(
    connection: str, schema: str, name: str, *, database: str = "analytics"
) -> GraphNode:
    return GraphNode(
        node_id=table_node_id(connection, schema, name),
        connection_name=connection,
        database=database,
        schema_name=schema,
        table_name=name,
        row_count=None,
        comment=None,
        entity_type=None,
        table_type="table",
        seq_scan_count=None,
        idx_scan_count=None,
        insert_count=None,
        update_count=None,
        delete_count=None,
        is_partitioned=None,
        partition_key=None,
        grants_json=None,
        access_read_count=None,
        access_write_count=None,
        days_since_last_access=None,
        potentially_unused=None,
        table_bytes=None,
        clustering_key=None,
    )


@pytest.fixture
def memory_store() -> Iterator[KuzuStore]:
    store = KuzuStore(Path(":memory:"))
    store.ensure_schema()
    try:
        yield store
    finally:
        store.close()


def test_write_dbt_metadata_table_description_tags_and_columns(
    memory_store: KuzuStore,
) -> None:
    cn = "warehouse"
    memory_store.upsert_table(_minimal_table(cn, "marts", "orders"))
    tid = table_node_id(cn, "marts", "orders")
    memory_store.upsert_column_for_table(
        column_node_id=column_node_id(cn, "marts", "orders", "id"),
        connection_name=cn,
        database="analytics",
        schema_name="marts",
        table_name="orders",
        column_name="id",
        data_type="bigint",
        nullable=False,
        is_primary_key=True,
        is_foreign_key=False,
        table_node_id=tid,
        ordinal_position=1,
    )

    manifest = DbtManifest(
        schema_version="https://schemas.getdbt.com/dbt/manifest/v10.json",
        nodes={
            "model.pkg.orders": DbtModel(
                unique_id="model.pkg.orders",
                name="orders",
                description="  Orders fact  ",
                tags=("core", "finance"),
                database="analytics",
                schema_name="marts",
                alias="orders",
                column_descriptions={"id": "  surrogate key  "},
            ),
        },
        sources={},
        exposures={},
        tests={},
        parent_map={},
    )

    stats = write_dbt_metadata(manifest, memory_store, cn)
    assert stats.tables_enriched == 1
    assert stats.tags_set == 2

    rows = memory_store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn, schema_name: 'marts', table_name: 'orders'})
        RETURN t.description, t.tags
        """,
        {"cn": cn},
    )
    assert len(rows) == 1
    desc, tags = rows[0]
    assert str(desc).strip() == "Orders fact"
    assert set(tags) == {"core", "finance"}

    crows = memory_store.query_all_rows(
        """
        MATCH (c:SchemaColumn {node_id: $cid})
        RETURN c.description
        """,
        {"cid": column_node_id(cn, "marts", "orders", "id")},
    )
    assert crows[0][0] == "surrogate key"


def test_write_dbt_metadata_preserves_existing_description_and_merges_tags(
    memory_store: KuzuStore,
) -> None:
    cn = "warehouse"
    base = _minimal_table(cn, "raw", "events")
    memory_store.upsert_table(base.model_copy(update={"tags": ["legacy"]}))
    memory_store.execute_write(
        """
        MATCH (t:SchemaTable {connection_name: $cn, schema_name: 'raw', table_name: 'events'})
        SET t.description = $d
        """,
        {"cn": cn, "d": "already set"},
    )

    manifest = DbtManifest(
        schema_version=None,
        nodes={},
        sources={
            "source.pkg.raw.events": DbtSource(
                unique_id="source.pkg.raw.events",
                name="events",
                source_name="raw",
                description="from dbt",
                tags=("ingest", "legacy"),
                database="analytics",
                schema_name="raw",
                identifier="events",
            ),
        },
        exposures={},
        tests={},
        parent_map={},
    )
    stats = write_dbt_metadata(manifest, memory_store, cn)
    assert stats.tables_enriched == 1
    # Only "ingest" is new — "legacy" was already present.
    assert stats.tags_set == 1

    rows = memory_store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn, schema_name: 'raw', table_name: 'events'})
        RETURN t.description, t.tags
        """,
        {"cn": cn},
    )
    desc, tags = rows[0]
    assert desc == "already set"
    assert set(tags) == {"legacy", "ingest"}


def test_write_dbt_metadata_skips_missing_nodes(
    memory_store: KuzuStore, caplog: pytest.LogCaptureFixture
) -> None:
    cn = "warehouse"
    manifest = DbtManifest(
        schema_version=None,
        nodes={
            "model.pkg.ghost": DbtModel(
                unique_id="model.pkg.ghost",
                name="ghost",
                description="n/a",
                tags=("x",),
                database="analytics",
                schema_name="public",
                alias="ghost",
            ),
        },
        sources={},
        exposures={},
        tests={},
        parent_map={},
    )
    with caplog.at_level("DEBUG"):
        stats = write_dbt_metadata(manifest, memory_store, cn)
    assert stats.tables_enriched == 0
    assert stats.tags_set == 0
    assert any("SchemaTable missing" in r.message for r in caplog.records)
