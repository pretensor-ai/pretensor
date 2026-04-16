"""Integration tests for dbt ``parent_map`` → Kuzu ``LINEAGE`` edges."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.lineage import write_dbt_lineage
from pretensor.enrichment.dbt.manifest import (
    DbtManifest,
    DbtModel,
    DbtSource,
)
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


def _synthetic_manifest() -> DbtManifest:
    return DbtManifest(
        schema_version="https://schemas.getdbt.com/dbt/manifest/v10.json",
        nodes={
            "model.pkg.stg_users": DbtModel(
                unique_id="model.pkg.stg_users",
                name="stg_users",
                database="analytics",
                schema_name="staging",
                alias="stg_users",
            ),
            "model.pkg.dim_customer": DbtModel(
                unique_id="model.pkg.dim_customer",
                name="dim_customer",
                database="analytics",
                schema_name="marts",
                alias="dim_customer",
            ),
            "model.pkg.fact_orders": DbtModel(
                unique_id="model.pkg.fact_orders",
                name="fact_orders",
                database="analytics",
                schema_name="marts",
                alias="fact_orders",
            ),
        },
        sources={
            "source.pkg.raw.orders": DbtSource(
                unique_id="source.pkg.raw.orders",
                name="orders",
                source_name="raw",
                database="analytics",
                schema_name="raw",
                identifier="orders",
            ),
            "source.pkg.raw.missing_in_graph": DbtSource(
                unique_id="source.pkg.raw.missing_in_graph",
                name="ghost",
                source_name="raw",
                database="analytics",
                schema_name="raw",
                identifier="ghost",
            ),
        },
        exposures={},
        tests={},
        parent_map={
            "model.pkg.stg_users": [],
            "model.pkg.dim_customer": ["model.pkg.stg_users"],
            "model.pkg.fact_orders": [
                "model.pkg.dim_customer",
                "source.pkg.raw.orders",
                "source.pkg.raw.missing_in_graph",
            ],
        },
    )


def test_write_dbt_lineage_model_and_source_parents(memory_store: KuzuStore) -> None:
    cn = "warehouse"
    for node in (
        _minimal_table(cn, "raw", "orders"),
        _minimal_table(cn, "staging", "stg_users"),
        _minimal_table(cn, "marts", "dim_customer"),
        _minimal_table(cn, "marts", "fact_orders"),
    ):
        memory_store.upsert_table(node)

    manifest = _synthetic_manifest()
    n = write_dbt_lineage(manifest, memory_store, cn)
    assert n == 3

    rows = memory_store.query_all_rows(
        """
        MATCH (a:SchemaTable)-[r:LINEAGE]->(b:SchemaTable)
        WHERE r.source = 'dbt' AND r.lineage_type = 'model_dependency'
        RETURN a.table_name, b.table_name
        ORDER BY a.table_name, b.table_name
        """
    )
    assert rows == [
        ("dim_customer", "fact_orders"),
        ("orders", "fact_orders"),
        ("stg_users", "dim_customer"),
    ]


def test_write_dbt_lineage_idempotent(memory_store: KuzuStore) -> None:
    cn = "warehouse"
    for node in (
        _minimal_table(cn, "raw", "orders"),
        _minimal_table(cn, "staging", "stg_users"),
        _minimal_table(cn, "marts", "dim_customer"),
        _minimal_table(cn, "marts", "fact_orders"),
    ):
        memory_store.upsert_table(node)
    manifest = _synthetic_manifest()
    assert write_dbt_lineage(manifest, memory_store, cn) == 3
    assert write_dbt_lineage(manifest, memory_store, cn) == 3
    total = memory_store.query_all_rows(
        """
        MATCH ()-[r:LINEAGE]->()
        WHERE r.source = 'dbt' AND r.lineage_type = 'model_dependency'
        RETURN count(*)
        """
    )
    assert int(total[0][0]) == 3


def test_write_dbt_lineage_skips_missing_schema_table(
    memory_store: KuzuStore, caplog: pytest.LogCaptureFixture
) -> None:
    cn = "warehouse"
    memory_store.upsert_table(_minimal_table(cn, "raw", "orders"))
    manifest = _synthetic_manifest()
    with caplog.at_level("DEBUG"):
        n = write_dbt_lineage(manifest, memory_store, cn)
    assert n == 0
    assert any("SchemaTable missing" in rec.message for rec in caplog.records)
