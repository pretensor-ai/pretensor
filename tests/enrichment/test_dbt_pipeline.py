"""Tests for ``run_dbt_enrichment`` orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.pipeline import run_dbt_enrichment
from pretensor.graph_models.node import GraphNode


def _minimal_table(
    connection: str, schema: str, name: str, *, database: str = "wh"
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


def test_run_dbt_enrichment_from_manifest_file(
    memory_store: KuzuStore, tmp_path: Path
) -> None:
    cn = "warehouse"
    memory_store.upsert_table(_minimal_table(cn, "dims", "dim_customer"))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "dbt_schema_version": (
                        "https://schemas.getdbt.com/dbt/manifest/v10.json"
                    )
                },
                "nodes": {
                    "model.pkg2.dim_customer": {
                        "resource_type": "model",
                        "name": "dim_customer",
                        "package_name": "pkg2",
                        "database": "wh",
                        "schema": "dims",
                        "description": "Customer dimension",
                        "tags": ["dim"],
                    }
                },
                "sources": {},
                "exposures": {},
                "parent_map": {"model.pkg2.dim_customer": []},
            }
        ),
        encoding="utf-8",
    )

    summary = run_dbt_enrichment(manifest_path, None, memory_store, cn)
    assert summary.lineage_edges == 0
    assert summary.tables_enriched == 1
    assert summary.tags_set == 1
    assert summary.exposures_marked == 0
    assert summary.freshness_rows_applied == 0
    assert summary.tests_counted == 0

    rows = memory_store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn, schema_name: 'dims', table_name: 'dim_customer'})
        RETURN t.description, t.tags
        """,
        {"cn": cn},
    )
    assert rows[0][0] == "Customer dimension"
    assert set(rows[0][1]) == {"dim"}
