"""Tests for dbt exposures / ``sources.json`` freshness → ``SchemaTable`` signals."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.manifest import (
    DbtExposure,
    DbtManifest,
    DbtModel,
    DbtSource,
    DbtTest,
)
from pretensor.enrichment.dbt.signals import write_dbt_signals
from pretensor.graph_models.node import GraphNode


def _minimal_table(
    connection: str,
    schema: str,
    name: str,
    *,
    database: str = "analytics",
    potentially_unused: bool | None = None,
    has_external_consumers: bool | None = None,
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
        potentially_unused=potentially_unused,
        table_bytes=None,
        clustering_key=None,
        has_external_consumers=has_external_consumers,
    )


@pytest.fixture
def memory_store() -> Iterator[KuzuStore]:
    store = KuzuStore(Path(":memory:"))
    store.ensure_schema()
    try:
        yield store
    finally:
        store.close()


def _manifest_with_exposure() -> DbtManifest:
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
        },
        exposures={
            "exposure.pkg.looker.orders_dash": DbtExposure(
                unique_id="exposure.pkg.looker.orders_dash",
                name="orders_dash",
                depends_on_nodes=("model.pkg.fact_orders",),
            ),
        },
        tests={},
        parent_map={
            "model.pkg.stg_users": [],
            "model.pkg.dim_customer": ["model.pkg.stg_users"],
            "model.pkg.fact_orders": [
                "model.pkg.dim_customer",
                "source.pkg.raw.orders",
            ],
        },
    )


def test_write_dbt_signals_exposure_marks_upstream_tables(
    memory_store: KuzuStore,
) -> None:
    cn = "warehouse"
    for node in (
        _minimal_table(cn, "raw", "orders"),
        _minimal_table(cn, "staging", "stg_users"),
        _minimal_table(cn, "marts", "dim_customer"),
        _minimal_table(cn, "marts", "fact_orders"),
    ):
        memory_store.upsert_table(node)

    stats = write_dbt_signals(_manifest_with_exposure(), memory_store, cn)
    assert stats.exposures_marked == 4

    rows = memory_store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn})
        RETURN t.table_name, t.has_external_consumers
        ORDER BY t.table_name
        """,
        {"cn": cn},
    )
    assert {r[0]: r[1] for r in rows} == {
        "dim_customer": True,
        "fact_orders": True,
        "orders": True,
        "stg_users": True,
    }


def test_write_dbt_signals_no_exposures_is_no_op(memory_store: KuzuStore) -> None:
    cn = "warehouse"
    memory_store.upsert_table(_minimal_table(cn, "raw", "orders"))
    manifest = DbtManifest(
        schema_version=None,
        nodes={},
        sources={},
        exposures={},
        tests={},
        parent_map={},
    )
    stats = write_dbt_signals(manifest, memory_store, cn)
    assert stats.exposures_marked == 0
    assert stats.freshness_rows_applied == 0
    assert stats.tests_counted == 0
    rows = memory_store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn})
        RETURN t.has_external_consumers, t.staleness_status
        """,
        {"cn": cn},
    )
    assert rows[0][0] is None
    assert rows[0][1] is None


def test_write_dbt_signals_fresh_source_sets_staleness_status_pass(
    memory_store: KuzuStore,
    tmp_path: Path,
) -> None:
    cn = "warehouse"
    memory_store.upsert_table(
        _minimal_table(cn, "raw", "orders", potentially_unused=True)
    )
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "unique_id": "source.pkg.raw.orders",
                        "status": "pass",
                        "max_loaded_at": "2024-01-01T00:00:00Z",
                        "max_loaded_at_time_ago_in_s": 60.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = DbtManifest(
        schema_version=None,
        nodes={},
        sources={
            "source.pkg.raw.orders": DbtSource(
                unique_id="source.pkg.raw.orders",
                name="orders",
                source_name="raw",
                database="analytics",
                schema_name="raw",
                identifier="orders",
            ),
        },
        exposures={},
        tests={},
        parent_map={},
    )
    stats = write_dbt_signals(manifest, memory_store, cn, sources_path=sources_path)
    assert stats.freshness_rows_applied == 1
    rows = memory_store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn, table_name: 'orders'})
        RETURN t.potentially_unused, t.staleness_status, t.staleness_as_of
        """,
        {"cn": cn},
    )
    # potentially_unused is untouched by dbt source freshness — that signal is
    # governed by access stats, not the dbt source freshness check.
    assert rows[0][0] is True
    assert rows[0][1] == "pass"
    assert rows[0][2] == "2024-01-01T00:00:00Z"


def test_write_dbt_signals_stale_source_writes_warn_and_logs(
    memory_store: KuzuStore,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cn = "warehouse"
    memory_store.upsert_table(
        _minimal_table(cn, "raw", "orders", potentially_unused=True)
    )
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "unique_id": "source.pkg.raw.orders",
                        "status": "warn",
                        "max_loaded_at": "2020-01-01T00:00:00Z",
                        "max_loaded_at_time_ago_in_s": 999999.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = DbtManifest(
        schema_version=None,
        nodes={},
        sources={
            "source.pkg.raw.orders": DbtSource(
                unique_id="source.pkg.raw.orders",
                name="orders",
                source_name="raw",
                database="analytics",
                schema_name="raw",
                identifier="orders",
            ),
        },
        exposures={},
        tests={},
        parent_map={},
    )
    with caplog.at_level("WARNING"):
        stats = write_dbt_signals(
            manifest, memory_store, cn, sources_path=sources_path
        )
    assert stats.freshness_rows_applied == 1
    rows = memory_store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn, table_name: 'orders'})
        RETURN t.potentially_unused, t.staleness_status, t.staleness_as_of
        """,
        {"cn": cn},
    )
    # potentially_unused must remain untouched — staleness lives in its own column now.
    assert rows[0][0] is True
    assert rows[0][1] == "warn"
    assert rows[0][2] == "2020-01-01T00:00:00Z"
    assert any("stale source" in r.message for r in caplog.records)


def test_write_dbt_signals_tests_counted_per_model(memory_store: KuzuStore) -> None:
    cn = "warehouse"
    memory_store.upsert_table(_minimal_table(cn, "marts", "fact_orders"))
    memory_store.upsert_table(_minimal_table(cn, "marts", "dim_customer"))
    manifest = DbtManifest(
        schema_version=None,
        nodes={
            "model.pkg.fact_orders": DbtModel(
                unique_id="model.pkg.fact_orders",
                name="fact_orders",
                database="analytics",
                schema_name="marts",
                alias="fact_orders",
            ),
            "model.pkg.dim_customer": DbtModel(
                unique_id="model.pkg.dim_customer",
                name="dim_customer",
                database="analytics",
                schema_name="marts",
                alias="dim_customer",
            ),
        },
        sources={},
        exposures={},
        tests={
            "test.pkg.t1": DbtTest(
                unique_id="test.pkg.t1",
                name="not_null_fact_orders_id",
                attached_node="model.pkg.fact_orders",
            ),
            "test.pkg.t2": DbtTest(
                unique_id="test.pkg.t2",
                name="unique_fact_orders_id",
                attached_node="model.pkg.fact_orders",
            ),
            "test.pkg.t3": DbtTest(
                unique_id="test.pkg.t3",
                name="not_null_dim_customer_id",
                attached_node="model.pkg.dim_customer",
            ),
            "test.pkg.orphan": DbtTest(
                unique_id="test.pkg.orphan",
                name="orphan_singular",
                attached_node=None,
            ),
        },
        parent_map={},
    )
    stats = write_dbt_signals(manifest, memory_store, cn)
    assert stats.tests_counted == 2
    rows = memory_store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn})
        RETURN t.table_name, t.test_count
        ORDER BY t.table_name
        """,
        {"cn": cn},
    )
    by_name = {r[0]: r[1] for r in rows}
    assert by_name["fact_orders"] == 2
    assert by_name["dim_customer"] == 1
