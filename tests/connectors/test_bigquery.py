"""Unit tests for BigQuery connector (mocked ``google.cloud.bigquery``)."""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from pretensor.connectors.bigquery import BigQueryConnector, _jobs_region_dataset
from pretensor.introspection.models.config import (
    ConnectionConfig,
    DatabaseType,
    SchemaFilter,
)


def _install_fake_bigquery() -> None:
    """Minimal stub so ``import google.cloud.bigquery`` succeeds without the real wheel."""
    if "google.cloud.bigquery" in sys.modules:
        return

    class _SchemaField:
        def __init__(
            self,
            name: str,
            field_type: str = "STRING",
            mode: str = "NULLABLE",
            fields: tuple = (),
            description: str | None = None,
        ) -> None:
            self.name = name
            self.field_type = field_type
            self.mode = mode
            self.fields = fields
            self.description = description

    mod = types.ModuleType("google.cloud.bigquery")
    mod.SchemaField = _SchemaField  # type: ignore[attr-defined]

    class _Client:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._query_result = MagicMock()
            self._query_job = MagicMock()
            self._query_job.result.return_value = []

        def query(self, sql: str) -> MagicMock:
            _ = sql
            return self._query_job

        def get_table(self, ref: str) -> MagicMock:
            _ = ref
            return MagicMock()

    mod.Client = _Client  # type: ignore[attr-defined]

    cloud = types.ModuleType("google.cloud")
    cloud.bigquery = mod  # type: ignore[attr-defined]
    sys.modules["google"] = types.ModuleType("google")
    sys.modules["google.cloud"] = cloud
    sys.modules["google.cloud.bigquery"] = mod


def _bq_config() -> ConnectionConfig:
    return ConnectionConfig(
        name="test",
        type=DatabaseType.BIGQUERY,
        host="p1",
        database="p1/ds1",
        schema_filter=SchemaFilter(include=["ds1"]),
        metadata_extra={"bq_project": "p1", "bq_location": "US"},
    )


@pytest.fixture(autouse=True)
def _fake_bigquery_module() -> Iterator[None]:
    _install_fake_bigquery()
    yield


def test_get_foreign_keys_empty() -> None:
    conn = BigQueryConnector(_bq_config())
    assert conn.get_foreign_keys() == []


def test_get_tables_maps_rows() -> None:
    cfg = _bq_config()
    conn = BigQueryConnector(cfg)

    query_rows: dict[str, list[dict[str, object]]] = {
        "TABLE_OPTIONS": [
            {"table_name": "t1", "description": "hello"},
        ],
        "TABLES": [
            {
                "table_schema": "ds1",
                "table_name": "t1",
                "table_type": "BASE TABLE",
                "creation_time": None,
            },
        ],
        "TABLE_STORAGE": [
            {"table_name": "t1", "total_rows": 42},
        ],
    }

    def fake_run(sql: str) -> list[dict[str, object]]:
        if "TABLE_OPTIONS" in sql:
            return query_rows["TABLE_OPTIONS"]
        if "FROM `p1.ds1.INFORMATION_SCHEMA.TABLES`" in sql:
            return query_rows["TABLES"]
        if "TABLE_STORAGE" in sql and "total_rows" in sql:
            return query_rows["TABLE_STORAGE"]
        return []

    conn.connect()
    with patch.object(conn, "_run_query", side_effect=fake_run):
        tables = conn.get_tables()
    conn.disconnect()

    assert len(tables) == 1
    assert tables[0].name == "t1"
    assert tables[0].schema_name == "ds1"
    assert tables[0].row_count == 42
    assert tables[0].comment == "hello"
    assert tables[0].table_type == "table"


def test_get_columns_nested_schema() -> None:
    from google.cloud import bigquery as bq  # type: ignore[import-untyped]

    cfg = _bq_config()
    conn = BigQueryConnector(cfg)

    outer = bq.SchemaField(
        "event",
        field_type="RECORD",
        mode="NULLABLE",
        fields=(
            bq.SchemaField("id", "INT64", "REQUIRED"),
            bq.SchemaField(
                "nested",
                "RECORD",
                "NULLABLE",
                fields=(bq.SchemaField("x", "STRING", "NULLABLE"),),
            ),
        ),
    )
    table_mock = MagicMock()
    table_mock.schema = [outer]

    conn.connect()
    conn.client.get_table = MagicMock(return_value=table_mock)  # type: ignore[method-assign]

    cols = conn.get_columns("t1", "ds1")
    conn.disconnect()

    by_name = {c.name: c for c in cols}
    assert "event" in by_name
    assert by_name["event"].parent_column is None
    assert "event.id" in by_name
    assert by_name["event.id"].parent_column == "event"
    assert "event.nested" in by_name
    assert by_name["event.nested"].parent_column == "event"
    assert "event.nested.x" in by_name
    assert by_name["event.nested.x"].parent_column == "event.nested"


def test_load_deep_catalog_jobs_forbidden_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _bq_config()
    conn = BigQueryConnector(cfg)
    conn.connect()

    def boom(sql: str) -> list[dict[str, object]]:
        if "INFORMATION_SCHEMA.JOBS" in sql:
            raise Exception("403 permission denied")
        if "active_logical_bytes" in sql:
            return [{"table_name": "t1", "active_logical_bytes": 99}]
        if "clustering_ordinal_position" in sql:
            return []
        if "is_partitioning_column" in sql:
            return []
        return []

    caplog.set_level(logging.WARNING)
    with patch.object(conn, "_run_query", side_effect=boom):
        table_ex, _col_ex = conn.load_deep_catalog(cfg.schema_filter)

    conn.disconnect()
    assert any("JOBS" in rec.message for rec in caplog.records)
    assert ("ds1", "t1") in table_ex
    assert table_ex[("ds1", "t1")].get("table_bytes") == 99


def test_load_deep_catalog_jobs_aggregates() -> None:
    from datetime import datetime, timezone

    cfg = _bq_config()
    conn = BigQueryConnector(cfg)
    conn.connect()

    lr = datetime(2026, 1, 1, tzinfo=timezone.utc)
    list_row: dict[str, object] = {"table_name": "t1"}

    def fake_run(sql: str) -> list[dict[str, object]]:
        if "INFORMATION_SCHEMA.JOBS" in sql:
            return [
                {
                    "project_id": "p1",
                    "dataset_id": "ds1",
                    "table_id": "t1",
                    "read_count": 3,
                    "last_read": lr,
                }
            ]
        if "active_logical_bytes" in sql and "COUNT" not in sql:
            return [{"table_name": "t1", "active_logical_bytes": 10}]
        if "clustering_ordinal_position" in sql:
            return []
        if "is_partitioning_column" in sql:
            return [{"table_name": "t1", "partition_key": "dt"}]
        if "FROM `p1.ds1.INFORMATION_SCHEMA.TABLES`" in sql:
            return [list_row]
        return []

    with patch.object(conn, "_run_query", side_effect=fake_run):
        table_ex, _ = conn.load_deep_catalog(cfg.schema_filter)

    conn.disconnect()
    key = ("ds1", "t1")
    assert table_ex[key]["access_read_count"] == 3
    assert table_ex[key]["table_bytes"] == 10
    assert table_ex[key]["is_partitioned"] is True
    assert table_ex[key]["partition_key"] == "dt"


def test_jobs_region_dataset_multi_region() -> None:
    assert _jobs_region_dataset(None) == "region-us"
    assert _jobs_region_dataset("US") == "region-us"
    assert _jobs_region_dataset("us") == "region-us"
    assert _jobs_region_dataset("EU") == "region-eu"
    assert _jobs_region_dataset("eu") == "region-eu"


def test_jobs_region_dataset_single_region() -> None:
    # Single-region locations must NOT be mapped to the multi-region datasets.
    assert _jobs_region_dataset("us-central1") == "region-us-central1"
    assert _jobs_region_dataset("us-east1") == "region-us-east1"
    assert _jobs_region_dataset("europe-west1") == "region-europe-west1"
    assert _jobs_region_dataset("europe-west4") == "region-europe-west4"
    assert _jobs_region_dataset("asia-southeast1") == "region-asia-southeast1"


def test_get_connector_bigquery_import_error() -> None:
    from pretensor.connectors import registry as reg  # noqa: PLC0415

    with patch.object(reg, "import_module", side_effect=ImportError("no module")):
        with pytest.raises(ImportError, match="pretensor\\[bigquery\\]"):
            reg.get_connector(_bq_config())
