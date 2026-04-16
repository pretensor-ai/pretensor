"""BigQuery introspection via ``google-cloud-bigquery`` (optional extra)."""

from __future__ import annotations

import importlib
import logging
from datetime import datetime, timezone
from typing import Any

from pretensor.connectors.base import (
    BaseConnector,
    ColumnInfo,
    ColumnStats,
    ForeignKeyInfo,
    TableInfo,
)
from pretensor.introspection.models.config import ConnectionConfig, SchemaFilter

logger = logging.getLogger(__name__)

_STALE_ACCESS_DAYS = 90
_JOBS_LOOKBACK_DAYS = 90
_NESTED_SCHEMA_MAX_DEPTH = 3

_BIGQUERY_EXTRA_HINT = (
    "Install the BigQuery extra: pip install 'pretensor[bigquery]' "
    "(or pip install google-cloud-bigquery)."
)


class BigQueryConnectorError(Exception):
    """Raised when BigQuery connector operations fail."""


def _ensure_bigquery() -> Any:
    try:
        return importlib.import_module("google.cloud.bigquery")
    except ImportError as exc:
        msg = f"BigQuery connector requires google-cloud-bigquery. {_BIGQUERY_EXTRA_HINT}"
        raise ImportError(msg) from exc


def _bq_table_type(raw: str | None) -> str:
    if not raw:
        return "table"
    u = raw.upper()
    if u == "BASE TABLE":
        return "table"
    if u == "VIEW":
        return "view"
    if u == "MATERIALIZED VIEW":
        return "materialized_view"
    if u == "EXTERNAL":
        return "foreign_table"
    return raw.lower().replace(" ", "_")


def _jobs_region_dataset(location: str | None) -> str:
    """Region qualifier for project-level ``INFORMATION_SCHEMA.JOBS``.

    Only the BigQuery multi-region identifiers (``US`` and ``EU``) have
    dedicated ``region-us`` / ``region-eu`` INFORMATION_SCHEMA datasets.
    All single-region locations (e.g. ``us-central1``, ``europe-west1``)
    must use their own ``region-<location>`` qualifier.
    """
    if not location:
        return "region-us"
    loc = location.strip().upper()
    if loc == "US":
        return "region-us"
    if loc == "EU":
        return "region-eu"
    return f"region-{location.strip().lower()}"


def _schema_field_to_columns(
    field: Any,
    *,
    parent_path: str | None,
    depth: int,
    ordinal_counter: list[int],
) -> list[ColumnInfo]:
    """Flatten ``SchemaField`` tree into :class:`ColumnInfo` rows (depth-capped)."""
    out: list[ColumnInfo] = []
    name = field.name
    full_name = f"{parent_path}.{name}" if parent_path else name
    mode = (getattr(field, "mode", None) or "").upper()
    is_repeated = mode == "REPEATED"
    field_type = (getattr(field, "field_type", None) or "").upper()
    is_record = field_type in ("RECORD", "STRUCT")

    bq_type = str(getattr(field, "field_type", "") or "STRING")
    if is_repeated and not is_record:
        data_type = f"ARRAY<{bq_type}>"
    elif is_record:
        data_type = "RECORD"
    else:
        data_type = bq_type

    nullable = mode != "REQUIRED" and not is_repeated
    ordinal_counter[0] += 1
    ord_pos = ordinal_counter[0]
    out.append(
        ColumnInfo(
            name=full_name,
            data_type=data_type,
            nullable=nullable,
            is_primary_key=False,
            default_value=None,
            comment=getattr(field, "description", None) or None,
            is_indexed=False,
            ordinal_position=ord_pos,
            parent_column=parent_path,
            is_array=is_repeated,
        )
    )
    subfields = list(getattr(field, "fields", ()) or ())
    if not subfields or depth >= _NESTED_SCHEMA_MAX_DEPTH:
        return out
    for sub in subfields:
        out.extend(
            _schema_field_to_columns(
                sub,
                parent_path=full_name,
                depth=depth + 1,
                ordinal_counter=ordinal_counter,
            )
        )
    return out


class BigQueryConnector(BaseConnector):
    """Introspect BigQuery via INFORMATION_SCHEMA and table metadata API."""

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._client: Any = None

    @property
    def project_id(self) -> str:
        extra = self.config.metadata_extra or {}
        pid = (extra.get("bq_project") or self.config.host or "").strip()
        if not pid:
            raise BigQueryConnectorError(
                "BigQuery project id is missing from connection config"
            )
        return pid

    @property
    def dataset_id(self) -> str:
        sf = self.config.schema_filter
        if sf.include and len(sf.include) == 1:
            return sf.include[0]
        db = (self.config.database or "").strip()
        if "/" in db:
            return db.split("/", 1)[1]
        raise BigQueryConnectorError(
            "BigQuery dataset id is required (DSN path bigquery://project/dataset)"
        )

    @property
    def location(self) -> str | None:
        extra = self.config.metadata_extra or {}
        loc = extra.get("bq_location")
        return str(loc).strip() if loc else None

    def _fq(self, suffix: str) -> str:
        return f"`{self.project_id}.{self.dataset_id}.{suffix}`"

    def _fq_jobs(self) -> str:
        region_ds = _jobs_region_dataset(self.location)
        return f"`{self.project_id}.{region_ds}.INFORMATION_SCHEMA.JOBS`"

    def connect(self) -> None:
        bigquery = _ensure_bigquery()
        try:
            client_kw: dict[str, Any] = {"project": self.project_id}
            if self.location:
                client_kw["location"] = self.location
            self._client = bigquery.Client(**client_kw)
            list(self._client.query("SELECT 1").result())
            logger.info(
                "Connected to BigQuery project %s dataset %s",
                self.project_id,
                self.dataset_id,
            )
        except Exception as exc:
            raise BigQueryConnectorError(f"Failed to connect to BigQuery: {exc}") from exc

    def disconnect(self) -> None:
        self._client = None

    @property
    def client(self) -> Any:
        if self._client is None:
            raise BigQueryConnectorError("Not connected.")
        return self._client

    def _run_query(self, sql: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        job = self.client.query(sql)
        for row in job.result(page_size=1000):
            rows.append(dict(row.items()))
        return rows

    def _schema_ok(self, schema_filter: SchemaFilter, schema_name: str) -> bool:
        if schema_filter.include and schema_name not in schema_filter.include:
            return False
        if schema_filter.exclude and schema_name in schema_filter.exclude:
            return False
        return True

    def get_tables(self, schema_filter: SchemaFilter | None = None) -> list[TableInfo]:
        sf = schema_filter or self.config.schema_filter
        desc_sql = f"""
            SELECT table_name, option_value AS description
            FROM {self._fq("INFORMATION_SCHEMA.TABLE_OPTIONS")}
            WHERE table_schema = '{self.dataset_id}'
              AND option_name = 'description'
        """
        descriptions: dict[str, str] = {}
        try:
            for row in self._run_query(desc_sql):
                tname = row.get("table_name")
                desc = row.get("description")
                if tname and desc:
                    descriptions[str(tname)] = str(desc)
        except Exception as exc:
            logger.warning("BigQuery TABLE_OPTIONS query skipped: %s", exc)

        tables_sql = f"""
            SELECT table_schema, table_name, table_type, creation_time
            FROM {self._fq("INFORMATION_SCHEMA.TABLES")}
            WHERE table_schema = '{self.dataset_id}'
            ORDER BY table_name
        """
        storage_sql = f"""
            SELECT table_name, total_rows
            FROM {self._fq("INFORMATION_SCHEMA.TABLE_STORAGE")}
            WHERE table_schema = '{self.dataset_id}'
        """

        storage: dict[str, int | None] = {}
        try:
            for row in self._run_query(storage_sql):
                tn = str(row["table_name"])
                tr = row.get("total_rows")
                storage[tn] = int(tr) if tr is not None else None
        except Exception as exc:
            logger.warning("BigQuery TABLE_STORAGE row counts skipped: %s", exc)

        out: list[TableInfo] = []
        for row in self._run_query(tables_sql):
            schema_name = str(row.get("table_schema") or self.dataset_id)
            if not self._schema_ok(sf, schema_name):
                continue
            tname = str(row["table_name"])
            out.append(
                TableInfo(
                    name=tname,
                    schema_name=schema_name,
                    row_count=storage.get(tname),
                    comment=descriptions.get(tname),
                    table_type=_bq_table_type(
                        str(row["table_type"]) if row.get("table_type") else None
                    ),
                )
            )
        return out

    def get_columns(self, table_name: str, schema_name: str) -> list[ColumnInfo]:
        _ = schema_name
        ref = f"{self.project_id}.{self.dataset_id}.{table_name}"
        try:
            table = self.client.get_table(ref)
        except Exception as exc:
            raise BigQueryConnectorError(
                f"Failed to load BigQuery table schema for {ref}: {exc}"
            ) from exc
        cols: list[ColumnInfo] = []
        counter = [0]
        for field in table.schema:
            cols.extend(
                _schema_field_to_columns(
                    field, parent_path=None, depth=1, ordinal_counter=counter
                )
            )
        return cols

    def get_foreign_keys(self) -> list[ForeignKeyInfo]:
        return []

    def get_table_row_count(self, table_name: str, schema_name: str) -> int:
        tables = self.get_tables(
            SchemaFilter(include=[schema_name]) if schema_name else None
        )
        for t in tables:
            if t.name == table_name and t.schema_name == schema_name:
                return int(t.row_count or 0)
        return 0

    def get_column_stats(
        self, table_name: str, column_name: str, schema_name: str
    ) -> ColumnStats:
        _ = (table_name, column_name, schema_name)
        return ColumnStats()

    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        return self._run_query(sql)

    def load_deep_catalog(
        self, schema_filter: SchemaFilter
    ) -> tuple[
        dict[tuple[str, str], dict[str, Any]],
        dict[tuple[str, str, str], dict[str, Any]],
    ]:
        """TABLE_STORAGE bytes, partition/cluster hints, and JOBS read aggregates."""
        table_extra: dict[tuple[str, str], dict[str, Any]] = {}
        column_extra: dict[tuple[str, str, str], dict[str, Any]] = {}

        def _ok(sn: str) -> bool:
            return self._schema_ok(schema_filter, sn)

        storage_sql = f"""
            SELECT table_name, active_logical_bytes
            FROM {self._fq("INFORMATION_SCHEMA.TABLE_STORAGE")}
            WHERE table_schema = '{self.dataset_id}'
        """
        try:
            for row in self._run_query(storage_sql):
                tn = str(row["table_name"])
                key = (self.dataset_id, tn)
                if not _ok(self.dataset_id):
                    continue
                b = row.get("active_logical_bytes")
                if b is not None:
                    table_extra.setdefault(key, {})["table_bytes"] = int(b)
        except Exception as exc:
            logger.warning("BigQuery TABLE_STORAGE bytes skipped: %s", exc)

        cluster_sql = f"""
            SELECT
              table_name,
              STRING_AGG(column_name ORDER BY clustering_ordinal_position) AS clustering_key
            FROM {self._fq("INFORMATION_SCHEMA.COLUMNS")}
            WHERE table_schema = '{self.dataset_id}'
              AND clustering_ordinal_position IS NOT NULL
            GROUP BY table_name
        """
        try:
            for row in self._run_query(cluster_sql):
                tn = row.get("table_name")
                ck = row.get("clustering_key")
                if tn and ck:
                    key = (self.dataset_id, str(tn))
                    if _ok(self.dataset_id):
                        table_extra.setdefault(key, {})["clustering_key"] = str(ck)
        except Exception as exc:
            logger.warning("BigQuery clustering metadata skipped: %s", exc)

        part_sql = f"""
            SELECT
              table_name,
              STRING_AGG(column_name ORDER BY ordinal_position) AS partition_key
            FROM {self._fq("INFORMATION_SCHEMA.COLUMNS")}
            WHERE table_schema = '{self.dataset_id}'
              AND is_partitioning_column = 'YES'
            GROUP BY table_name
        """
        try:
            for row in self._run_query(part_sql):
                tn = row.get("table_name")
                pk = row.get("partition_key")
                if tn and pk:
                    key = (self.dataset_id, str(tn))
                    if _ok(self.dataset_id):
                        table_extra.setdefault(key, {}).update(
                            {
                                "is_partitioned": True,
                                "partition_key": str(pk),
                            }
                        )
        except Exception as exc:
            logger.warning("BigQuery partition metadata skipped: %s", exc)

        jobs_sql = f"""
            SELECT
              ref.project_id AS project_id,
              ref.dataset_id AS dataset_id,
              ref.table_id AS table_id,
              COUNT(*) AS read_count,
              MAX(j.creation_time) AS last_read
            FROM {self._fq_jobs()} AS j,
            UNNEST(j.referenced_tables) AS ref
            WHERE j.job_type = 'QUERY'
              AND j.creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {_JOBS_LOOKBACK_DAYS} DAY)
              AND (j.cache_hit IS NOT TRUE)
            GROUP BY project_id, dataset_id, table_id
        """
        jobs_ok = False
        try:
            read_rows = self._run_query(jobs_sql)
            jobs_ok = True
        except Exception as exc:
            logger.warning(
                "BigQuery INFORMATION_SCHEMA.JOBS access skipped (permissions or region): %s",
                exc,
            )
            read_rows = []

        now = datetime.now(timezone.utc)
        for row in read_rows:
            pid = (row.get("project_id") or "").strip()
            did = (row.get("dataset_id") or "").strip()
            tid = (row.get("table_id") or "").strip()
            if pid != self.project_id or did != self.dataset_id or not tid:
                continue
            if not _ok(self.dataset_id):
                continue
            key = (self.dataset_id, tid)
            rc = int(row.get("read_count") or 0)
            last_read = row.get("last_read")
            days_since: int | None = None
            if last_read is not None:
                lr = last_read
                if isinstance(lr, datetime):
                    if lr.tzinfo is None:
                        lr = lr.replace(tzinfo=timezone.utc)
                    days_since = max(0, int((now - lr.astimezone(timezone.utc)).days))
            potentially_unused = rc == 0 and (
                days_since is None or days_since > _STALE_ACCESS_DAYS
            )
            table_extra.setdefault(key, {}).update(
                {
                    "access_read_count": rc,
                    "days_since_last_access": days_since,
                    "potentially_unused": potentially_unused,
                }
            )

        if jobs_ok:
            list_sql = f"""
                SELECT table_name
                FROM {self._fq("INFORMATION_SCHEMA.TABLES")}
                WHERE table_schema = '{self.dataset_id}'
            """
            try:
                table_names = [str(r["table_name"]) for r in self._run_query(list_sql)]
            except Exception:
                table_names = []
            for tn in table_names:
                if not _ok(self.dataset_id):
                    continue
                key = (self.dataset_id, tn)
                entry = table_extra.setdefault(key, {})
                if "access_read_count" not in entry:
                    entry["access_read_count"] = 0
                if "days_since_last_access" not in entry:
                    entry["days_since_last_access"] = None
                if "potentially_unused" not in entry:
                    entry["potentially_unused"] = True

        return table_extra, column_extra
