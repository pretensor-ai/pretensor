"""Full-schema introspection via the connector registry."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from pretensor.connectors.base import ForeignKeyInfo
from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
    ViewDependency,
)
from pretensor.connectors.registry import get_connector
from pretensor.introspection.models.config import ConnectionConfig

logger = logging.getLogger(__name__)

_PROFILE_INDEX = os.environ.get("PRETENSOR_PROFILE_INDEX", "").lower() not in (
    "",
    "0",
    "false",
    "no",
)

LOW_CARDINALITY_THRESHOLD = 50
SKIP_STATS_ROW_THRESHOLD = 0
LARGE_TABLE_ROW_THRESHOLD = 10_000_000

# Column types whose values cannot be aggregated with MIN/MAX or cast to text
# in a meaningful way for stats collection. Pre-filtering avoids noisy
# "Failed to collect stats" warnings on benign type mismatches.
_STATS_UNSUPPORTED_TYPES: frozenset[str] = frozenset({
    "boolean",
    "bool",
    "bytea",
    "tsvector",
    "tsquery",
    "xml",
    "json",
    "point",
    "line",
    "lseg",
    "box",
    "path",
    "polygon",
    "circle",
    "pg_lsn",
    "txid_snapshot",
})


def _index_foreign_keys(
    fks: list[ForeignKeyInfo],
) -> dict[tuple[str, str], list[ForeignKeyInfo]]:
    index: dict[tuple[str, str], list[ForeignKeyInfo]] = {}
    for fk in fks:
        key = (fk.source_schema, fk.source_table)
        index.setdefault(key, []).append(fk)
    return index


def _fk_columns_for_table(
    fk_index: dict[tuple[str, str], list[ForeignKeyInfo]],
    schema_name: str,
    table_name: str,
) -> set[str]:
    return {fk.source_column for fk in fk_index.get((schema_name, table_name), [])}


def inspect(config: ConnectionConfig) -> SchemaSnapshot:
    """Run full schema introspection and return a :class:`SchemaSnapshot`.

    Dispatches to the connector registered for ``config.type``.

    Args:
        config: Database connection configuration.

    Returns:
        Tables, columns, FKs, and optional column statistics.
    """
    start = time.monotonic()
    connector = get_connector(config)

    with connector:
        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        tables_info = connector.get_tables()
        if _PROFILE_INDEX:
            print(
                f"[profile] inspect.get_tables: {(time.perf_counter() - _t) * 1000:.0f}ms ({len(tables_info)} tables)",
                flush=True,
            )
        logger.info("Discovered %d tables", len(tables_info))

        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        all_fks = connector.get_foreign_keys()
        if _PROFILE_INDEX:
            print(
                f"[profile] inspect.get_foreign_keys: {(time.perf_counter() - _t) * 1000:.0f}ms ({len(all_fks)} fks)",
                flush=True,
            )
        fk_index = _index_foreign_keys(all_fks)
        sf = config.schema_filter
        try:
            _t = time.perf_counter() if _PROFILE_INDEX else 0.0
            table_catalog_extra, column_catalog_extra = connector.load_deep_catalog(sf)
            if _PROFILE_INDEX:
                print(
                    f"[profile] inspect.load_deep_catalog: {(time.perf_counter() - _t) * 1000:.0f}ms",
                    flush=True,
                )
        except Exception:
            logger.warning("Deep catalog enrichment failed; continuing without it", exc_info=True)
            table_catalog_extra, column_catalog_extra = {}, {}

        schemas_seen: set[str] = set()
        tables: list[Table] = []

        # Per-table loop instrumentation accumulators
        _columns_total = 0.0
        _stats_total = 0.0
        _stats_calls = 0
        _stats_max = 0.0
        _stats_max_target = ""
        _loop_start = time.perf_counter() if _PROFILE_INDEX else 0.0

        for idx, table_info in enumerate(tables_info, 1):
            tname = table_info.name
            sname = table_info.schema_name
            schemas_seen.add(sname)

            logger.info(
                "Introspecting [%d/%d] %s.%s",
                idx,
                len(tables_info),
                sname,
                tname,
            )
            if _PROFILE_INDEX:
                print(
                    f"[profile] inspect.table[{idx}/{len(tables_info)}] {sname}.{tname} (row_count={table_info.row_count})",
                    flush=True,
                )

            _t = time.perf_counter() if _PROFILE_INDEX else 0.0
            columns_info = connector.get_columns(tname, sname)
            if _PROFILE_INDEX:
                _columns_total += time.perf_counter() - _t
            row_count = table_info.row_count

            fk_columns = _fk_columns_for_table(fk_index, sname, tname)

            columns: list[Column] = []
            for col_info in columns_info:
                stats_kwargs: dict[str, Any] = {}
                cat_col = column_catalog_extra.get((sname, tname, col_info.name))

                col_type = (col_info.data_type or "").lower()
                should_collect_stats = (
                    row_count is not None
                    and row_count > SKIP_STATS_ROW_THRESHOLD
                    and row_count < LARGE_TABLE_ROW_THRESHOLD
                    and col_type not in _STATS_UNSUPPORTED_TYPES
                )

                if should_collect_stats:
                    try:
                        _t_stat = (
                            time.perf_counter() if _PROFILE_INDEX else 0.0
                        )
                        stats = connector.get_column_stats(tname, col_info.name, sname)
                        if _PROFILE_INDEX:
                            _elapsed = time.perf_counter() - _t_stat
                            _stats_total += _elapsed
                            _stats_calls += 1
                            if _elapsed > _stats_max:
                                _stats_max = _elapsed
                                _stats_max_target = f"{sname}.{tname}.{col_info.name}"
                        stats_kwargs = {
                            "distinct_count": stats.distinct_count,
                            "min_value": stats.min_value,
                            "max_value": stats.max_value,
                            "null_percentage": stats.null_percentage,
                            "sample_distinct_values": stats.sample_distinct_values,
                        }
                    except Exception:
                        logger.debug(
                            "Failed to collect stats for %s.%s.%s, skipping",
                            sname,
                            tname,
                            col_info.name,
                        )

                if cat_col:
                    for k in (
                        "most_common_values",
                        "histogram_bounds",
                        "stats_correlation",
                        "null_percentage",
                        "column_cardinality",
                        "index_type",
                        "index_is_unique",
                    ):
                        if k in cat_col and cat_col[k] is not None:
                            stats_kwargs[k] = cat_col[k]

                columns.append(
                    Column(
                        name=col_info.name,
                        data_type=col_info.data_type,
                        nullable=col_info.nullable,
                        is_primary_key=col_info.is_primary_key,
                        is_foreign_key=col_info.name in fk_columns,
                        default_value=col_info.default_value,
                        comment=col_info.comment,
                        is_indexed=col_info.is_indexed,
                        parent_column=col_info.parent_column,
                        is_array=col_info.is_array,
                        check_constraints=list(col_info.check_constraints or []),
                        ordinal_position=col_info.ordinal_position,
                        **stats_kwargs,
                    )
                )

            table_fks = [
                ForeignKey(
                    constraint_name=fk.constraint_name,
                    source_schema=fk.source_schema,
                    source_table=fk.source_table,
                    source_column=fk.source_column,
                    target_schema=fk.target_schema,
                    target_table=fk.target_table,
                    target_column=fk.target_column,
                )
                for fk in fk_index.get((sname, tname), [])
            ]

            tex = table_catalog_extra.get((sname, tname), {})
            table_kwargs: dict[str, Any] = {}
            for k in (
                "seq_scan_count",
                "idx_scan_count",
                "insert_count",
                "update_count",
                "delete_count",
                "is_partitioned",
                "partition_key",
                "grants",
                "access_read_count",
                "access_write_count",
                "days_since_last_access",
                "potentially_unused",
                "table_bytes",
                "clustering_key",
            ):
                if k in tex and tex[k] is not None:
                    table_kwargs[k] = tex[k]

            tables.append(
                Table(
                    name=tname,
                    schema_name=sname,
                    table_type=table_info.table_type,
                    columns=columns,
                    row_count=row_count,
                    row_count_source=table_info.row_count_source,
                    comment=table_info.comment,
                    foreign_keys=table_fks,
                    **table_kwargs,
                )
            )

        if _PROFILE_INDEX:
            _loop_elapsed = time.perf_counter() - _loop_start
            print(
                f"[profile] inspect.table_loop_total: {_loop_elapsed:.2f}s "
                f"(get_columns sum={_columns_total:.2f}s; "
                f"get_column_stats sum={_stats_total:.2f}s, "
                f"calls={_stats_calls}, "
                f"max={_stats_max * 1000:.0f}ms @ {_stats_max_target or '-'})",
                flush=True,
            )

        _t = time.perf_counter() if _PROFILE_INDEX else 0.0
        view_deps: list[ViewDependency] = []
        try:
            view_deps = connector.load_view_dependencies(config.schema_filter)
        except Exception:
            logger.warning(
                "View/trigger lineage extraction failed; continuing without it"
            )
        if _PROFILE_INDEX:
            print(
                f"[profile] inspect.load_view_dependencies: {(time.perf_counter() - _t) * 1000:.0f}ms",
                flush=True,
            )

        elapsed = time.monotonic() - start
        logger.info(
            "Introspection complete: %d tables, %d columns, %d foreign keys, %d lineage deps in %.1fs",
            len(tables),
            sum(len(t.columns) for t in tables),
            sum(len(t.foreign_keys) for t in tables),
            len(view_deps),
            elapsed,
        )

        return SchemaSnapshot(
            connection_name=config.name,
            database=config.database or "",
            schemas=sorted(schemas_seen),
            tables=tables,
            introspected_at=datetime.now(timezone.utc),
            metadata={"connector": config.type.value},
            view_dependencies=view_deps,
        )
