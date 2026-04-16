"""MCP ``list_databases`` tool payload."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pretensor.config import GraphConfig
from pretensor.visibility.filter import VisibilityFilter

from ..payload_types import (
    DIALECT_TO_DB_TYPE,
    DatabaseListItem,
    iso_format,
    stale_threshold_days,
    staleness_days,
    utc_now,
)
from ..service_context import (
    get_effective_graph_config,
    get_effective_visibility_filter,
)
from ..service_registry import (
    counts_for_graph,
    graph_path_for_entry,
    load_registry,
    open_store_for_entry,
)


def list_databases_payload(
    graph_dir: Path,
    *,
    config: GraphConfig | None = None,
    visibility_filter: VisibilityFilter | None = None,
) -> dict[str, Any]:
    """Build JSON-serializable payload for ``list_databases``."""
    cfg = get_effective_graph_config(config)
    vf = visibility_filter or get_effective_visibility_filter()
    reg = load_registry(graph_dir)
    items: list[DatabaseListItem] = []
    now = utc_now()
    threshold = stale_threshold_days(cfg)
    for entry in reg.list_entries():
        table_count = 0
        column_count = 0
        row_sum = 0
        schemas: list[str] = []
        has_dbt_manifest: str = "not_attempted"
        has_llm_enrichment: str = "not_attempted"
        has_external_consumers: str = "not_attempted"
        gp = graph_path_for_entry(entry)
        if gp.exists():
            store = open_store_for_entry(entry)
            try:
                counts = counts_for_graph(
                    store,
                    connection_name=entry.connection_name,
                    visibility_filter=vf,
                    entry=entry,
                )
                table_count = counts.table_count
                column_count = counts.column_count
                row_sum = counts.row_count
                schemas = counts.schemas
                has_dbt_manifest = counts.has_dbt_manifest
                has_llm_enrichment = counts.has_llm_enrichment
                has_external_consumers = counts.has_external_consumers
            finally:
                store.close()
        days = staleness_days(entry.last_indexed_at, now)
        stale = days > threshold
        capabilities = ["schema", "fk_edges", "inferred_joins", "clustering"]
        if has_dbt_manifest == "present":
            capabilities.append("dbt_lineage")
            capabilities.append("dbt_metrics")
        if has_llm_enrichment == "present":
            capabilities.append("llm_enrichment")
        if has_external_consumers == "present":
            capabilities.append("external_consumers")
        row: DatabaseListItem = {
            "name": entry.connection_name,
            "db_type": DIALECT_TO_DB_TYPE.get(entry.dialect, "postgresql"),
            "table_count": table_count,
            "column_count": column_count,
            "row_count": row_sum,
            "schemas": schemas,
            "has_dbt_manifest": has_dbt_manifest,
            "has_llm_enrichment": has_llm_enrichment,
            "has_external_consumers": has_external_consumers,
            "capabilities": capabilities,
            "last_indexed": iso_format(entry.last_indexed_at),
            "is_stale": stale,
            "staleness_days": days,
            "graph_path": str(graph_path_for_entry(entry)),
        }
        # Only emit ``database`` when it differs from ``name`` — otherwise the
        # duplicate field is noise (e.g. ``{"name": "pagila", "database": "pagila"}``).
        if entry.database and entry.database != entry.connection_name:
            row["database"] = entry.database
        if stale:
            row["stale_warning"] = (
                f"Graph index is {days} days old (threshold {threshold} days). "
                "Consider `pretensor reindex <dsn>` or `pretensor index`."
            )
        items.append(row)
    return {"databases": items}


__all__ = ["list_databases_payload"]
