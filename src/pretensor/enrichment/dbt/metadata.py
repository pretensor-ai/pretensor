"""Enrich ``SchemaTable`` / ``SchemaColumn`` nodes from dbt ``manifest.json`` metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pretensor.core.ids import column_node_id
from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.manifest import DbtManifest, DbtModel, DbtSource
from pretensor.enrichment.dbt.resolution import (
    physical_table_name_for_model,
    physical_table_name_for_source,
)

__all__ = ["DbtMetadataWriteStats", "write_dbt_metadata"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DbtMetadataWriteStats:
    """Counts from ``write_dbt_metadata`` for CLI summaries.

    ``tags_set`` counts tags *actually added* to the graph (delta against any
    pre-existing tags on the node), not tags offered from the manifest.
    """

    tables_enriched: int
    tags_set: int


def _empty_stats() -> DbtMetadataWriteStats:
    return DbtMetadataWriteStats(tables_enriched=0, tags_set=0)


def _non_empty_str(value: str | None) -> str | None:
    if value is None:
        return None
    s = value.strip()
    return s if s else None


def _existing_table_tags(
    store: KuzuStore,
    connection_name: str,
    schema_name: str,
    table_name: str,
) -> set[str]:
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn, schema_name: $s, table_name: $n})
        RETURN COALESCE(t.tags, CAST([] AS STRING[])) AS tags
        """,
        {"cn": connection_name, "s": schema_name, "n": table_name},
    )
    if not rows:
        return set()
    raw = rows[0][0]
    if not isinstance(raw, (list, tuple)):
        return set()
    return {str(x) for x in raw if str(x).strip()}


def _apply_table_metadata(
    store: KuzuStore,
    connection_name: str,
    schema_name: str,
    table_name: str,
    *,
    description: str | None,
    tags: tuple[str, ...],
    dbt_ref: str,
) -> DbtMetadataWriteStats:
    desc = _non_empty_str(description)
    tag_list = [t.strip() for t in tags if t and str(t).strip()]
    if desc is None and not tag_list:
        return _empty_stats()

    pre_existing = (
        _existing_table_tags(store, connection_name, schema_name, table_name)
        if tag_list
        else set()
    )

    params: dict[str, object] = {
        "cn": connection_name,
        "s": schema_name,
        "n": table_name,
    }
    set_parts: list[str] = []
    if desc is not None:
        set_parts.append(
            "t.description = CASE WHEN t.description IS NULL OR t.description = '' "
            "THEN $tbl_desc ELSE t.description END"
        )
        params["tbl_desc"] = desc
    if tag_list:
        set_parts.append(
            "t.tags = list_distinct(list_concat(COALESCE(t.tags, CAST([] AS STRING[])), $new_tags))"
        )
        params["new_tags"] = tag_list
    if not set_parts:
        return _empty_stats()
    result = store.query_all_rows(
        f"""
        MATCH (t:SchemaTable {{connection_name: $cn, schema_name: $s, table_name: $n}})
        SET {", ".join(set_parts)}
        RETURN t.node_id
        """,
        params,
    )
    if not result:
        logger.debug(
            "dbt metadata: SchemaTable missing for %s (schema=%r table=%r)",
            dbt_ref,
            schema_name,
            table_name,
        )
        return _empty_stats()

    new_tags_added = len({t for t in tag_list if t not in pre_existing})
    return DbtMetadataWriteStats(
        tables_enriched=1,
        tags_set=new_tags_added,
    )


def _apply_column_description(
    store: KuzuStore,
    connection_name: str,
    schema_name: str,
    table_name: str,
    column_name: str,
    description: str,
    *,
    dbt_ref: str,
) -> None:
    desc = _non_empty_str(description)
    if desc is None:
        return
    cid = column_node_id(connection_name, schema_name, table_name, column_name)
    result = store.query_all_rows(
        """
        MATCH (c:SchemaColumn {node_id: $cid})
        SET c.description = CASE WHEN c.description IS NULL OR c.description = ''
            THEN $col_desc ELSE c.description END
        RETURN c.node_id
        """,
        {"cid": cid, "col_desc": desc},
    )
    if not result:
        logger.debug(
            "dbt metadata: SchemaColumn missing for %s column %r (node_id=%s)",
            dbt_ref,
            column_name,
            cid,
        )


def _enrich_model(
    store: KuzuStore, connection_name: str, model: DbtModel
) -> DbtMetadataWriteStats:
    schema = model.schema_name
    phys = physical_table_name_for_model(model)
    if not schema or not phys:
        logger.debug(
            "dbt metadata: incomplete model metadata for %s (schema=%r table=%r)",
            model.unique_id,
            schema,
            phys,
        )
        return _empty_stats()
    stats = _apply_table_metadata(
        store,
        connection_name,
        schema,
        phys,
        description=model.description,
        tags=model.tags,
        dbt_ref=model.unique_id,
    )
    for col_name, col_desc in model.column_descriptions.items():
        _apply_column_description(
            store,
            connection_name,
            schema,
            phys,
            col_name,
            col_desc,
            dbt_ref=model.unique_id,
        )
    return stats


def _enrich_source(
    store: KuzuStore, connection_name: str, source: DbtSource
) -> DbtMetadataWriteStats:
    schema = source.schema_name
    phys = physical_table_name_for_source(source)
    if not schema or not phys:
        logger.debug(
            "dbt metadata: incomplete source metadata for %s (schema=%r table=%r)",
            source.unique_id,
            schema,
            phys,
        )
        return _empty_stats()
    stats = _apply_table_metadata(
        store,
        connection_name,
        schema,
        phys,
        description=source.description,
        tags=source.tags,
        dbt_ref=source.unique_id,
    )
    for col_name, col_desc in source.column_descriptions.items():
        _apply_column_description(
            store,
            connection_name,
            schema,
            phys,
            col_name,
            col_desc,
            dbt_ref=source.unique_id,
        )
    return stats


def write_dbt_metadata(
    manifest: DbtManifest, store: KuzuStore, connection_name: str
) -> DbtMetadataWriteStats:
    """Merge dbt model/source descriptions, column descriptions, and tags into the graph.

    Table-level: sets ``SchemaTable.description`` only when the current value is null or
    empty; merges dbt ``tags`` into ``SchemaTable.tags`` (union, order not preserved).

    Column-level: sets ``SchemaColumn.description`` when null or empty, matching by
    ``connection_name``, schema, physical table name, and column name.

    Skips nodes that are not present in the store (debug log only). Does not resolve
    ``docs()`` blocks — only inline string descriptions in the manifest.

    Args:
        manifest: Parsed dbt manifest.
        store: Open Kuzu store (``ensure_schema`` recommended).
        connection_name: Pretensor connection name used when tables were indexed.

    Returns:
        Counts of tables updated and number of distinct new tags actually added.
    """
    total_tables = 0
    total_tags = 0
    for model in manifest.nodes.values():
        st = _enrich_model(store, connection_name, model)
        total_tables += st.tables_enriched
        total_tags += st.tags_set
    for source in manifest.sources.values():
        st = _enrich_source(store, connection_name, source)
        total_tables += st.tables_enriched
        total_tags += st.tags_set
    return DbtMetadataWriteStats(tables_enriched=total_tables, tags_set=total_tags)
