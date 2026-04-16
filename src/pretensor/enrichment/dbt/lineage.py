"""Write table-level ``LINEAGE`` edges from dbt ``manifest.json`` ``parent_map``."""

from __future__ import annotations

import logging

from pretensor.core.ids import dbt_lineage_edge_id
from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.manifest import DbtManifest
from pretensor.enrichment.dbt.resolution import (
    resolve_dbt_child_node_id,
    resolve_dbt_parent_node_id,
)
from pretensor.graph_models.edge import LineageEdge

__all__ = ["write_dbt_lineage"]

logger = logging.getLogger(__name__)

_DBT_LINEAGE_SOURCE = "dbt"
_DBT_LINEAGE_TYPE = "model_dependency"
_DBT_LINEAGE_CONFIDENCE = 1.0


def _schema_table_exists(store: KuzuStore, node_id: str) -> bool:
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {node_id: $nid})
        RETURN 1
        LIMIT 1
        """,
        {"nid": node_id},
    )
    return len(rows) > 0


def write_dbt_lineage(
    manifest: DbtManifest, store: KuzuStore, connection_name: str
) -> int:
    """Emit ``LINEAGE`` edges from dbt ``parent_map`` for one indexed connection.

    For each ``model.*`` entry, creates one edge per direct parent that is a
    ``source.*`` or ``model.*`` node, when both endpoints exist as
    ``SchemaTable`` rows (matched by ``connection_name``, schema, and physical
    table name). Upstream is the parent; downstream is the child model.

    Idempotent per connection: existing dbt ``model_dependency`` edges for this
    connection are removed first, then rewritten. Note that this scopes the
    clear to edges where **both** endpoints are on ``connection_name`` — in
    ``--unified`` mode, a dbt project that crosses two Pretensor connections
    would need per-connection re-runs to fully refresh. Cross-connection dbt
    lineage is not supported in Phase 1 (out-of-scope).

    Args:
        manifest: Parsed dbt manifest (includes ``parent_map``).
        store: Open Kuzu store with schema ensured.
        connection_name: Pretensor connection name for ``SchemaTable`` nodes.

    Returns:
        Number of ``LINEAGE`` edges written.
    """
    store.clear_dbt_model_dependency_lineage(connection_name)
    written = 0
    for child_id, parents in manifest.parent_map.items():
        child_nid = resolve_dbt_child_node_id(manifest, connection_name, child_id)
        if child_nid is None:
            continue
        if not _schema_table_exists(store, child_nid):
            logger.debug(
                "dbt lineage: child SchemaTable missing for %s -> %s",
                child_id,
                child_nid,
            )
            continue
        for parent_id in parents:
            parent_nid = resolve_dbt_parent_node_id(
                manifest, connection_name, parent_id
            )
            if parent_nid is None:
                continue
            if not _schema_table_exists(store, parent_nid):
                logger.debug(
                    "dbt lineage: parent SchemaTable missing for %s -> %s",
                    parent_id,
                    parent_nid,
                )
                continue
            eid = dbt_lineage_edge_id(connection_name, parent_id, child_id)
            store.upsert_lineage_edge(
                LineageEdge(
                    edge_id=eid,
                    source_node_id=parent_nid,
                    target_node_id=child_nid,
                    source=_DBT_LINEAGE_SOURCE,
                    lineage_type=_DBT_LINEAGE_TYPE,
                    confidence=_DBT_LINEAGE_CONFIDENCE,
                )
            )
            written += 1
    return written
