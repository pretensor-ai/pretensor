"""Name and node-id resolution helpers shared across dbt enrichment writers.

These map dbt ``unique_id`` values (``model.<pkg>.<name>`` / ``source.<pkg>.<src>.<name>``)
to physical ``SchemaTable`` node ids for an indexed Pretensor connection.
"""

from __future__ import annotations

import logging

from pretensor.core.ids import table_node_id
from pretensor.enrichment.dbt.manifest import DbtManifest, DbtModel, DbtSource

__all__ = [
    "physical_table_name_for_model",
    "physical_table_name_for_source",
    "resolve_dbt_parent_node_id",
    "resolve_dbt_child_node_id",
]

logger = logging.getLogger(__name__)


def physical_table_name_for_model(model: DbtModel) -> str | None:
    """Return the physical table name dbt writes the model to (``alias`` or ``name``)."""
    return model.alias or model.name


def physical_table_name_for_source(source: DbtSource) -> str | None:
    """Return the physical table name dbt reads the source from (``identifier`` or ``name``)."""
    return source.identifier or source.name


def resolve_dbt_parent_node_id(
    manifest: DbtManifest,
    connection_name: str,
    parent_id: str,
) -> str | None:
    """Resolve a dbt ``unique_id`` (model or source) to its ``SchemaTable`` node id."""
    if parent_id.startswith("source."):
        src = manifest.sources.get(parent_id)
        if src is None:
            logger.debug("dbt resolution: unknown source id %s", parent_id)
            return None
        schema = src.schema_name
        phys = physical_table_name_for_source(src)
        if not schema or not phys:
            logger.debug(
                "dbt resolution: incomplete source metadata for %s (schema=%r table=%r)",
                parent_id,
                schema,
                phys,
            )
            return None
        return table_node_id(connection_name, schema, phys)
    if parent_id.startswith("model."):
        mdl = manifest.nodes.get(parent_id)
        if mdl is None:
            logger.debug("dbt resolution: unknown model id %s", parent_id)
            return None
        schema = mdl.schema_name
        phys = physical_table_name_for_model(mdl)
        if not schema or not phys:
            logger.debug(
                "dbt resolution: incomplete model metadata for %s (schema=%r table=%r)",
                parent_id,
                schema,
                phys,
            )
            return None
        return table_node_id(connection_name, schema, phys)
    logger.debug("dbt resolution: skipping non-model non-source id %s", parent_id)
    return None


def resolve_dbt_child_node_id(
    manifest: DbtManifest,
    connection_name: str,
    child_id: str,
) -> str | None:
    """Resolve a dbt ``model.*`` ``unique_id`` to its ``SchemaTable`` node id.

    Only models can be lineage children (sources have no upstream in dbt).
    """
    if not child_id.startswith("model."):
        return None
    mdl = manifest.nodes.get(child_id)
    if mdl is None:
        logger.debug("dbt resolution: unknown child model id %s", child_id)
        return None
    schema = mdl.schema_name
    phys = physical_table_name_for_model(mdl)
    if not schema or not phys:
        logger.debug(
            "dbt resolution: incomplete child model metadata for %s (schema=%r table=%r)",
            child_id,
            schema,
            phys,
        )
        return None
    return table_node_id(connection_name, schema, phys)
