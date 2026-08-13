"""Stable identifiers for graph nodes and explicit FK edges."""

from __future__ import annotations

import hashlib

from pretensor.connectors.models import ForeignKey, ViewDependency

__all__ = [
    "table_node_id",
    "column_node_id",
    "fk_edge_id",
    "lineage_edge_id",
    "dbt_lineage_edge_id",
    "entity_node_id",
    "metric_template_node_id",
    "external_consumer_node_id",
    "consumes_edge_id",
]


def table_node_id(connection_name: str, schema_name: str, table_name: str) -> str:
    """Stable node id for a physical table within one indexed connection."""
    return f"{connection_name}::{schema_name}::{table_name}"


def column_node_id(
    connection_name: str, schema_name: str, table_name: str, column_name: str
) -> str:
    """Stable node id for a physical column."""
    return f"{connection_name}::{schema_name}::{table_name}::{column_name}"


def entity_node_id(connection_name: str, entity_name: str) -> str:
    """Stable node id for a business entity within one indexed connection."""
    return f"{connection_name}::entity::{entity_name}"


def metric_template_node_id(
    connection_name: str, database: str, template_name: str
) -> str:
    """Stable node id for a metric template scoped to one logical database."""
    return f"{connection_name}::{database}::metric::{template_name}"


def fk_edge_id(fk: ForeignKey) -> str:
    """Stable edge id from explicit FK column pair (per source table)."""
    return (
        f"{fk.source_schema}.{fk.source_table}.{fk.source_column}"
        f"->{fk.target_schema}.{fk.target_table}.{fk.target_column}"
    )


def lineage_edge_id(connection_name: str, dep: ViewDependency) -> str:
    """Stable id for a LINEAGE edge (dedupes re-index and multiple evidence paths)."""
    raw = (
        f"{connection_name}|{dep.source_schema}.{dep.source_table}"
        f"->{dep.target_schema}.{dep.target_table}|{dep.lineage_type}"
    )
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"lineage::{digest}"


def dbt_lineage_edge_id(
    connection_name: str, parent_dbt_id: str, child_dbt_id: str
) -> str:
    """Stable id for a dbt ``parent_map`` LINEAGE edge (idempotent re-runs)."""
    raw = f"{connection_name}|dbt|{parent_dbt_id}->{child_dbt_id}|model_dependency"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"lineage::dbt::{digest}"


def external_consumer_node_id(
    connection_name: str,
    service_name: str,
    file_path: str,
    line_start: int,
    sql_fingerprint: str,
) -> str:
    """Stable node id for one code location that issues a specific SQL statement.

    Keyed by ``(connection, service, repo-relative file, line, normalized SQL
    fingerprint)`` so the same statement in the same place dedupes across
    re-scans, while the same statement at two call sites — or resolved against
    two different connections — gets its own node. Line drift between scans
    produces a fresh id; the stale-consumer sweep reclaims the old one.
    """
    raw = f"{connection_name}|{service_name}|{file_path}|{line_start}|{sql_fingerprint}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"consumer::{digest}"


def consumes_edge_id(consumer_node_id: str, table_node_id: str, op: str) -> str:
    """Stable id for a CONSUMES edge (idempotent by consumer, table, and op).

    ``op`` is part of the key so a statement that both reads and writes the same
    table yields two distinct edges rather than clobbering one.
    """
    raw = f"{consumer_node_id}->{table_node_id}|{op}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
    return f"consumes::{digest}"
