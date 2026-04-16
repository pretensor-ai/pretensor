"""MCP-facing read logic: stable import surface for tools and resources.

Implementation lives in ``pretensor.mcp.tools`` and sibling modules; this module
re-exports the public API used by ``server.py`` and tests.
"""

from __future__ import annotations

from pretensor.mcp.config_json import mcp_config_json
from pretensor.mcp.resources import (
    clusters_resource_markdown,
    cross_db_entities_resource_markdown,
    databases_resource_markdown,
    db_overview_resource_markdown,
    metrics_resource_markdown,
)
from pretensor.mcp.service_registry import resolve_registry_entry
from pretensor.mcp.tools.compile_metric import compile_metric_payload
from pretensor.mcp.tools.context import context_payload
from pretensor.mcp.tools.cypher import cypher_payload
from pretensor.mcp.tools.detect_changes import detect_changes_payload
from pretensor.mcp.tools.impact import impact_payload
from pretensor.mcp.tools.list import list_databases_payload
from pretensor.mcp.tools.schema import schema_payload
from pretensor.mcp.tools.search import query_payload
from pretensor.mcp.tools.traverse import traverse_payload
from pretensor.mcp.tools.validate_sql import validate_sql_payload

__all__ = [
    "list_databases_payload",
    "query_payload",
    "cypher_payload",
    "context_payload",
    "traverse_payload",
    "impact_payload",
    "detect_changes_payload",
    "compile_metric_payload",
    "schema_payload",
    "validate_sql_payload",
    "databases_resource_markdown",
    "db_overview_resource_markdown",
    "metrics_resource_markdown",
    "clusters_resource_markdown",
    "cross_db_entities_resource_markdown",
    "resolve_registry_entry",
    "mcp_config_json",
]
