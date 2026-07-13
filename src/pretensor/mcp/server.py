"""MCP server (stdio) exposing Pretensor graph tools and resources."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl

from pretensor.config import PretensorConfig
from pretensor.mcp.service import (
    clusters_resource_markdown,
    databases_resource_markdown,
    db_overview_resource_markdown,
    mcp_config_json,
    metrics_resource_markdown,
)
from pretensor.mcp.service_context import build_server_context, set_server_context
from pretensor.mcp.tool_registry import McpTool, McpToolRegistry
from pretensor.mcp.tools import (
    compile_metric as compile_metric_tool,
)
from pretensor.mcp.tools import (
    context as context_tool,
)
from pretensor.mcp.tools import (
    cypher as cypher_tool,
)
from pretensor.mcp.tools import (
    detect_changes as detect_changes_tool,
)
from pretensor.mcp.tools import (
    impact as impact_tool,
)
from pretensor.mcp.tools import (
    list as list_tool,
)
from pretensor.mcp.tools import (
    schema as schema_tool,
)
from pretensor.mcp.tools import (
    search as search_tool,
)
from pretensor.mcp.tools import (
    semantic_search as semantic_search_tool,
)
from pretensor.mcp.tools import (
    traverse as traverse_tool,
)
from pretensor.mcp.tools import (
    validate_sql as validate_sql_tool,
)

__all__ = ["create_server", "run_server", "print_mcp_config"]

logger = logging.getLogger(__name__)

_DB_OVERVIEW_PATTERN = re.compile(
    r"^pretensor://db/(?P<name>[^/]+)/overview$",
)
_DB_CLUSTERS_PATTERN = re.compile(
    r"^pretensor://db/(?P<name>[^/]+)/clusters$",
)
_DB_METRICS_PATTERN = re.compile(
    r"^pretensor://db/(?P<name>[^/]+)/metrics$",
)


def print_mcp_config(graph_dir: Path, *, stream: Any = None) -> None:
    """Print the ``mcpServers`` JSON block (for Claude / Cursor) to a text stream.

    Defaults to **stderr** so stdout stays reserved for MCP JSON-RPC over stdio
    after ``serve`` starts the server.
    """
    out = stream if stream is not None else __import__("sys").stderr
    out.write(mcp_config_json(graph_dir) + "\n")
    out.flush()


def _build_oss_registry(graph_dir: Path) -> McpToolRegistry:
    """Build a :class:`McpToolRegistry` pre-loaded with all OSS tools.

    Internal helper — use :func:`create_server` as the extension point (it
    accepts ``extra_tools`` for downstream registration).
    """
    registry = McpToolRegistry()
    for create in (
        list_tool.create_tool,
        schema_tool.create_tool,
        cypher_tool.create_tool,
        search_tool.create_tool,
        semantic_search_tool.create_tool,
        context_tool.create_tool,
        traverse_tool.create_tool,
        impact_tool.create_tool,
        detect_changes_tool.create_tool,
        compile_metric_tool.create_tool,
        validate_sql_tool.create_tool,
    ):
        registry.register(create(graph_dir))
    return registry


def render_resource_markdown(graph_dir: Path, key: str) -> str:
    """Resolve a resource URI to markdown, with a sanitized error fallback.

    On any handler error the full stack is logged server-side under a
    correlation id and the client receives a generic message — never the
    exception text or a filesystem path.
    """
    try:
        if key == "pretensor://databases":
            return databases_resource_markdown(graph_dir)
        match = _DB_OVERVIEW_PATTERN.match(key)
        if match:
            return db_overview_resource_markdown(graph_dir, match.group("name"))
        match_c = _DB_CLUSTERS_PATTERN.match(key)
        if match_c:
            return clusters_resource_markdown(graph_dir, match_c.group("name"))
        match_m = _DB_METRICS_PATTERN.match(key)
        if match_m:
            return metrics_resource_markdown(graph_dir, match_m.group("name"))
        return f"# Resource not found\n\nUnknown URI: `{key}`"
    except Exception:
        correlation_id = uuid.uuid4().hex
        logger.exception(
            "Error reading MCP resource %r [correlation_id=%s]",
            key,
            correlation_id,
        )
        return f"# Error\n\nFailed to load resource. Correlation ID: `{correlation_id}`"


def create_server(
    graph_dir: Path,
    *,
    visibility_path: Path | None = None,
    profile: str | None = None,
    config: PretensorConfig | None = None,
    extra_tools: list[McpTool] | None = None,
) -> Server[object, object]:
    """Build the low-level MCP :class:`Server` bound to ``graph_dir``.

    Args:
        graph_dir: Root directory of the Pretensor graph store.
        visibility_path: Optional path to a visibility filter config file.
        profile: Optional named profile for the server context.
        config: Optional central Pretensor configuration for MCP runtime wiring.
        extra_tools: Additional :class:`McpTool` instances to register on top
            of the 7 built-in OSS tools (e.g. from plugins).
    """
    set_server_context(
        build_server_context(
            graph_dir,
            visibility_path=visibility_path,
            profile=profile,
            config=config,
        )
    )

    registry = _build_oss_registry(graph_dir)
    for tool in extra_tools or []:
        registry.register(tool)

    server = Server[object, object](
        "pretensor",
        instructions=(
            "Pretensor graph MCP: discover indexed databases, search metadata, "
            "and fetch full context for physical tables (Kuzu schema graph)."
        ),
    )

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return registry.list_tools()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        try:
            return await registry.call_tool(name, arguments)
        except Exception:
            correlation_id = uuid.uuid4().hex
            logger.exception(
                "Unexpected error dispatching MCP tool %r [correlation_id=%s]",
                name,
                correlation_id,
            )
            return {
                "error": "Internal server error",
                "tool": name,
                "correlation_id": correlation_id,
            }

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=AnyUrl("pretensor://databases"),
                name="databases",
                title="All indexed databases",
                description="Registry overview (markdown)",
                mimeType="text/markdown",
            ),
        ]

    @server.list_resource_templates()
    async def _list_resource_templates() -> list[types.ResourceTemplate]:
        return [
            types.ResourceTemplate(
                uriTemplate="pretensor://db/{name}/overview",
                name="db-overview",
                title="Per-database overview",
                description="Table count, entity count, staleness (markdown)",
                mimeType="text/markdown",
            ),
            types.ResourceTemplate(
                uriTemplate="pretensor://db/{name}/clusters",
                name="db-clusters",
                title="Per-database domain clusters",
                description="Leiden clusters and table groupings (markdown)",
                mimeType="text/markdown",
            ),
            types.ResourceTemplate(
                uriTemplate="pretensor://db/{name}/metrics",
                name="db-metrics",
                title="Per-database metric templates",
                description="MetricTemplate nodes: SQL, validation, dependencies (markdown)",
                mimeType="text/markdown",
            ),
        ]

    @server.read_resource()
    async def _read_resource(uri: AnyUrl) -> str:
        return render_resource_markdown(graph_dir, str(uri))

    return server


async def _run_async(
    graph_dir: Path,
    *,
    visibility_path: Path | None = None,
    profile: str | None = None,
    config: PretensorConfig | None = None,
) -> None:
    server = create_server(
        graph_dir,
        visibility_path=visibility_path,
        profile=profile,
        config=config,
    )
    init = server.create_initialization_options(
        notification_options=NotificationOptions(resources_changed=False)
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init)


def run_server(
    graph_dir: Path,
    *,
    visibility_path: Path | None = None,
    profile: str | None = None,
    config: PretensorConfig | None = None,
) -> None:
    """Start the MCP server on stdio (blocking)."""
    asyncio.run(
        _run_async(
            graph_dir,
            visibility_path=visibility_path,
            profile=profile,
            config=config,
        )
    )
