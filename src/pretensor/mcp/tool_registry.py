"""MCP tool plugin registry — McpTool and McpToolRegistry.

Allows dynamic, pluggable registration of MCP tools so that Cloud extensions
can add tools (e.g. ``suggest_query``, ``semantic_search``) without modifying
OSS server code.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import mcp.types as types

__all__ = ["McpTool", "McpToolRegistry"]

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class McpTool:
    """Descriptor and handler for a single MCP tool.

    Args:
        name: Unique tool name exposed to the MCP client.
        description: Human-readable description shown to the client.
        input_schema: JSON Schema object describing the tool's input.
        handler: Async callable that receives ``arguments`` and returns a
            JSON-serialisable ``dict``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def to_mcp_type(self) -> types.Tool:
        """Return the :class:`mcp.types.Tool` representation."""
        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
        )


class McpToolRegistry:
    """Registry that owns the canonical set of MCP tools for a server.

    Usage::

        registry = McpToolRegistry()
        registry.register(McpTool(name="my_tool", ...))

        # In the MCP list_tools handler:
        tools = registry.list_tools()

        # In the MCP call_tool handler:
        result = await registry.call_tool("my_tool", {"arg": "value"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, McpTool] = {}

    def register(self, tool: McpTool) -> None:
        """Register a tool; raises :exc:`ValueError` if the name is already taken.

        Args:
            tool: The tool descriptor and handler to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.debug("Registered MCP tool: %s", tool.name)

    def list_tools(self) -> list[types.Tool]:
        """Return all registered tools as :class:`mcp.types.Tool` objects.

        Returns:
            List of MCP tool descriptors in registration order.
        """
        return [t.to_mcp_type() for t in self._tools.values()]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Dispatch a tool call to its registered handler.

        All handler exceptions are caught and returned as ``{"error": ...}``
        so the MCP server never crashes on a bad tool invocation.

        Args:
            name: Tool name as sent by the MCP client.
            arguments: Raw argument dict (may be ``None`` or empty).

        Returns:
            JSON-serialisable result dict from the handler, or
            ``{"error": "..."}`` when the handler raises.
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        args = arguments or {}
        try:
            return await tool.handler(args)
        except Exception as exc:
            logger.exception("Unhandled exception in MCP tool %r", name)
            return {
                "error": f"Internal tool error: {exc}",
                "tool": name,
                "traceback": traceback.format_exc(),
            }

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
