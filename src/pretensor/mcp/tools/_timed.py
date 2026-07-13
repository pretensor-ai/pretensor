"""Shared timing context manager for MCP tool handlers."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from pretensor.observability import log_timed_operation


def timed_tool(
    tool_name: str, graph_dir: Path, **fields: Any
) -> AbstractContextManager[None]:
    # Logger name is intentionally pinned to "pretensor.mcp.server" (not
    # __name__) so these handler-timing records keep emitting under the same
    # logger they used when the handlers lived in server.py — preserving any
    # downstream log routing/filtering. Do not switch to getLogger(__name__).
    return log_timed_operation(
        logging.getLogger("pretensor.mcp.server"),
        event="mcp.tool_handler",
        tool=tool_name,
        graph_dir=str(graph_dir),
        **fields,
    )
