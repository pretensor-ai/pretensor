"""MCP client config snippet (Cursor / Claude Desktop)."""

from __future__ import annotations

import json
from pathlib import Path


def mcp_config_json(graph_dir: Path) -> str:
    """Claude Desktop / Cursor style ``mcpServers`` block for stdio transport."""
    cmd = "pretensor"
    graph_dir_resolved = str(graph_dir.resolve())
    block = {
        "mcpServers": {
            "pretensor": {
                "command": cmd,
                "args": ["serve", "--graph-dir", graph_dir_resolved],
            }
        }
    }
    return json.dumps(block, indent=2)


__all__ = ["mcp_config_json"]
