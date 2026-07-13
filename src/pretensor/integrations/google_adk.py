"""Google ADK adapter — wraps Pretensor graph tools as ADK ``FunctionTool`` objects.

Install the optional extra before use::

    pip install 'pretensor[google-adk]'

Usage::

    from pathlib import Path
    from pretensor.integrations.google_adk import load_adk_tools
    from google.adk.agents import LlmAgent

    tools = load_adk_tools(Path(".pretensor"))
    agent = LlmAgent(model="gemini-2.0-flash", tools=tools)
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pretensor.integrations._base import make_tool_functions

__all__ = ["load_adk_tools"]


def load_adk_tools(graph_dir: Path | str) -> list[Any]:
    """Return Pretensor graph tools as Google ADK ``FunctionTool`` objects.

    Args:
        graph_dir: Root directory of the Pretensor graph store.

    Returns:
        List of six ``google.adk.tools.FunctionTool`` instances:
        schema, context, traverse, impact, query, validate_sql.

    Raises:
        ImportError: If ``google-adk`` is not installed.
    """
    try:
        adk_tools = importlib.import_module("google.adk.tools")
    except ImportError as exc:
        raise ImportError(
            "Google ADK is not installed. Run: pip install 'pretensor[google-adk]'"
        ) from exc

    function_tool = adk_tools.FunctionTool
    return [function_tool(fn) for fn in make_tool_functions(Path(graph_dir))]
