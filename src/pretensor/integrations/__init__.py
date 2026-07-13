"""Agent-framework SDK adapters for Pretensor graph tools.

Exposes the six graph tools (schema, context, traverse, impact, query,
validate_sql) as native tool objects for LangChain, LlamaIndex, and Google ADK.
No MCP server process is required.

Usage::

    from pretensor.integrations import load_langchain_tools
    from pretensor.integrations import load_llamaindex_tools
    from pretensor.integrations import load_adk_tools
"""

# pyright: reportUnsupportedDunderAll=false
# Names in __all__ are provided via __getattr__; Pyright does not model that.

from __future__ import annotations

from typing import Any

__all__ = [
    "load_langchain_tools",
    "load_llamaindex_tools",
    "load_adk_tools",
]


def __getattr__(name: str) -> Any:
    if name == "load_langchain_tools":
        from pretensor.integrations.langchain import load_langchain_tools

        return load_langchain_tools
    if name == "load_llamaindex_tools":
        from pretensor.integrations.llamaindex import load_llamaindex_tools

        return load_llamaindex_tools
    if name == "load_adk_tools":
        from pretensor.integrations.google_adk import load_adk_tools

        return load_adk_tools
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
