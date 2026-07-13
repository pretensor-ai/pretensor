"""LangChain adapter — wraps Pretensor graph tools as ``StructuredTool`` objects.

Install the optional extra before use::

    pip install 'pretensor[langchain]'

Usage::

    from pathlib import Path
    from pretensor.integrations.langchain import load_langchain_tools
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor

    tools = load_langchain_tools(Path(".pretensor"))
    llm = ChatOpenAI(model="gpt-4o")
    agent = AgentExecutor(
        agent=create_tool_calling_agent(llm, tools, prompt), tools=tools
    )
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pretensor.integrations._base import make_tool_functions

__all__ = ["load_langchain_tools"]


def load_langchain_tools(graph_dir: Path | str) -> list[Any]:
    """Return Pretensor graph tools as LangChain ``StructuredTool`` objects.

    Args:
        graph_dir: Root directory of the Pretensor graph store.

    Returns:
        List of six ``langchain_core.tools.StructuredTool`` instances:
        schema, context, traverse, impact, query, validate_sql.

    Raises:
        ImportError: If ``langchain-core`` is not installed.
    """
    try:
        lc_tools = importlib.import_module("langchain_core.tools")
    except ImportError as exc:
        raise ImportError(
            "LangChain is not installed. Run: pip install 'pretensor[langchain]'"
        ) from exc

    structured_tool = lc_tools.StructuredTool
    return [
        structured_tool.from_function(fn) for fn in make_tool_functions(Path(graph_dir))
    ]
