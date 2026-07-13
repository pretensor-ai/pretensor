"""LlamaIndex adapter — wraps Pretensor graph tools as ``FunctionTool`` objects.

Install the optional extra before use::

    pip install 'pretensor[llama-index]'

Usage::

    from pathlib import Path
    from pretensor.integrations.llamaindex import load_llamaindex_tools
    from llama_index.core.agent import ReActAgent
    from llama_index.llms.openai import OpenAI

    tools = load_llamaindex_tools(Path(".pretensor"))
    agent = ReActAgent.from_tools(tools, llm=OpenAI(model="gpt-4o"), verbose=True)
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pretensor.integrations._base import make_tool_functions

__all__ = ["load_llamaindex_tools"]


def load_llamaindex_tools(graph_dir: Path | str) -> list[Any]:
    """Return Pretensor graph tools as LlamaIndex ``FunctionTool`` objects.

    Args:
        graph_dir: Root directory of the Pretensor graph store.

    Returns:
        List of six ``llama_index.core.tools.FunctionTool`` instances:
        schema, context, traverse, impact, query, validate_sql.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.
    """
    try:
        li_tools = importlib.import_module("llama_index.core.tools")
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex is not installed. Run: pip install 'pretensor[llama-index]'"
        ) from exc

    function_tool = li_tools.FunctionTool
    return [
        function_tool.from_defaults(fn=fn)
        for fn in make_tool_functions(Path(graph_dir))
    ]
