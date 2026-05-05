"""L3 benchmark runners — agent task success.

L3 measures end-to-end NL-to-SQL against a real database with two control
conditions:

* ``baseline`` — the agent receives the raw schema DDL only.
* ``pretensor`` — the agent receives a live ``pretensor serve`` MCP
  session against an indexed graph of the same database.

This package is LLM-gated and lives outside the OSS deterministic core
(``docs/contracts/architecture.md`` Invariant #6). All LLM calls go through
``httpx`` against provider HTTP endpoints — no SDK dependency.
"""

from __future__ import annotations

from pretensor.benchmark.l3.agent import (
    DEFAULT_MAX_ITERATIONS,
    AgentLlmClient,
    AgentLoopError,
    AgentLoopResult,
    AgentMessage,
    AgentStep,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    ToolCallTraceEntry,
    ToolInvocationOutcome,
    ToolInvoker,
    run_agent_loop,
)
from pretensor.benchmark.l3.llm_client import (
    AnthropicHttpClient,
    LlmCallError,
    LlmClient,
    LlmResponse,
    OpenAIHttpClient,
)
from pretensor.benchmark.l3.mcp_client import (
    McpClient,
    McpClientError,
    McpToolResult,
    StdioMcpClient,
)
from pretensor.benchmark.l3.pretensor_runner import run_l3_pretensor
from pretensor.benchmark.l3.runner import run_l3_baseline

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "AgentLlmClient",
    "AgentLoopError",
    "AgentLoopResult",
    "AgentMessage",
    "AgentStep",
    "AgentTool",
    "AgentToolCall",
    "AgentToolResult",
    "AnthropicHttpClient",
    "LlmCallError",
    "LlmClient",
    "LlmResponse",
    "McpClient",
    "McpClientError",
    "McpToolResult",
    "OpenAIHttpClient",
    "StdioMcpClient",
    "ToolCallTraceEntry",
    "ToolInvocationOutcome",
    "ToolInvoker",
    "run_agent_loop",
    "run_l3_baseline",
    "run_l3_pretensor",
]
