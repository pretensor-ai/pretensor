"""Unit tests for McpToolRegistry and McpTool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pretensor.mcp.tool_registry import McpTool, McpToolRegistry


def _make_tool(name: str, return_value: dict | None = None) -> McpTool:
    value = return_value or {"ok": True}

    async def _handler(args: dict) -> dict:
        return value

    return McpTool(
        name=name,
        description=f"Test tool: {name}",
        input_schema={"type": "object", "properties": {}},
        handler=_handler,
    )


# ---------------------------------------------------------------------------
# McpTool
# ---------------------------------------------------------------------------


def test_mcp_tool_to_mcp_type_sets_name_and_description() -> None:
    tool = _make_tool("my_tool")
    mcp_t = tool.to_mcp_type()
    assert mcp_t.name == "my_tool"
    assert mcp_t.description == "Test tool: my_tool"


def test_mcp_tool_to_mcp_type_sets_input_schema() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    tool = McpTool(
        name="search",
        description="desc",
        input_schema=schema,
        handler=_make_tool("search").handler,
    )
    mcp_t = tool.to_mcp_type()
    assert mcp_t.inputSchema == schema


# ---------------------------------------------------------------------------
# McpToolRegistry — registration
# ---------------------------------------------------------------------------


def test_registry_register_and_len() -> None:
    reg = McpToolRegistry()
    assert len(reg) == 0
    reg.register(_make_tool("a"))
    assert len(reg) == 1
    reg.register(_make_tool("b"))
    assert len(reg) == 2


def test_registry_contains() -> None:
    reg = McpToolRegistry()
    reg.register(_make_tool("alpha"))
    assert "alpha" in reg
    assert "beta" not in reg


def test_registry_duplicate_raises() -> None:
    reg = McpToolRegistry()
    reg.register(_make_tool("dup"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_make_tool("dup"))


# ---------------------------------------------------------------------------
# McpToolRegistry — list_tools
# ---------------------------------------------------------------------------


def test_list_tools_returns_all_registered() -> None:
    reg = McpToolRegistry()
    reg.register(_make_tool("tool_a"))
    reg.register(_make_tool("tool_b"))
    tools = reg.list_tools()
    names = [t.name for t in tools]
    assert names == ["tool_a", "tool_b"]


def test_list_tools_empty_registry() -> None:
    reg = McpToolRegistry()
    assert reg.list_tools() == []


# ---------------------------------------------------------------------------
# McpToolRegistry — call_tool
# ---------------------------------------------------------------------------


def test_call_tool_dispatches_correctly() -> None:
    reg = McpToolRegistry()
    reg.register(_make_tool("ping", return_value={"pong": 1}))
    result = asyncio.run(reg.call_tool("ping", {}))
    assert result == {"pong": 1}


def test_call_tool_passes_arguments_to_handler() -> None:
    received: list[dict] = []

    async def _capture(args: dict) -> dict:
        received.append(args)
        return {}

    reg = McpToolRegistry()
    reg.register(
        McpTool(
            name="capture",
            description="",
            input_schema={},
            handler=_capture,
        )
    )
    asyncio.run(reg.call_tool("capture", {"key": "value"}))
    assert received == [{"key": "value"}]


def test_call_tool_none_arguments_treated_as_empty() -> None:
    received: list[dict] = []

    async def _capture(args: dict) -> dict:
        received.append(args)
        return {}

    reg = McpToolRegistry()
    reg.register(
        McpTool(name="cap", description="", input_schema={}, handler=_capture)
    )
    asyncio.run(reg.call_tool("cap", None))
    assert received == [{}]


def test_call_tool_unknown_returns_error() -> None:
    reg = McpToolRegistry()
    result = asyncio.run(reg.call_tool("nonexistent", {}))
    assert "error" in result
    assert "nonexistent" in result["error"]


# ---------------------------------------------------------------------------
# Integration — all 10 OSS tools registered in create_server
# ---------------------------------------------------------------------------


def test_oss_registry_has_expected_tools() -> None:
    """Verify _build_oss_registry registers exactly the 10 OSS tools."""
    from pathlib import Path

    from pretensor.mcp.server import _build_oss_registry

    registry = _build_oss_registry(Path("/tmp/fake"))
    assert len(registry) == 10
    expected_tools = {
        "list_databases",
        "schema",
        "cypher",
        "query",
        "context",
        "traverse",
        "impact",
        "detect_changes",
        "compile_metric",
        "validate_sql",
    }
    assert {t.name for t in registry.list_tools()} == expected_tools


def test_create_server_accepts_extra_tools(tmp_path: Path) -> None:
    """Extra tools passed to create_server are registered alongside OSS tools."""
    from pretensor.mcp.server import _build_oss_registry
    from pretensor.mcp.tool_registry import McpTool

    async def _noop(args: dict) -> dict:
        return {}

    extra = McpTool(
        name="suggest_query",
        description="Cloud semantic suggest",
        input_schema={"type": "object"},
        handler=_noop,
    )

    registry = _build_oss_registry(tmp_path)
    registry.register(extra)
    assert "suggest_query" in registry
    assert len(registry) == 11
