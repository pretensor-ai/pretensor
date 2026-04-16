"""Context tool schema and handler cover only the canonical ``db`` parameter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from pretensor.mcp import server as server_module


def _get_context_tool(tmp_path: Path) -> Any:
    registry = server_module._build_oss_registry(tmp_path)
    return registry._tools["context"]


def _call(tool: Any, args: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(tool.handler(args))


def test_context_schema_has_only_db(tmp_path: Path) -> None:
    tool = _get_context_tool(tmp_path)
    props = tool.input_schema["properties"]
    assert "db" in props
    assert "database" not in props


def test_context_handler_passes_db_to_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def _fake(graph_dir: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(server_module, "context_payload", _fake)
    tool = _get_context_tool(tmp_path)
    out = _call(tool, {"table": "public.t", "db": "demo"})
    assert out == {"ok": True}
    assert captured["db"] == "demo"


def test_context_handler_rejects_unknown_keys(tmp_path: Path) -> None:
    """Schema declares additionalProperties=False, so `database` is silently
    ignored by Python but rejected by MCP JSON schema validation upstream.
    Handler itself now has no fallback — missing `db` means `db_s is None`."""
    tool = _get_context_tool(tmp_path)
    # Sanity: schema forbids extra keys.
    assert tool.input_schema["additionalProperties"] is False
