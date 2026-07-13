"""Tests for pretensor.integrations.google_adk adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pretensor.integrations.google_adk import load_adk_tools

# ---------------------------------------------------------------------------
# ImportError path
# ---------------------------------------------------------------------------


def test_load_adk_tools_raises_import_error_when_not_installed(
    tmp_path: Path,
) -> None:
    saved = sys.modules.pop("google.adk.tools", ...)
    saved_adk = sys.modules.pop("google.adk", ...)
    sys.modules["google.adk"] = None  # type: ignore[assignment]
    sys.modules["google.adk.tools"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError, match="pretensor\\[google-adk\\]"):
            load_adk_tools(tmp_path)
    finally:
        if saved is ...:
            sys.modules.pop("google.adk.tools", None)
        else:
            sys.modules["google.adk.tools"] = saved  # type: ignore[assignment]
        if saved_adk is ...:
            sys.modules.pop("google.adk", None)
        else:
            sys.modules["google.adk"] = saved_adk  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Happy path — mock google.adk so no install needed
# ---------------------------------------------------------------------------


class _FakeAdkFunctionTool:
    def __init__(self, fn: object) -> None:
        self.name = getattr(fn, "__name__", "unknown")
        self._fn = fn


def test_load_adk_tools_returns_six_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_module = MagicMock()
    fake_module.FunctionTool = _FakeAdkFunctionTool
    monkeypatch.setitem(sys.modules, "google.adk.tools", fake_module)

    tools = load_adk_tools(tmp_path)
    assert len(tools) == 6


def test_load_adk_tools_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = MagicMock()
    fake_module.FunctionTool = _FakeAdkFunctionTool
    monkeypatch.setitem(sys.modules, "google.adk.tools", fake_module)

    tools = load_adk_tools(tmp_path)
    names = [t.name for t in tools]
    assert names == ["schema", "context", "traverse", "impact", "query", "validate_sql"]


def test_load_adk_tools_accepts_str_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_module = MagicMock()
    fake_module.FunctionTool = _FakeAdkFunctionTool
    monkeypatch.setitem(sys.modules, "google.adk.tools", fake_module)

    tools = load_adk_tools(str(tmp_path))
    assert len(tools) == 6


# ---------------------------------------------------------------------------
# Optional: skip-based tests when google-adk is actually installed
# ---------------------------------------------------------------------------


def test_load_adk_tools_real_function_tool(tmp_path: Path) -> None:
    adk_tools = pytest.importorskip(
        "google.adk.tools", reason="google-adk not installed"
    )
    function_tool = adk_tools.FunctionTool

    tools = load_adk_tools(tmp_path)
    assert len(tools) == 6
    for t in tools:
        assert isinstance(t, function_tool)
