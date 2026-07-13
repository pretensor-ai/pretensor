"""Tests for pretensor.integrations.llamaindex adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pretensor.integrations.llamaindex import load_llamaindex_tools

# ---------------------------------------------------------------------------
# ImportError path
# ---------------------------------------------------------------------------


def test_load_llamaindex_tools_raises_import_error_when_not_installed(
    tmp_path: Path,
) -> None:
    saved = sys.modules.pop("llama_index.core.tools", ...)
    saved_core = sys.modules.pop("llama_index.core", ...)
    sys.modules["llama_index.core"] = None  # type: ignore[assignment]
    sys.modules["llama_index.core.tools"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError, match="pretensor\\[llama-index\\]"):
            load_llamaindex_tools(tmp_path)
    finally:
        if saved is ...:
            sys.modules.pop("llama_index.core.tools", None)
        else:
            sys.modules["llama_index.core.tools"] = saved  # type: ignore[assignment]
        if saved_core is ...:
            sys.modules.pop("llama_index.core", None)
        else:
            sys.modules["llama_index.core"] = saved_core  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Happy path — mock llama_index so no install needed
# ---------------------------------------------------------------------------


class _FakeFunctionTool:
    def __init__(self, name: str, fn: object) -> None:
        self.name = name
        self._fn = fn

    @classmethod
    def from_defaults(cls, fn: object, **kwargs: object) -> "_FakeFunctionTool":
        return cls(getattr(fn, "__name__", "unknown"), fn)


def test_load_llamaindex_tools_returns_six_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_module = MagicMock()
    fake_module.FunctionTool = _FakeFunctionTool
    monkeypatch.setitem(sys.modules, "llama_index.core.tools", fake_module)

    tools = load_llamaindex_tools(tmp_path)
    assert len(tools) == 6


def test_load_llamaindex_tools_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_module = MagicMock()
    fake_module.FunctionTool = _FakeFunctionTool
    monkeypatch.setitem(sys.modules, "llama_index.core.tools", fake_module)

    tools = load_llamaindex_tools(tmp_path)
    names = [t.name for t in tools]
    assert names == ["schema", "context", "traverse", "impact", "query", "validate_sql"]


def test_load_llamaindex_tools_accepts_str_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_module = MagicMock()
    fake_module.FunctionTool = _FakeFunctionTool
    monkeypatch.setitem(sys.modules, "llama_index.core.tools", fake_module)

    tools = load_llamaindex_tools(str(tmp_path))
    assert len(tools) == 6


# ---------------------------------------------------------------------------
# Optional: skip-based tests when llama_index is actually installed
# ---------------------------------------------------------------------------


def test_load_llamaindex_tools_real_function_tool(tmp_path: Path) -> None:
    li_tools = pytest.importorskip(
        "llama_index.core.tools", reason="llama-index-core not installed"
    )
    function_tool = li_tools.FunctionTool

    tools = load_llamaindex_tools(tmp_path)
    assert len(tools) == 6
    for t in tools:
        assert isinstance(t, function_tool)

    names = [t.metadata.name for t in tools]
    assert names == ["schema", "context", "traverse", "impact", "query", "validate_sql"]
