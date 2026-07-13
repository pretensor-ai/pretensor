"""Tests for pretensor.integrations.langchain adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pretensor.integrations.langchain import load_langchain_tools

# ---------------------------------------------------------------------------
# ImportError path — test without framework installed
# ---------------------------------------------------------------------------


def test_load_langchain_tools_raises_import_error_when_not_installed(
    tmp_path: Path,
) -> None:
    # Setting sys.modules[name] = None is Python's negative-cache sentinel:
    # any subsequent "from name import ..." raises ImportError.
    saved = sys.modules.pop("langchain_core.tools", ...)
    saved_parent = sys.modules.pop("langchain_core", ...)
    sys.modules["langchain_core"] = None  # type: ignore[assignment]
    sys.modules["langchain_core.tools"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError, match="pretensor\\[langchain\\]"):
            load_langchain_tools(tmp_path)
    finally:
        if saved is ...:
            sys.modules.pop("langchain_core.tools", None)
        else:
            sys.modules["langchain_core.tools"] = saved  # type: ignore[assignment]
        if saved_parent is ...:
            sys.modules.pop("langchain_core", None)
        else:
            sys.modules["langchain_core"] = saved_parent  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Happy path — mock langchain_core so no install needed
# ---------------------------------------------------------------------------


class _FakeStructuredTool:
    def __init__(self, name: str, fn: object) -> None:
        self.name = name
        self._fn = fn

    @classmethod
    def from_function(cls, fn: object) -> "_FakeStructuredTool":
        return cls(getattr(fn, "__name__", "unknown"), fn)


def test_load_langchain_tools_returns_six_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_module = MagicMock()
    fake_module.StructuredTool = _FakeStructuredTool
    monkeypatch.setitem(sys.modules, "langchain_core.tools", fake_module)

    tools = load_langchain_tools(tmp_path)
    assert len(tools) == 6


def test_load_langchain_tools_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_module = MagicMock()
    fake_module.StructuredTool = _FakeStructuredTool
    monkeypatch.setitem(sys.modules, "langchain_core.tools", fake_module)

    tools = load_langchain_tools(tmp_path)
    names = [t.name for t in tools]
    assert names == ["schema", "context", "traverse", "impact", "query", "validate_sql"]


def test_load_langchain_tools_accepts_str_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_module = MagicMock()
    fake_module.StructuredTool = _FakeStructuredTool
    monkeypatch.setitem(sys.modules, "langchain_core.tools", fake_module)

    tools = load_langchain_tools(str(tmp_path))
    assert len(tools) == 6


# ---------------------------------------------------------------------------
# Optional: skip-based tests when langchain_core is actually installed
# ---------------------------------------------------------------------------


def test_load_langchain_tools_real_structured_tool(tmp_path: Path) -> None:
    lc_tools = pytest.importorskip(
        "langchain_core.tools", reason="langchain-core not installed"
    )
    structured_tool = lc_tools.StructuredTool

    tools = load_langchain_tools(tmp_path)
    assert len(tools) == 6
    for t in tools:
        assert isinstance(t, structured_tool)

    names = [t.name for t in tools]
    assert names == ["schema", "context", "traverse", "impact", "query", "validate_sql"]
