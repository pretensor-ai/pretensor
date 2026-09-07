"""The MCP initialize handshake must advertise the pretensor package version."""

from __future__ import annotations

from collections.abc import Generator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

import pretensor.version as version_mod
from pretensor.mcp.server import create_server
from pretensor.mcp.service_context import reset_server_context
from pretensor.version import package_version


@pytest.fixture(autouse=True)
def _clear_server_context() -> Generator[None, None, None]:
    """create_server installs a global server context; close and clear it.

    Leaking it would hand later tests a stale StoreCache whose open Kuzu
    stores accumulate across the suite until mmap reservations fail.
    """
    reset_server_context()
    yield
    reset_server_context()


def test_server_info_reports_pretensor_version(tmp_path: Path) -> None:
    """serverInfo.version must be the pretensor package version, not the mcp SDK's.

    The mcp SDK falls back to its own package version when ``Server`` is built
    without an explicit ``version``, which breaks naive version checks in MCP
    clients.
    """
    server = create_server(tmp_path)
    init = server.create_initialization_options()
    assert init.server_version == version("pretensor")
    assert init.server_version != version("mcp")


def test_package_version_fallback_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare source checkout (no installed dist) yields the fallback string."""

    def _raise(_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(version_mod, "version", _raise)
    assert package_version() == "unknown"
    assert package_version(fallback="0.0.0+unknown") == "0.0.0+unknown"
