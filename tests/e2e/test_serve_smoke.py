"""E2E tests — pretensor serve --config-only subprocess exits 0."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

if not os.getenv("PRETENSOR_E2E"):
    pytest.skip("set PRETENSOR_E2E=1", allow_module_level=True)

pytestmark = pytest.mark.e2e


def test_serve_config_only_exits_zero(indexed_state: Path) -> None:
    result = subprocess.run(
        ["uv", "run", "pretensor", "serve", "--config-only", "--graph-dir", str(indexed_state)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"Expected exit 0; got {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "mcpServers" in result.stdout, (
        f"Expected 'mcpServers' in stdout; got: {result.stdout!r}"
    )


def test_serve_config_only_json_is_valid(indexed_state: Path) -> None:
    result = subprocess.run(
        ["uv", "run", "pretensor", "serve", "--config-only", "--graph-dir", str(indexed_state)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"serve --config-only failed (exit {result.returncode}):\n{result.stderr}"
    )
    parsed = json.loads(result.stdout)
    assert "mcpServers" in parsed, (
        f"Expected 'mcpServers' key in parsed JSON; got keys: {list(parsed)}"
    )
