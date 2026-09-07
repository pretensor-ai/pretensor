"""Tests for MCP client detection and registration."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any
from unittest import mock

from pretensor.cli.init.mcp_clients import (
    McpClient,
    detect_clients,
    register_client,
)


def _cursor_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / ".cursor" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_detect_finds_cursor_config(tmp_path: Path) -> None:
    _cursor_config(tmp_path, {"mcpServers": {}})
    with mock.patch("pretensor.cli.init.mcp_clients.shutil.which", return_value=None):
        found = detect_clients(home=tmp_path, cwd=tmp_path)
    assert [c.key for c in found] == ["cursor"]


def test_detect_returns_empty_when_nothing_installed(tmp_path: Path) -> None:
    with mock.patch("pretensor.cli.init.mcp_clients.shutil.which", return_value=None):
        assert detect_clients(home=tmp_path, cwd=tmp_path) == ()


def test_detect_finds_windows_claude_desktop_config(
    tmp_path: Path, monkeypatch
) -> None:
    """Claude Desktop on Windows keeps its config under %APPDATA%, not $HOME."""
    appdata = tmp_path / "AppData" / "Roaming"
    config_path = appdata / "Claude" / "claude_desktop_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")

    monkeypatch.setenv("APPDATA", str(appdata))
    with mock.patch("pretensor.cli.init.mcp_clients.shutil.which", return_value=None):
        found = detect_clients(home=tmp_path / "unused-home", cwd=tmp_path)
    assert [c.key for c in found] == ["claude-desktop"]
    assert found[0].config_path == config_path


def test_detect_ignores_appdata_when_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APPDATA", raising=False)
    with mock.patch("pretensor.cli.init.mcp_clients.shutil.which", return_value=None):
        assert detect_clients(home=tmp_path, cwd=tmp_path) == ()


def test_detect_prefers_explicit_env_over_process_environment(
    tmp_path: Path, monkeypatch
) -> None:
    """The ``env`` keyword lets callers probe without touching os.environ."""
    real_appdata = tmp_path / "real-appdata"
    fake_appdata = tmp_path / "fake-appdata"
    config_path = fake_appdata / "Claude" / "claude_desktop_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")

    monkeypatch.setenv("APPDATA", str(real_appdata))
    with mock.patch("pretensor.cli.init.mcp_clients.shutil.which", return_value=None):
        found = detect_clients(
            home=tmp_path / "unused-home",
            cwd=tmp_path,
            env={"APPDATA": str(fake_appdata)},
        )
    assert [c.key for c in found] == ["claude-desktop"]
    assert found[0].config_path == config_path


def test_register_preserves_unrelated_servers(tmp_path: Path) -> None:
    path = _cursor_config(
        tmp_path, {"mcpServers": {"other": {"command": "keepme"}}, "theme": "dark"}
    )
    client = McpClient(key="cursor", label="Cursor", config_path=path)
    register_client(client, tmp_path / ".pretensor")
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["mcpServers"]["other"] == {"command": "keepme"}
    assert written["theme"] == "dark"
    assert written["mcpServers"]["pretensor"]["command"] == "pretensor"


def test_register_creates_backup(tmp_path: Path) -> None:
    path = _cursor_config(tmp_path, {"mcpServers": {}})
    original_content = path.read_bytes()

    client = McpClient(key="cursor", label="Cursor", config_path=path)
    register_client(client, tmp_path / ".pretensor")
    backups = list(path.parent.glob("mcp.json.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original_content


def test_register_preserves_existing_file_mode(tmp_path: Path) -> None:
    """Registering must not silently tighten an existing config's permissions.

    ``tempfile.mkstemp`` creates its temp file 0600; without preserving the
    original mode, replacing a world-readable config with that temp file
    would leave it 0600 behind the user's back.
    """
    path = _cursor_config(tmp_path, {"mcpServers": {}})
    path.chmod(0o644)

    client = McpClient(key="cursor", label="Cursor", config_path=path)
    register_client(client, tmp_path / ".pretensor")

    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_register_claude_code_shells_out(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs: Any) -> Any:
        calls.append(cmd)

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    client = McpClient(key="claude-code", label="Claude Code", config_path=None)
    register_client(client, tmp_path / ".pretensor", runner=fake_runner)
    assert calls[0][:3] == ["claude", "mcp", "add"]
    assert "pretensor" in calls[0]


def test_register_cli_failure_on_non_zero_returncode(tmp_path: Path) -> None:
    def fake_runner(cmd: list[str], **kwargs: Any) -> Any:
        class _Result:
            returncode = 1
            stdout = ""
            stderr = "claude binary not found"

        return _Result()

    client = McpClient(key="claude-code", label="Claude Code", config_path=None)
    result = register_client(client, tmp_path / ".pretensor", runner=fake_runner)
    assert "registration failed" in result
    assert "claude binary not found" in result


def test_register_non_dict_json_root(tmp_path: Path) -> None:
    path = tmp_path / ".cursor" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([]), encoding="utf-8")
    original_content = path.read_bytes()

    client = McpClient(key="cursor", label="Cursor", config_path=path)
    try:
        register_client(client, tmp_path / ".pretensor")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected a JSON object" in str(e)
        assert path.read_bytes() == original_content
        assert list(path.parent.glob("mcp.json.bak.*")) == []


def test_register_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / ".cursor" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text("{invalid json", encoding="utf-8")
    original_content = path.read_bytes()

    client = McpClient(key="cursor", label="Cursor", config_path=path)
    try:
        register_client(client, tmp_path / ".pretensor")
        assert False, "Should have raised JSONDecodeError"
    except json.JSONDecodeError:
        assert path.read_bytes() == original_content
        assert list(path.parent.glob("mcp.json.bak.*")) == []


def test_register_mcpservers_null(tmp_path: Path) -> None:
    path = _cursor_config(tmp_path, {"mcpServers": None})
    original_content = path.read_bytes()

    client = McpClient(key="cursor", label="Cursor", config_path=path)
    try:
        register_client(client, tmp_path / ".pretensor")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected mcpServers to be a dict" in str(e)
        assert path.read_bytes() == original_content
        assert list(path.parent.glob("mcp.json.bak.*")) == []


def test_register_mcpservers_non_dict(tmp_path: Path) -> None:
    path = _cursor_config(tmp_path, {"mcpServers": "not-a-dict"})
    original_content = path.read_bytes()

    client = McpClient(key="cursor", label="Cursor", config_path=path)
    try:
        register_client(client, tmp_path / ".pretensor")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected mcpServers to be a dict" in str(e)
        assert path.read_bytes() == original_content
        assert list(path.parent.glob("mcp.json.bak.*")) == []
