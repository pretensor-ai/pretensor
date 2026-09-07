"""Detect installed MCP clients and register the pretensor server with them.

Claude Code is registered through its own CLI rather than by editing
``~/.claude.json``: that file is large and stateful, holding far more than MCP
registrations, and rewriting it from a third-party tool risks damage well
outside the scope of this feature.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["McpClient", "detect_clients", "register_client"]


@dataclass(frozen=True, slots=True)
class McpClient:
    """An MCP client that pretensor can register itself with.

    ``config_path`` is None for clients managed through their own CLI.
    """

    key: str
    label: str
    config_path: Path | None


def detect_clients(
    *, home: Path, cwd: Path, env: Mapping[str, str] | None = None
) -> tuple[McpClient, ...]:
    """Return the MCP clients that appear to be installed.

    ``env`` defaults to ``os.environ`` and is only consulted to probe the
    Windows Claude Desktop config path (``%APPDATA%``); it is a keyword so
    tests can point it at a fake environment without touching the real one.
    """
    if env is None:
        env = os.environ
    found: list[McpClient] = []

    if shutil.which("claude") is not None:
        found.append(
            McpClient(key="claude-code", label="Claude Code", config_path=None)
        )

    desktop_candidates = [
        home
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json",
        home / ".config" / "Claude" / "claude_desktop_config.json",
    ]
    appdata = env.get("APPDATA")
    if appdata:
        desktop_candidates.append(
            Path(appdata) / "Claude" / "claude_desktop_config.json"
        )

    for candidate in desktop_candidates:
        if candidate.is_file():
            found.append(
                McpClient(
                    key="claude-desktop",
                    label="Claude Desktop",
                    config_path=candidate,
                )
            )
            break

    for candidate in (cwd / ".cursor" / "mcp.json", home / ".cursor" / "mcp.json"):
        if candidate.is_file():
            found.append(McpClient(key="cursor", label="Cursor", config_path=candidate))
            break

    return tuple(found)


def register_client(
    client: McpClient,
    state_dir: Path,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> str:
    """Register the pretensor MCP server with *client*. Returns a result line."""
    resolved = str(Path(state_dir).resolve())
    if client.config_path is None:
        result = runner(
            [
                "claude",
                "mcp",
                "add",
                "pretensor",
                "--",
                "pretensor",
                "serve",
                "--state-dir",
                resolved,
            ],
            capture_output=True,
            text=True,
        )
        if getattr(result, "returncode", 1) != 0:
            detail = (getattr(result, "stderr", "") or "").strip()
            return f"{client.label}: registration failed. {detail}"
        return f"{client.label}: registered."

    payload = _load_json(client.config_path)
    servers = payload.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        msg = f"Expected mcpServers to be a dict in {client.config_path}"
        raise ValueError(msg)
    servers["pretensor"] = {
        "command": "pretensor",
        "args": ["serve", "--state-dir", resolved],
    }
    # Only back up once the file has been proven readable and valid: a failed
    # registration must not strand a `.bak` copy behind.
    backup = _backup(client.config_path)
    _atomic_write_json(client.config_path, payload)
    return f"{client.label}: updated {client.config_path} (backup at {backup.name})."


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        msg = f"Expected a JSON object at {path}"
        raise ValueError(msg)
    return loaded


def _backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    shutil.copy2(path, backup)
    return backup


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    # mkstemp creates the temp file 0600, which would silently tighten an
    # existing config's permissions on replace; preserve the original mode
    # when there was one to preserve.
    original_mode: int | None = None
    if path.exists():
        original_mode = stat.S_IMODE(path.stat().st_mode)

    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        if original_mode is not None:
            os.chmod(temp_name, original_mode)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
