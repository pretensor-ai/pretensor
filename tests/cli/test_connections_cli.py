"""CLI tests for ``pretensor add`` and ``pretensor remove`` connection commands."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from pretensor.cli.main import app

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _ANSI_ESCAPE_RE.sub("", text)).strip()


def _write_registry(state_dir: Path, entries: dict[str, Any]) -> None:
    reg_path = state_dir / "registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "entries": entries}
    reg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_registry(state_dir: Path) -> dict[str, Any]:
    reg_path = state_dir / "registry.json"
    return json.loads(reg_path.read_text(encoding="utf-8"))


class _FakeDSNEncryptor:
    """Minimal stub that stores ciphertexts as ``enc:<plaintext>``."""

    def __init__(self, _path: Path) -> None:
        pass

    def encrypt(self, value: str) -> str:
        return f"enc:{value}"

    def decrypt(self, value: str) -> str:
        return value.removeprefix("enc:")


def test_add_help_shows_options() -> None:
    """``pretensor add --help`` documents key options."""
    result = CliRunner().invoke(app, ["add", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--name" in plain or "-n" in plain
    assert "--state-dir" in plain


def test_add_registers_connection(tmp_path: Path) -> None:
    """``add`` writes a registry entry and prints a confirmation message."""
    dsn = "postgresql://u:p@localhost/mydb"

    with patch(
        "pretensor.cli.commands.connections.add_remove.DSNEncryptor",
        _FakeDSNEncryptor,
    ):
        result = CliRunner().invoke(
            app,
            ["add", dsn, "--state-dir", str(tmp_path)],
        )

    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "Registered" in plain

    reg = _read_registry(tmp_path)
    assert "mydb" in reg["entries"]


def test_add_name_flag_overrides_default(tmp_path: Path) -> None:
    """``--name`` sets the logical connection name in the registry."""
    dsn = "postgresql://u:p@localhost/mydb"

    with patch(
        "pretensor.cli.commands.connections.add_remove.DSNEncryptor",
        _FakeDSNEncryptor,
    ):
        result = CliRunner().invoke(
            app,
            ["add", dsn, "--name", "my-warehouse", "--state-dir", str(tmp_path)],
        )

    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "my-warehouse" in plain

    reg = _read_registry(tmp_path)
    assert "my-warehouse" in reg["entries"]


def test_add_prints_index_hint(tmp_path: Path) -> None:
    """``add`` prints a hint to run ``pretensor index`` next."""
    dsn = "postgresql://u:p@localhost/mydb"

    with patch(
        "pretensor.cli.commands.connections.add_remove.DSNEncryptor",
        _FakeDSNEncryptor,
    ):
        result = CliRunner().invoke(
            app,
            ["add", dsn, "--state-dir", str(tmp_path)],
        )

    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "pretensor index" in plain


def test_add_encrypts_dsn_in_registry(tmp_path: Path) -> None:
    """The DSN stored in registry.json must be the encrypted form, not plaintext."""
    dsn = "postgresql://u:p@localhost/secretdb"

    with patch(
        "pretensor.cli.commands.connections.add_remove.DSNEncryptor",
        _FakeDSNEncryptor,
    ):
        CliRunner().invoke(
            app,
            ["add", dsn, "--state-dir", str(tmp_path)],
        )

    reg = _read_registry(tmp_path)
    entry = reg["entries"]["secretdb"]
    assert entry.get("dsn", "") != dsn
    assert entry.get("dsn_encrypted") is not None


def test_remove_help_shows_options() -> None:
    """``pretensor remove --help`` documents the name argument and state-dir."""
    result = CliRunner().invoke(app, ["remove", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--state-dir" in plain


def test_remove_deletes_connection(tmp_path: Path) -> None:
    """``remove`` drops the entry from registry.json and prints confirmation."""
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
    graph_file = tmp_path / "graphs" / "unified.kuzu"
    graph_file.parent.mkdir(parents=True)
    graph_file.touch()
    _write_registry(
        tmp_path,
        {
            "mydb": {
                "connection_name": "mydb",
                "database": "mydb",
                "dsn": "postgresql://u:p@h/mydb",
                "graph_path": str(graph_file),
                "unified_graph_path": str(graph_file),
                "last_indexed_at": ts,
                "dialect": "postgres",
                "table_count": 3,
                "dsn_encrypted": None,
            }
        },
    )

    result = CliRunner().invoke(
        app,
        ["remove", "mydb", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "Removed" in plain
    assert "mydb" in plain

    reg = _read_registry(tmp_path)
    assert "mydb" not in reg["entries"]


def test_remove_unknown_connection_exits_1(tmp_path: Path) -> None:
    """Removing a connection that is not in the registry exits with code 1."""
    _write_registry(tmp_path, {})

    result = CliRunner().invoke(
        app,
        ["remove", "nonexistent", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "nonexistent" in plain or "Unknown connection" in plain


def test_remove_missing_registry_still_exits_gracefully(tmp_path: Path) -> None:
    """``remove`` with no registry does not crash unexpectedly (exits non-zero or prints error)."""
    result = CliRunner().invoke(
        app,
        ["remove", "nonexistent", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code != 0 or "Unknown connection" in _normalize(result.stdout)


def test_add_idempotent_overwrites_existing(tmp_path: Path) -> None:
    """Running ``add`` twice with the same name overwrites the previous entry."""
    dsn1 = "postgresql://u:p@localhost/mydb"
    dsn2 = "postgresql://u:p@other-host/mydb"

    with patch(
        "pretensor.cli.commands.connections.add_remove.DSNEncryptor",
        _FakeDSNEncryptor,
    ):
        CliRunner().invoke(app, ["add", dsn1, "--state-dir", str(tmp_path)])
        CliRunner().invoke(app, ["add", dsn2, "--state-dir", str(tmp_path)])

    reg = _read_registry(tmp_path)
    entries = reg["entries"]
    assert len([k for k in entries if k == "mydb"]) == 1
