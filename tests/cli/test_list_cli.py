"""CLI tests for ``pretensor list`` command."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def test_list_no_registry(tmp_path: Path) -> None:
    """When no registry.json exists, list exits 0 and prints a 'no registry' hint."""
    result = CliRunner().invoke(app, ["list", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No registry found" in _normalize(result.stdout)


def test_list_empty_registry(tmp_path: Path) -> None:
    """An existing but empty registry prints 'Registry is empty'."""
    _write_registry(tmp_path, {})
    result = CliRunner().invoke(app, ["list", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Registry is empty" in _normalize(result.stdout)


def test_list_shows_connection_details(tmp_path: Path) -> None:
    """Each entry's connection name, database, and graph path appear in output."""
    graph_file = tmp_path / "graphs" / "mydb.kuzu"
    graph_file.parent.mkdir(parents=True)
    graph_file.touch()
    _write_registry(
        tmp_path,
        {
            "mydb": {
                "connection_name": "mydb",
                "database": "mydb",
                "dsn": "postgresql://u:p@host/mydb",
                "graph_path": str(graph_file),
                "unified_graph_path": None,
                "last_indexed_at": datetime(2024, 6, 1, tzinfo=timezone.utc).isoformat(),
                "dialect": "postgres",
                "table_count": 42,
            }
        },
    )
    result = CliRunner().invoke(app, ["list", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    compact = result.stdout.replace("\n", "").replace(" ", "")
    assert "mydb" in plain
    assert "db=mydb" in plain
    assert "tables=42" in plain
    assert str(graph_file).replace(" ", "") in compact


def test_list_multiple_entries_sorted(tmp_path: Path) -> None:
    """Multiple entries are emitted in alphabetical order by connection name."""
    graph_file = tmp_path / "graphs" / "x.kuzu"
    graph_file.parent.mkdir(parents=True)
    graph_file.touch()
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
    _write_registry(
        tmp_path,
        {
            "zebra": {
                "connection_name": "zebra",
                "database": "zebra",
                "dsn": "postgresql://u:p@h/z",
                "graph_path": str(graph_file),
                "unified_graph_path": None,
                "last_indexed_at": ts,
                "dialect": "postgres",
                "table_count": None,
            },
            "alpha": {
                "connection_name": "alpha",
                "database": "alpha",
                "dsn": "postgresql://u:p@h/a",
                "graph_path": str(graph_file),
                "unified_graph_path": None,
                "last_indexed_at": ts,
                "dialect": "postgres",
                "table_count": None,
            },
        },
    )
    result = CliRunner().invoke(app, ["list", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    plain = result.stdout
    assert plain.index("alpha") < plain.index("zebra")


def test_list_shows_unified_graph_path(tmp_path: Path) -> None:
    """When unified_graph_path is set, it should appear in the list output."""
    unified = tmp_path / "graphs" / "unified.kuzu"
    unified.parent.mkdir(parents=True)
    unified.touch()
    individual = tmp_path / "graphs" / "db.kuzu"
    individual.touch()
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
    _write_registry(
        tmp_path,
        {
            "db": {
                "connection_name": "db",
                "database": "db",
                "dsn": "postgresql://u:p@h/db",
                "graph_path": str(individual),
                "unified_graph_path": str(unified),
                "last_indexed_at": ts,
                "dialect": "postgres",
                "table_count": 5,
            }
        },
    )
    result = CliRunner().invoke(app, ["list", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    # Rich may wrap long paths; collapse all whitespace to check the path fragments.
    compact = result.stdout.replace("\n", "").replace(" ", "")
    assert "unified.kuzu" in compact


def test_list_help_shows_state_dir_option() -> None:
    """``pretensor list --help`` documents the --state-dir option."""
    result = CliRunner().invoke(app, ["list", "--help"])
    assert result.exit_code == 0
    assert "--state-dir" in _normalize(result.stdout)
