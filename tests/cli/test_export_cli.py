"""CLI tests for ``pretensor export`` command."""

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


def test_export_help_shows_required_options() -> None:
    """``pretensor export --help`` documents required flags."""
    result = CliRunner().invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--database" in plain
    assert "--output" in plain
    assert "--state-dir" in plain


def test_export_no_registry_exits_1(tmp_path: Path) -> None:
    """When no registry exists, export exits non-zero with a hint."""
    out_file = tmp_path / "graph.json"
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--database",
            "mydb",
            "--output",
            str(out_file),
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "No registry found" in _normalize(result.stdout)


def test_export_unknown_connection_exits_1(tmp_path: Path) -> None:
    """Unknown ``--database`` value exits 1 with a lookup hint."""
    graph_path = tmp_path / "graphs" / "mydb.kuzu"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.touch()
    _write_registry(
        tmp_path,
        {
            "mydb": {
                "connection_name": "mydb",
                "database": "mydb",
                "dsn": "postgresql://u:p@localhost/mydb",
                "graph_path": str(graph_path),
                "unified_graph_path": None,
                "last_indexed_at": datetime(2024, 6, 1, tzinfo=timezone.utc).isoformat(),
                "dialect": "postgres",
                "table_count": 2,
            }
        },
    )
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--database",
            "otherdb",
            "--output",
            str(tmp_path / "graph.json"),
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "Unknown connection" in plain
    assert "otherdb" in plain


def test_export_missing_graph_file_exits_1(tmp_path: Path) -> None:
    """Registry entries with a missing graph file exit 1."""
    graph_path = tmp_path / "graphs" / "mydb.kuzu"
    _write_registry(
        tmp_path,
        {
            "mydb": {
                "connection_name": "mydb",
                "database": "mydb",
                "dsn": "postgresql://u:p@localhost/mydb",
                "graph_path": str(graph_path),
                "unified_graph_path": None,
                "last_indexed_at": datetime(2024, 6, 1, tzinfo=timezone.utc).isoformat(),
                "dialect": "postgres",
                "table_count": 2,
            }
        },
    )
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--database",
            "mydb",
            "--output",
            str(tmp_path / "graph.json"),
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "Graph file missing" in _normalize(result.stdout)


def test_export_writes_pretty_json_file(tmp_path: Path) -> None:
    """A successful export writes pretty-printed JSON and summary output."""
    graph_path = tmp_path / "graphs" / "mydb.kuzu"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.touch()
    _write_registry(
        tmp_path,
        {
            "mydb": {
                "connection_name": "mydb",
                "database": "mydb",
                "dsn": "postgresql://u:p@localhost/mydb",
                "graph_path": str(graph_path),
                "unified_graph_path": None,
                "last_indexed_at": datetime(2024, 6, 1, tzinfo=timezone.utc).isoformat(),
                "dialect": "postgres",
                "table_count": 2,
            }
        },
    )
    payload = {
        "connection_name": "mydb",
        "database": "mydb",
        "graph_path": str(graph_path),
        "exported_at": "2026-04-13T00:00:00+00:00",
        "node_types": [
            {
                "type": "SchemaTable",
                "properties": ["node_id", "table_name"],
                "rows": [{"node_id": "t1", "table_name": "orders"}],
            }
        ],
        "edge_types": [
            {
                "type": "FK_REFERENCES",
                "properties": ["edge_id", "source_column", "target_column"],
                "rows": [
                    {
                        "__from_node_id": "t1",
                        "__to_node_id": "t2",
                        "edge_id": "e1",
                        "source_column": "customer_id",
                        "target_column": "id",
                    }
                ],
            }
        ],
        "stats": {
            "node_types": 1,
            "edge_types": 1,
            "node_count": 1,
            "edge_count": 1,
        },
    }
    output_path = tmp_path / "exports" / "graph.json"
    with patch(
        "pretensor.cli.commands.export.export_graph_payload",
        return_value=payload,
    ):
        result = CliRunner().invoke(
            app,
            [
                "export",
                "--database",
                "mydb",
                "--output",
                str(output_path),
                "--state-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 0, _normalize(result.stdout)
    raw = output_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert "\n  \"" in raw
    assert json.loads(raw) == payload
    plain = _normalize(result.stdout)
    assert "Graph exported" in plain
    assert "node_types=1" in plain
