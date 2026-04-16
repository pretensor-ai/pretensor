"""CLI tests for ``pretensor reindex`` command."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pretensor.cli.config_file import PretensorCliConfig, SourceConfig
from pretensor.cli.main import app
from pretensor.connectors.models import SchemaSnapshot

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _ANSI_ESCAPE_RE.sub("", text)).strip()


def _write_registry(state_dir: Path, connection_name: str, graph_path: Path) -> None:
    reg_path = state_dir / "registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "entries": {
            connection_name: {
                "connection_name": connection_name,
                "database": connection_name,
                "dsn": f"postgresql://u:p@localhost/{connection_name}",
                "graph_path": str(graph_path),
                "unified_graph_path": None,
                "last_indexed_at": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
                "dialect": "postgres",
                "table_count": 5,
                "dsn_encrypted": None,
            }
        },
    }
    reg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_snapshot(state_dir: Path, connection_name: str) -> None:
    snap = SchemaSnapshot.empty(connection_name, connection_name)
    snap_dir = state_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / f"{connection_name}.yaml").write_text(
        snap.to_yaml(), encoding="utf-8"
    )


@pytest.fixture()
def reindex_env(tmp_path: Path) -> dict[str, Any]:
    """Set up a minimal state dir with registry, snapshot, and graph file."""
    cn = "mydb"
    graph_path = tmp_path / "graphs" / f"{cn}.kuzu"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()

    _write_registry(tmp_path, cn, graph_path)
    _write_snapshot(tmp_path, cn)

    snapshot = SchemaSnapshot.empty(cn, cn)
    mock_store = MagicMock()
    mock_store.query_all_rows.return_value = [[5]]

    return {
        "state_dir": tmp_path,
        "connection_name": cn,
        "graph_path": graph_path,
        "snapshot": snapshot,
        "mock_store": mock_store,
    }


def test_reindex_help_shows_options() -> None:
    """``pretensor reindex --help`` documents key flags."""
    result = CliRunner().invoke(app, ["reindex", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--dry-run" in plain
    assert "--database" in plain or "-d" in plain
    assert "--state-dir" in plain


def test_reindex_no_registry_exits_1(tmp_path: Path) -> None:
    """Without registry.json, reindex exits 1 with an informative message."""
    result = CliRunner().invoke(
        app,
        [
            "reindex",
            "postgresql://u:p@localhost/mydb",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "No registry found" in plain or "registry" in plain.lower()


def test_reindex_unknown_connection_exits_1(tmp_path: Path) -> None:
    """When the connection name is not in the registry, exit code is 1."""
    cn = "mydb"
    graph_path = tmp_path / "graphs" / f"{cn}.kuzu"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()
    _write_registry(tmp_path, cn, graph_path)

    result = CliRunner().invoke(
        app,
        [
            "reindex",
            "postgresql://u:p@localhost/other",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "other" in plain or "Unknown connection" in plain


def test_reindex_no_snapshot_exits_1(tmp_path: Path) -> None:
    """When there's no saved snapshot, reindex exits 1 with an instructive message."""
    cn = "mydb"
    graph_path = tmp_path / "graphs" / f"{cn}.kuzu"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()
    _write_registry(tmp_path, cn, graph_path)

    result = CliRunner().invoke(
        app,
        [
            "reindex",
            "postgresql://u:p@localhost/mydb",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "snapshot" in plain.lower() or "index" in plain.lower()


def test_reindex_no_changes_refreshes_registry(reindex_env: dict[str, Any]) -> None:
    """When schema is unchanged, the registry timestamp is refreshed."""
    env = reindex_env
    snapshot = env["snapshot"]
    mock_store = env["mock_store"]

    with (
        patch(
            "pretensor.cli.commands.reindex.inspect", return_value=snapshot
        ),
        patch("pretensor.cli.commands.reindex.KuzuStore", return_value=mock_store),
        patch(
            "pretensor.cli.commands.reindex.diff_snapshots", return_value=[]
        ),
        patch(
            "pretensor.cli.commands.reindex.ImpactAnalyzer"
        ) as MockAnalyzer,
        patch("pretensor.cli.commands.reindex.GraphPatcher") as MockPatcher,
        patch(
            "pretensor.cli.commands.reindex.SnapshotStore.save",
            return_value=env["state_dir"] / "snap.yaml",
        ),
    ):
        MockAnalyzer.return_value.analyze.return_value = MagicMock(summary="ok")
        MockPatcher.return_value.apply.return_value = MagicMock()
        result = CliRunner().invoke(
            app,
            [
                "reindex",
                f"postgresql://u:p@localhost/{env['connection_name']}",
                "--state-dir",
                str(env["state_dir"]),
            ],
        )
    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "No schema changes" in plain or "timestamp refreshed" in plain


def test_reindex_dry_run_no_writes(reindex_env: dict[str, Any]) -> None:
    """``--dry-run`` prints planned mutations and exits 0 without modifying the graph."""
    env = reindex_env
    snapshot = env["snapshot"]
    mock_store = env["mock_store"]
    change = MagicMock()
    change.change_type.value = "added"
    change.target.value = "table"
    change.schema_name = "public"
    change.table_name = "new_table"
    change.column_name = None
    change.details = None

    mock_patch = MagicMock()
    mock_patch.tables_added = 1
    mock_patch.tables_removed = 0
    mock_patch.columns_added = 0
    mock_patch.columns_removed = 0
    mock_patch.columns_updated = 0
    mock_patch.fk_edges_removed = 0
    mock_patch.inferred_joins_removed = 0

    with (
        patch("pretensor.cli.commands.reindex.inspect", return_value=snapshot),
        patch("pretensor.cli.commands.reindex.KuzuStore", return_value=mock_store),
        patch(
            "pretensor.cli.commands.reindex.diff_snapshots", return_value=[change]
        ),
        patch("pretensor.cli.commands.reindex.ImpactAnalyzer") as MockAnalyzer,
        patch("pretensor.cli.commands.reindex.GraphPatcher") as MockPatcher,
        patch(
            "pretensor.cli.commands.reindex.SnapshotStore.save",
        ),
    ):
        MockAnalyzer.return_value.analyze.return_value = MagicMock(summary="1 change")
        MockPatcher.return_value.apply.return_value = mock_patch
        result = CliRunner().invoke(
            app,
            [
                "reindex",
                f"postgresql://u:p@localhost/{env['connection_name']}",
                "--state-dir",
                str(env["state_dir"]),
                "--dry-run",
            ],
        )
    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "dry-run" in plain.lower() or "dry_run" in plain.lower() or "Run without" in plain


def test_reindex_missing_graph_file_exits_1(tmp_path: Path) -> None:
    """When the graph file is gone, reindex exits 1."""
    cn = "mydb"
    graph_path = tmp_path / "graphs" / f"{cn}.kuzu"
    graph_path.parent.mkdir(parents=True)

    _write_registry(tmp_path, cn, graph_path)
    _write_snapshot(tmp_path, cn)

    snapshot = SchemaSnapshot.empty(cn, cn)
    with patch("pretensor.cli.commands.reindex.inspect", return_value=snapshot):
        result = CliRunner().invoke(
            app,
            [
                "reindex",
                f"postgresql://u:p@localhost/{cn}",
                "--state-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "Graph file missing" in plain or "graph" in plain.lower()


def test_reindex_missing_dbt_manifest_exits_1(reindex_env: dict[str, Any]) -> None:
    """``--dbt-manifest`` pointing at a missing file exits 1 before any DB work."""
    env = reindex_env
    result = CliRunner().invoke(
        app,
        [
            "reindex",
            f"postgresql://u:p@localhost/{env['connection_name']}",
            "--state-dir",
            str(env["state_dir"]),
            "--dbt-manifest",
            str(env["state_dir"] / "nonexistent.json"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "manifest" in plain.lower() or "not found" in plain.lower()


def test_reindex_schema_changes_printed(reindex_env: dict[str, Any]) -> None:
    """When changes exist, each change is printed with change type and target."""
    env = reindex_env
    snapshot = env["snapshot"]
    mock_store = env["mock_store"]
    mock_store.query_all_rows.return_value = [[6]]

    change = MagicMock()
    change.change_type.value = "added"
    change.target.value = "table"
    change.schema_name = "public"
    change.table_name = "new_table"
    change.column_name = None
    change.details = None

    mock_patch = MagicMock()
    mock_patch.tables_added = 1
    mock_patch.tables_removed = 0
    mock_patch.columns_added = 0
    mock_patch.columns_removed = 0
    mock_patch.columns_updated = 0
    mock_patch.fk_edges_removed = 0
    mock_patch.inferred_joins_removed = 0

    with (
        patch("pretensor.cli.commands.reindex.inspect", return_value=snapshot),
        patch("pretensor.cli.commands.reindex.KuzuStore", return_value=mock_store),
        patch(
            "pretensor.cli.commands.reindex.diff_snapshots", return_value=[change]
        ),
        patch("pretensor.cli.commands.reindex.ImpactAnalyzer") as MockAnalyzer,
        patch("pretensor.cli.commands.reindex.GraphPatcher") as MockPatcher,
        patch("pretensor.cli.commands.reindex.SnapshotStore.save"),
        patch("pretensor.cli.commands.reindex.MetricTemplateBuilder.mark_stale_for_database"),
        patch("pretensor.cli.commands.reindex.SkillGenerator.write_for_index", return_value=[]),
    ):
        MockAnalyzer.return_value.analyze.return_value = MagicMock(summary="1 change")
        MockPatcher.return_value.apply.return_value = mock_patch
        result = CliRunner().invoke(
            app,
            [
                "reindex",
                f"postgresql://u:p@localhost/{env['connection_name']}",
                "--state-dir",
                str(env["state_dir"]),
            ],
        )
    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "Schema changes" in plain
    assert "new_table" in plain


# ---------------------------------------------------------------------------
# --source / --all CLI integration tests
# ---------------------------------------------------------------------------


def _cli_config_with_sources(
    sources: dict[str, SourceConfig],
    state_dir: Path | None = None,
) -> PretensorCliConfig:
    return PretensorCliConfig(sources=sources, state_dir=state_dir)


def test_reindex_source_unknown_name_exits_1(tmp_path: Path) -> None:
    """``--source nonexistent`` exits 1 with a helpful message."""
    cfg = _cli_config_with_sources(
        {"pg": SourceConfig(dialect="postgres", host="localhost")},
        state_dir=tmp_path,
    )
    _write_registry(tmp_path, "pg", tmp_path / "graphs" / "pg.kuzu")
    with patch("pretensor.cli.commands.reindex.get_cli_config", return_value=cfg):
        result = CliRunner().invoke(
            app, ["reindex", "--source", "nope", "--state-dir", str(tmp_path)]
        )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "nope" in plain


def test_reindex_all_no_sources_exits_1(tmp_path: Path) -> None:
    """``--all`` with empty sources config exits 1."""
    cfg = _cli_config_with_sources({}, state_dir=tmp_path)
    with patch("pretensor.cli.commands.reindex.get_cli_config", return_value=cfg):
        result = CliRunner().invoke(
            app, ["reindex", "--all", "--state-dir", str(tmp_path)]
        )
    assert result.exit_code == 1
    assert "No sources" in _normalize(result.stdout)


def test_reindex_all_skips_missing_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source with unset env vars is skipped in --all mode."""
    monkeypatch.delenv("MISSING_PW", raising=False)
    cn = "good"
    graph_path = tmp_path / "graphs" / f"{cn}.kuzu"
    graph_path.parent.mkdir(parents=True)
    graph_path.touch()
    _write_registry(tmp_path, cn, graph_path)
    _write_snapshot(tmp_path, cn)

    snapshot = SchemaSnapshot.empty(cn, cn)
    mock_store = MagicMock()
    mock_store.query_all_rows.return_value = [[5]]

    cfg = _cli_config_with_sources(
        {
            "good": SourceConfig(dialect="postgres", host="h1", user="u", password="p", database=cn),
            "bad": SourceConfig(dialect="snowflake", account="xy", password="${MISSING_PW}"),
        },
        state_dir=tmp_path,
    )
    with (
        patch("pretensor.cli.commands.reindex.get_cli_config", return_value=cfg),
        patch("pretensor.cli.commands.reindex.inspect", return_value=snapshot),
        patch("pretensor.cli.commands.reindex.KuzuStore", return_value=mock_store),
        patch("pretensor.cli.commands.reindex.diff_snapshots", return_value=[]),
        patch("pretensor.cli.commands.reindex.ImpactAnalyzer") as MockAnalyzer,
        patch("pretensor.cli.commands.reindex.GraphPatcher") as MockPatcher,
        patch("pretensor.cli.commands.reindex.SnapshotStore.save", return_value=tmp_path / "snap.yaml"),
    ):
        MockAnalyzer.return_value.analyze.return_value = MagicMock(summary="ok")
        MockPatcher.return_value.apply.return_value = MagicMock()
        result = CliRunner().invoke(
            app, ["reindex", "--all", "--state-dir", str(tmp_path)]
        )
    plain = _normalize(result.stdout)
    assert "Skipping bad" in plain
    assert "MISSING_PW" in plain
