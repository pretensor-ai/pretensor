"""CLI tests for ``pretensor index`` command."""

from __future__ import annotations

import json
import re
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


def _empty_snapshot(connection_name: str = "testdb") -> SchemaSnapshot:
    return SchemaSnapshot.empty(connection_name, connection_name)


def _fake_kuzu_store(tmp_path: Path) -> MagicMock:
    """Return a mock KuzuStore that mimics the real interface."""
    store = MagicMock()
    store.query_all_rows.return_value = [[5]]
    return store


@pytest.fixture()
def patched_index(tmp_path: Path) -> Any:
    """Patch the heavy dependencies of ``pretensor index`` for unit testing."""
    snapshot = _empty_snapshot("testdb")
    mock_store = MagicMock()
    mock_store.query_all_rows.return_value = [[3]]

    with (
        patch(
            "pretensor.cli.commands.index.inspect", return_value=snapshot
        ) as mock_inspect,
        patch(
            "pretensor.cli.commands.index.KuzuStore", return_value=mock_store
        ),
        patch("pretensor.cli.commands.index.GraphBuilder") as MockBuilder,
        patch(
            "pretensor.cli.commands.index.SkillGenerator.write_for_index",
            return_value=[],
        ),
        patch(
            "pretensor.cli.commands.index.SnapshotStore.save",
            return_value=tmp_path / "snap.yaml",
        ),
    ):
        MockBuilder.return_value.build.return_value = None
        yield {
            "mock_inspect": mock_inspect,
            "mock_store": mock_store,
            "mock_builder": MockBuilder,
        }


def test_index_help_shows_dsn_argument() -> None:
    """``pretensor index --help`` documents the DSN argument and key options."""
    result = CliRunner().invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "DSN" in plain or "dsn" in plain.lower()
    assert "--state-dir" in plain
    assert "--name" in plain or "-n" in plain


def test_index_writes_graph_and_registry(tmp_path: Path, patched_index: Any) -> None:
    """A successful ``index`` run prints 'Graph written' and 'Registry updated'."""
    result = CliRunner().invoke(
        app,
        ["index", "postgresql://u:p@localhost/testdb", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "Graph written" in plain
    assert "Registry updated" in plain


def test_index_name_flag_overrides_default(tmp_path: Path, patched_index: Any) -> None:
    """``--name`` changes the logical connection name used in output."""
    result = CliRunner().invoke(
        app,
        [
            "index",
            "postgresql://u:p@localhost/testdb",
            "--name",
            "custom-conn",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "custom-conn" in plain


def test_index_inspect_is_called(tmp_path: Path, patched_index: Any) -> None:
    """``inspect`` is invoked exactly once with a valid config."""
    CliRunner().invoke(
        app,
        ["index", "postgresql://u:p@localhost/testdb", "--state-dir", str(tmp_path)],
    )
    patched_index["mock_inspect"].assert_called_once()


def test_index_invalid_visibility_profile_exits_1(
    tmp_path: Path, patched_index: Any
) -> None:
    """A bad ``--profile`` (unknown profile name) exits with code 1."""
    vis_file = tmp_path / "visibility.yml"
    vis_file.write_text(
        "hidden_schemas: []\nhidden_tables: []\nprofiles: {}\n", encoding="utf-8"
    )

    with patch(
        "pretensor.cli.commands.index.merge_profile_into_base",
        side_effect=ValueError("unknown profile 'nope'"),
    ):
        result = CliRunner().invoke(
            app,
            [
                "index",
                "postgresql://u:p@localhost/testdb",
                "--state-dir",
                str(tmp_path),
                "--profile",
                "nope",
            ],
        )
    assert result.exit_code == 1
    assert "nope" in _normalize(result.stdout)


def test_index_missing_dbt_manifest_exits_1(
    tmp_path: Path, patched_index: Any
) -> None:
    """A ``--dbt-manifest`` pointing to a non-existent file exits 1."""
    result = CliRunner().invoke(
        app,
        [
            "index",
            "postgresql://u:p@localhost/testdb",
            "--state-dir",
            str(tmp_path),
            "--dbt-manifest",
            str(tmp_path / "nope.json"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "not found" in plain.lower() or "manifest" in plain.lower()


def test_index_unified_flag_uses_unified_graph_path(
    tmp_path: Path, patched_index: Any
) -> None:
    """``--unified`` includes 'unified' in the output path."""
    result = CliRunner().invoke(
        app,
        [
            "index",
            "postgresql://u:p@localhost/testdb",
            "--state-dir",
            str(tmp_path),
            "--unified",
        ],
    )
    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "unified" in plain


def test_index_introspecting_message_printed(
    tmp_path: Path, patched_index: Any
) -> None:
    """The 'Introspecting' banner appears during a successful run."""
    result = CliRunner().invoke(
        app,
        ["index", "postgresql://u:p@localhost/testdb", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, _normalize(result.stdout)
    assert "Introspecting" in _normalize(result.stdout)


def test_index_json_logs_written_to_file(tmp_path: Path, patched_index: Any) -> None:
    """Global log flags produce JSON logs in the configured log file."""
    log_file = tmp_path / "logs" / "index.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "--log-format",
            "json",
            "--log-file",
            str(log_file),
            "index",
            "postgresql://u:p@localhost/testdb",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, _normalize(result.stdout)
    lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines
    records = [json.loads(line) for line in lines]
    events = {str(r.get("event")) for r in records}
    assert "index.inspect" in events
    assert "index.graph_build" in events
    assert "index.total" in events


# ---------------------------------------------------------------------------
# --source / --all CLI integration tests
# ---------------------------------------------------------------------------


def _cli_config_with_sources(
    sources: dict[str, SourceConfig],
    state_dir: Path | None = None,
) -> PretensorCliConfig:
    return PretensorCliConfig(sources=sources, state_dir=state_dir)


def test_index_source_unknown_name_exits_1(tmp_path: Path) -> None:
    """``--source nonexistent`` exits 1 with a helpful message."""
    cfg = _cli_config_with_sources(
        {"pg": SourceConfig(dialect="postgres", host="localhost")},
        state_dir=tmp_path,
    )
    with patch("pretensor.cli.commands.index.get_cli_config", return_value=cfg):
        result = CliRunner().invoke(app, ["index", "--source", "nope"])
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "nope" in plain
    assert "pg" in plain  # should list available sources


def test_index_source_runs_named_source(tmp_path: Path, patched_index: Any) -> None:
    """``--source pg`` calls inspect with the correct config."""
    cfg = _cli_config_with_sources(
        {"pg": SourceConfig(dialect="postgres", host="localhost", user="u", password="p", database="testdb")},
        state_dir=tmp_path,
    )
    with patch("pretensor.cli.commands.index.get_cli_config", return_value=cfg):
        result = CliRunner().invoke(app, ["index", "--source", "pg"])
    assert result.exit_code == 0, _normalize(result.stdout)
    patched_index["mock_inspect"].assert_called_once()


def test_index_all_no_sources_exits_1(tmp_path: Path) -> None:
    """``--all`` with empty sources config exits 1."""
    cfg = _cli_config_with_sources({}, state_dir=tmp_path)
    with patch("pretensor.cli.commands.index.get_cli_config", return_value=cfg):
        result = CliRunner().invoke(app, ["index", "--all"])
    assert result.exit_code == 1
    assert "No sources" in _normalize(result.stdout)


def test_index_all_iterates_sources(tmp_path: Path, patched_index: Any) -> None:
    """``--all`` indexes every configured source and prints a summary."""
    cfg = _cli_config_with_sources(
        {
            "pg1": SourceConfig(dialect="postgres", host="h1", user="u", password="p", database="db1"),
            "pg2": SourceConfig(dialect="postgres", host="h2", user="u", password="p", database="db2"),
        },
        state_dir=tmp_path,
    )
    with patch("pretensor.cli.commands.index.get_cli_config", return_value=cfg):
        result = CliRunner().invoke(app, ["index", "--all"])
    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "pg1" in plain
    assert "pg2" in plain
    assert patched_index["mock_inspect"].call_count == 2


def test_index_all_skips_missing_env_vars(
    tmp_path: Path, patched_index: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source with unset env vars is skipped; other sources proceed."""
    monkeypatch.delenv("MISSING_PW", raising=False)
    cfg = _cli_config_with_sources(
        {
            "good": SourceConfig(dialect="postgres", host="h1", user="u", password="p", database="db1"),
            "bad": SourceConfig(dialect="snowflake", account="xy", password="${MISSING_PW}"),
        },
        state_dir=tmp_path,
    )
    with patch("pretensor.cli.commands.index.get_cli_config", return_value=cfg):
        result = CliRunner().invoke(app, ["index", "--all"])
    plain = _normalize(result.stdout)
    assert "Skipping bad" in plain
    assert "MISSING_PW" in plain
    # The good source should still have been indexed
    patched_index["mock_inspect"].assert_called_once()


def test_index_dsn_and_source_mutual_exclusion(tmp_path: Path) -> None:
    """Passing both DSN and --source exits 1."""
    cfg = _cli_config_with_sources(
        {"pg": SourceConfig(dialect="postgres", host="localhost")},
        state_dir=tmp_path,
    )
    with patch("pretensor.cli.commands.index.get_cli_config", return_value=cfg):
        result = CliRunner().invoke(
            app, ["index", "postgresql://u:p@localhost/db", "--source", "pg"]
        )
    assert result.exit_code == 1
    assert "mutually exclusive" in _normalize(result.stdout).lower()
