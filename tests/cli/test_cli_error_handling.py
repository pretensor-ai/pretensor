"""Tests for graceful CLI error handling.

Covers bad DSNs, unreachable databases, corrupt graph/registry files, and
permission-denied scenarios for all write-path CLI commands.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from pretensor.cli.main import app

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_ansi(text)).strip()


# ---------------------------------------------------------------------------
# pretensor index — bad DSN
# ---------------------------------------------------------------------------


def test_index_bad_dsn_no_scheme(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["index", "not-a-dsn", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "invalid dsn" in out.lower() or "dsn must be" in out.lower()


def test_index_bad_dsn_unknown_scheme(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["index", "mysql://user@localhost/db", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "invalid dsn" in out.lower() or "unsupported" in out.lower()


def test_index_bad_dialect_override(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "index",
            "postgresql://u@h/db",
            "--dialect",
            "oracledb",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "invalid dsn" in out.lower() or "unknown dialect" in out.lower()


# ---------------------------------------------------------------------------
# pretensor index — unreachable database
# ---------------------------------------------------------------------------


def test_index_unreachable_database(tmp_path: Path) -> None:
    with patch(
        "pretensor.cli.commands.index.inspect",
        side_effect=OSError("Connection refused"),
    ):
        result = CliRunner().invoke(
            app,
            [
                "index",
                "postgresql://u@localhost:9/nonexistent",
                "--state-dir",
                str(tmp_path),
            ],
        )
    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "cannot connect" in out.lower() or "connection refused" in out.lower()


# ---------------------------------------------------------------------------
# pretensor index — state directory not writable
# ---------------------------------------------------------------------------


def test_index_state_dir_not_writable(tmp_path: Path) -> None:
    locked_dir = tmp_path / "locked"
    with patch.object(Path, "mkdir", side_effect=PermissionError("No write")):
        result = CliRunner().invoke(
            app,
            [
                "index",
                "postgresql://u@localhost/db",
                "--state-dir",
                str(locked_dir),
            ],
        )
    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "permission" in out.lower() or "state directory" in out.lower()


# ---------------------------------------------------------------------------
# pretensor index — corrupt registry
# ---------------------------------------------------------------------------


def test_index_corrupt_registry(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text("{ INVALID JSON }", encoding="utf-8")

    with (
        patch(
            "pretensor.cli.commands.index.inspect",
            return_value=MagicMock(
                database="db",
                schemas=["public"],
                tables=[],
            ),
        ),
        patch(
            "pretensor.cli.commands.index.GraphBuilder",
            return_value=MagicMock(build=MagicMock()),
        ),
        patch("pretensor.cli.commands.index.KuzuStore"),
    ):
        result = CliRunner().invoke(
            app,
            [
                "index",
                "postgresql://u@localhost/db",
                "--state-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "registry" in out.lower() or "corrupt" in out.lower() or "cannot read" in out.lower()


# ---------------------------------------------------------------------------
# pretensor reindex — unreachable database
# ---------------------------------------------------------------------------


def test_reindex_unreachable_database(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from pretensor.connectors.models import Column, SchemaSnapshot, Table
    from pretensor.core.builder import GraphBuilder
    from pretensor.core.registry import GraphRegistry
    from pretensor.core.store import KuzuStore
    from pretensor.staleness.snapshot_store import SnapshotStore

    graph = tmp_path / "graphs" / "mydb.kuzu"
    graph.parent.mkdir(parents=True)
    store = KuzuStore(graph)
    try:
        snap = SchemaSnapshot(
            connection_name="mydb",
            database="mydb",
            schemas=["public"],
            tables=[
                Table(
                    name="t",
                    schema_name="public",
                    columns=[Column(name="id", data_type="int")],
                )
            ],
            introspected_at=datetime.now(timezone.utc),
        )
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="mydb",
        database="mydb",
        dsn="postgresql://u@localhost/mydb",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()

    SnapshotStore(tmp_path).save("mydb", snap)

    with patch(
        "pretensor.cli.commands.reindex.inspect",
        side_effect=OSError("Connection refused"),
    ):
        result = CliRunner().invoke(
            app,
            [
                "reindex",
                "postgresql://u@localhost/mydb",
                "--state-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "cannot connect" in out.lower() or "connection refused" in out.lower()


# ---------------------------------------------------------------------------
# pretensor reindex — corrupt graph file
# ---------------------------------------------------------------------------


def test_reindex_corrupt_graph_file(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from pretensor.connectors.models import Column, SchemaSnapshot, Table
    from pretensor.core.builder import GraphBuilder
    from pretensor.core.registry import GraphRegistry
    from pretensor.core.store import KuzuStore
    from pretensor.staleness.snapshot_store import SnapshotStore

    graph = tmp_path / "graphs" / "mydb.kuzu"
    graph.parent.mkdir(parents=True)
    store = KuzuStore(graph)
    try:
        snap = SchemaSnapshot(
            connection_name="mydb",
            database="mydb",
            schemas=["public"],
            tables=[
                Table(
                    name="t",
                    schema_name="public",
                    columns=[Column(name="id", data_type="int")],
                )
            ],
            introspected_at=datetime.now(timezone.utc),
        )
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()

    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="mydb",
        database="mydb",
        dsn="postgresql://u@localhost/mydb",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()

    SnapshotStore(tmp_path).save("mydb", snap)

    with (
        patch(
            "pretensor.cli.commands.reindex.inspect",
            return_value=snap,
        ),
        patch(
            "pretensor.cli.commands.reindex.KuzuStore",
            side_effect=RuntimeError("Corrupt database file"),
        ),
    ):
        result = CliRunner().invoke(
            app,
            [
                "reindex",
                "postgresql://u@localhost/mydb",
                "--state-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "corrupt" in out.lower() or "cannot open" in out.lower()


# ---------------------------------------------------------------------------
# pretensor list — no registry (graceful, exit 0)
# ---------------------------------------------------------------------------


def test_list_no_registry(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["list", "--state-dir", str(tmp_path)])
    assert result.exit_code == 0
    out = _normalize(result.output)
    assert "no registry" in out.lower()


# ---------------------------------------------------------------------------
# pretensor list — corrupt registry
# ---------------------------------------------------------------------------


def test_list_corrupt_registry(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text("{ not valid json }", encoding="utf-8")

    result = CliRunner().invoke(app, ["list", "--state-dir", str(tmp_path)])
    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "registry" in out.lower() or "corrupt" in out.lower() or "cannot read" in out.lower()


# ---------------------------------------------------------------------------
# pretensor sync-grants — bad DSN
# ---------------------------------------------------------------------------


def test_sync_grants_bad_dsn_no_scheme(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["sync-grants", "--dsn", "nope", "--output", str(tmp_path / "vis.yml")],
    )
    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "invalid dsn" in out.lower() or "dsn must be" in out.lower()


def test_sync_grants_bad_dsn_bigquery(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "sync-grants",
            "--dsn",
            "bigquery://proj/dataset",
            "--output",
            str(tmp_path / "vis.yml"),
        ],
    )
    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "bigquery" in out.lower()


def test_sync_grants_connection_error(tmp_path: Path) -> None:
    connector_cm = MagicMock()
    connector_cm.__enter__.side_effect = OSError("Refused")
    connector_cm.__exit__ = MagicMock(return_value=None)

    with patch(
        "pretensor.cli.commands.sync_grants.get_connector",
        return_value=connector_cm,
    ):
        result = CliRunner().invoke(
            app,
            [
                "sync-grants",
                "--dsn",
                "postgresql://u@localhost/db",
                "--output",
                str(tmp_path / "vis.yml"),
            ],
        )
    assert result.exit_code == 1
    out = _normalize(result.output)
    assert "cannot connect" in out.lower() or "refused" in out.lower()
