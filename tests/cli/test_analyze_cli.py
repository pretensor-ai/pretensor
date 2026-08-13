"""CLI tests for ``pretensor analyze``."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from pretensor.cli.main import app
from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _ANSI_ESCAPE_RE.sub("", text)).strip()


def _setup_graph(
    tmp_path: Path, *, connection: str = "demo", with_tables: bool = True
) -> None:
    """Build a graph + registry entry for ``connection`` under ``tmp_path``."""
    tables = (
        [
            Table(
                name="orders",
                schema_name="public",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            ),
            Table(
                name="users",
                schema_name="public",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            ),
        ]
        if with_tables
        else []
    )
    snap = SchemaSnapshot(
        connection_name=connection,
        database=connection,
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / f"{connection}.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name=connection,
        database=connection,
        dsn=f"postgresql://localhost/{connection}",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()


def _make_repo(root: Path) -> Path:
    """Create a tiny repo with one SQL-bearing Python file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "app.py").write_text(
        'query = "SELECT id FROM public.orders WHERE id = 1"\n',
        encoding="utf-8",
    )
    return root


def _consumer_count(tmp_path: Path, connection: str = "demo") -> int:
    store = KuzuStore(tmp_path / "graphs" / f"{connection}.kuzu")
    try:
        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN count(*)")
        return int(rows[0][0])
    finally:
        store.close()


def test_analyze_help() -> None:
    result = CliRunner().invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    for flag in (
        "--connection",
        "--service",
        "--include",
        "--exclude",
        "--max-file-bytes",
        "--min-confidence",
        "--default-schema",
        "--dry-run",
        "--json",
    ):
        assert flag in plain, flag


def test_analyze_writes_consumers(tmp_path: Path) -> None:
    _setup_graph(tmp_path)
    repo = _make_repo(tmp_path / "svc")
    result = CliRunner().invoke(
        app,
        ["analyze", str(repo), "--connection", "demo", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert "orders" in result.stdout
    assert _consumer_count(tmp_path) == 1


def test_analyze_dry_run_writes_nothing(tmp_path: Path) -> None:
    _setup_graph(tmp_path)
    repo = _make_repo(tmp_path / "svc")
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            str(repo),
            "--connection",
            "demo",
            "--state-dir",
            str(tmp_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert _consumer_count(tmp_path) == 0  # nothing persisted


def test_analyze_json_output(tmp_path: Path) -> None:
    _setup_graph(tmp_path)
    repo = _make_repo(tmp_path / "svc")
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            str(repo),
            "--connection",
            "demo",
            "--state-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    for key in (
        "files_scanned",
        "consumers_written",
        "edges_written",
        "cross_connection_dropped",
        "scan_run_id",
    ):
        assert key in payload
    assert payload["consumers_written"] == 1


def test_analyze_empty_graph_hard_errors(tmp_path: Path) -> None:
    _setup_graph(tmp_path, connection="empty", with_tables=False)
    repo = _make_repo(tmp_path / "svc")
    result = CliRunner().invoke(
        app,
        ["analyze", str(repo), "--connection", "empty", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "pretensor index --connection" in _normalize(result.stdout)


def test_analyze_unknown_connection_hard_errors(tmp_path: Path) -> None:
    _setup_graph(tmp_path)
    repo = _make_repo(tmp_path / "svc")
    result = CliRunner().invoke(
        app,
        ["analyze", str(repo), "--connection", "nope", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "pretensor index --connection" in _normalize(result.stdout)


def test_analyze_service_flag_overrides_default(tmp_path: Path) -> None:
    _setup_graph(tmp_path)
    repo = _make_repo(tmp_path / "svc")
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            str(repo),
            "--connection",
            "demo",
            "--service",
            "custom_name",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    store = KuzuStore(tmp_path / "graphs" / "demo.kuzu")
    try:
        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer) RETURN DISTINCT c.service_name"
        )
    finally:
        store.close()
    assert [r[0] for r in rows] == ["custom_name"]


def test_analyze_include_exclude_narrows_scan(tmp_path: Path) -> None:
    _setup_graph(tmp_path)
    repo = tmp_path / "svc"
    repo.mkdir()
    (repo / "keep.py").write_text(
        'query = "SELECT id FROM public.orders"\n', encoding="utf-8"
    )
    (repo / "skip.py").write_text(
        'query = "SELECT id FROM public.users"\n', encoding="utf-8"
    )
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            str(repo),
            "--connection",
            "demo",
            "--include",
            "keep.py",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    store = KuzuStore(tmp_path / "graphs" / "demo.kuzu")
    try:
        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN c.file_path")
    finally:
        store.close()
    files = {r[0] for r in rows}
    assert files == {"keep.py"}
