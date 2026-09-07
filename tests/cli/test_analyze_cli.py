"""CLI tests for ``pretensor analyze``."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

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
    tmp_path: Path,
    *,
    connection: str = "demo",
    with_tables: bool = True,
    schema: str = "public",
    extra_schema: str | None = None,
    dialect: Literal["postgres", "mysql", "snowflake", "bigquery"] = "postgres",
    database: str | None = None,
) -> None:
    """Build a graph + registry entry for ``connection`` under ``tmp_path``."""
    tables = (
        [
            Table(
                name="orders",
                schema_name=schema,
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            ),
            Table(
                name="users",
                schema_name=schema,
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            ),
        ]
        if with_tables
        else []
    )
    if extra_schema is not None:
        tables.append(
            Table(
                name="logs",
                schema_name=extra_schema,
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
                foreign_keys=[],
            )
        )
    snap = SchemaSnapshot(
        connection_name=connection,
        database=database or connection,
        schemas=[schema] + ([extra_schema] if extra_schema is not None else []),
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
        database=database or connection,
        dsn=f"postgresql://localhost/{connection}",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
        dialect=dialect,
    )
    reg.save()


def _make_repo(
    root: Path, *, sql: str = "SELECT id FROM public.orders WHERE id = 1"
) -> Path:
    """Create a tiny repo with one SQL-bearing Python file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "app.py").write_text(f'query = "{sql}"\n', encoding="utf-8")
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


def test_analyze_help_notes_path_and_service_ignored_with_all() -> None:
    result = CliRunner().invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert plain.count("Ignored with --all.") == 2


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
        "default_schema",
    ):
        assert key in payload
    assert payload["consumers_written"] == 1
    assert payload["default_schema"] == "public"


def test_analyze_derives_default_schema_from_connection(tmp_path: Path) -> None:
    """Unqualified refs resolve without --default-schema on a non-public graph."""
    _setup_graph(tmp_path, schema="analytics")
    repo = _make_repo(tmp_path / "svc", sql="SELECT id FROM orders WHERE id = 1")
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
    assert payload["default_schema"] == "analytics"
    assert payload["consumers_written"] == 1
    assert payload["cross_connection_dropped"] == 0
    assert _consumer_count(tmp_path) == 1


def test_analyze_explicit_default_schema_overrides_derived(tmp_path: Path) -> None:
    _setup_graph(tmp_path, schema="analytics")
    repo = _make_repo(tmp_path / "svc", sql="SELECT id FROM orders WHERE id = 1")
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            str(repo),
            "--connection",
            "demo",
            "--state-dir",
            str(tmp_path),
            "--default-schema",
            "public",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["default_schema"] == "public"
    # "public.orders" does not exist in the analytics-only graph → dropped.
    assert payload["consumers_written"] == 0
    assert payload["cross_connection_dropped"] == 1
    assert _consumer_count(tmp_path) == 0


def test_analyze_text_summary_reports_resolved_schema(tmp_path: Path) -> None:
    """The non-JSON summary names the schema unqualified refs resolved against."""
    _setup_graph(tmp_path, schema="analytics")
    repo = _make_repo(tmp_path / "svc", sql="SELECT id FROM orders WHERE id = 1")
    result = CliRunner().invoke(
        app,
        ["analyze", str(repo), "--connection", "demo", "--state-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    plain = _normalize(result.stdout)
    assert "Unqualified refs resolved against schema 'analytics'." in plain


def test_analyze_mysql_dialect_fallback_through_registry(tmp_path: Path) -> None:
    """Multi-schema graph + mysql registry entry derives the database name."""
    _setup_graph(
        tmp_path,
        schema="appdb",
        extra_schema="otherdb",
        dialect="mysql",
        database="appdb",
    )
    repo = _make_repo(tmp_path / "svc", sql="SELECT id FROM orders WHERE id = 1")
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
    assert payload["default_schema"] == "appdb"
    assert payload["consumers_written"] == 1
    assert _consumer_count(tmp_path) == 1


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


def test_analyze_all_help_lists_flag() -> None:
    result = CliRunner().invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--all" in _normalize(result.stdout)


def test_analyze_all_errors_when_no_repositories(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text("state_dir: .\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["--config", str(cfg_dir / "config.yaml"), "analyze", "--all"]
    )
    assert result.exit_code == 1
    assert "no repositories" in _normalize(result.stdout).lower()


def test_analyze_requires_connection_without_all() -> None:
    result = CliRunner().invoke(app, ["analyze", "."])
    assert result.exit_code != 0
    assert "--connection" in _normalize(result.stdout)


def test_analyze_all_runs_each_repo_and_aggregates_worst_code(
    monkeypatch, tmp_path: Path
) -> None:
    """`--all` must call `run_analyze_one` per repo and return the worst code."""
    import pretensor.cli.commands.analyze as analyze_module

    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "repositories:\n"
        f"  - path: {repo_a}\n"
        "    connection: conn-a\n"
        f"  - path: {repo_b}\n"
        "    service: custom-b\n"
        "    connection: conn-b\n",
        encoding="utf-8",
    )

    calls: list[tuple[Path, str, str]] = []

    def fake_analyze_one(
        *, repo_path: Path, connection: str, service: str, **kwargs: object
    ) -> int:
        calls.append((repo_path, connection, service))
        return 0 if connection == "conn-a" else 1

    monkeypatch.setattr(analyze_module, "run_analyze_one", fake_analyze_one)

    result = CliRunner().invoke(
        app, ["--config", str(cfg_dir / "config.yaml"), "analyze", "--all"]
    )

    assert result.exit_code == 1, result.stdout
    assert len(calls) == 2
    assert (repo_a.resolve(), "conn-a", "repo-a") in calls
    assert (repo_b.resolve(), "conn-b", "custom-b") in calls


def _write_two_repo_config(cfg_dir: Path, repo_a: Path, repo_b: Path) -> Path:
    cfg_dir.mkdir()
    repo_a.mkdir()
    repo_b.mkdir()
    config_path = cfg_dir / "config.yaml"
    config_path.write_text(
        "repositories:\n"
        f"  - path: {repo_a}\n"
        "    connection: conn-a\n"
        f"  - path: {repo_b}\n"
        "    connection: conn-b\n",
        encoding="utf-8",
    )
    return config_path


def test_analyze_all_json_emits_single_array(monkeypatch, tmp_path: Path) -> None:
    """`--all --json` emits one parseable JSON array; no narration text leaks in."""
    import pretensor.cli.commands.analyze as analyze_module

    config_path = _write_two_repo_config(
        tmp_path / ".pretensor", tmp_path / "repo-a", tmp_path / "repo-b"
    )

    def fake_run_analyze_one(
        *,
        connection: str,
        as_json: bool,
        json_sink: list[dict] | None = None,
        **kwargs: object,
    ) -> int:
        if as_json and json_sink is not None:
            json_sink.append({"connection": connection, "files_scanned": 1})
        return 0

    monkeypatch.setattr(analyze_module, "run_analyze_one", fake_run_analyze_one)

    result = CliRunner().invoke(
        app, ["--config", str(config_path), "analyze", "--all", "--json"]
    )

    assert result.exit_code == 0, result.stdout
    assert "Analyzing" not in result.stdout
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert {p["connection"] for p in payload} == {"conn-a", "conn-b"}


def test_analyze_all_with_explicit_service_warns(monkeypatch, tmp_path: Path) -> None:
    import pretensor.cli.commands.analyze as analyze_module

    config_path = _write_two_repo_config(
        tmp_path / ".pretensor", tmp_path / "repo-a", tmp_path / "repo-b"
    )

    monkeypatch.setattr(
        analyze_module,
        "run_analyze_one",
        lambda **kwargs: 0,
    )

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "analyze",
            "--service",
            "custom",
            "--all",
        ],
    )

    assert result.exit_code == 0, result.stdout
    plain = _normalize(result.stdout)
    assert "--service" in plain
    assert "Ignoring" in plain


def test_analyze_all_without_explicit_service_does_not_warn(
    monkeypatch, tmp_path: Path
) -> None:
    import pretensor.cli.commands.analyze as analyze_module

    config_path = _write_two_repo_config(
        tmp_path / ".pretensor", tmp_path / "repo-a", tmp_path / "repo-b"
    )

    monkeypatch.setattr(
        analyze_module,
        "run_analyze_one",
        lambda **kwargs: 0,
    )

    result = CliRunner().invoke(app, ["--config", str(config_path), "analyze", "--all"])

    assert result.exit_code == 0, result.stdout
    assert "Ignoring" not in _normalize(result.stdout)


def test_analyze_all_json_with_explicit_service_stays_parseable(
    monkeypatch, tmp_path: Path
) -> None:
    """The ignored-arg warning must not land on stdout when --json is set.

    Regression test: --all --json --service used to print the yellow
    "Ignoring ..." warning to stdout *before* the JSON array, which broke
    json.loads() for any scripted caller.
    """
    import pretensor.cli.commands.analyze as analyze_module

    config_path = _write_two_repo_config(
        tmp_path / ".pretensor", tmp_path / "repo-a", tmp_path / "repo-b"
    )

    def fake_run_analyze_one(
        *, as_json: bool, json_sink: list[dict] | None = None, **kwargs: object
    ) -> int:
        if as_json and json_sink is not None:
            json_sink.append({"files_scanned": 1})
        return 0

    monkeypatch.setattr(analyze_module, "run_analyze_one", fake_run_analyze_one)

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "analyze",
            "--service",
            "custom",
            "--all",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    # stdout must be pure JSON: no warning text, and it must parse.
    assert "Ignoring" not in result.stdout
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert len(payload) == 2
    # The warning was not dropped, only relocated: it's on stderr instead.
    assert "Ignoring" in result.stderr
    assert "--service" in result.stderr


def test_analyze_all_json_with_explicit_path_warns_on_stderr(
    monkeypatch, tmp_path: Path
) -> None:
    """An explicit positional path with --all --json also warns, via stderr."""
    import pretensor.cli.commands.analyze as analyze_module

    config_path = _write_two_repo_config(
        tmp_path / ".pretensor", tmp_path / "repo-a", tmp_path / "repo-b"
    )
    explicit_path = tmp_path / "somewhere-else"
    explicit_path.mkdir()

    def fake_run_analyze_one(
        *, as_json: bool, json_sink: list[dict] | None = None, **kwargs: object
    ) -> int:
        if as_json and json_sink is not None:
            json_sink.append({"files_scanned": 1})
        return 0

    monkeypatch.setattr(analyze_module, "run_analyze_one", fake_run_analyze_one)

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "analyze",
            str(explicit_path),
            "--all",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Ignoring" not in result.stdout
    json.loads(result.stdout)  # still parses despite the explicit path
    assert "Ignoring" in result.stderr
    assert "path" in result.stderr


def test_analyze_all_json_includes_real_summary_fields(tmp_path: Path) -> None:
    """`--all --json`, run against a real repo/graph, carries genuine summary data.

    The plumbing tests above monkeypatch `run_analyze_one` entirely, so the
    real `dataclasses.asdict(summary)` -> `json_sink.append(...)` branch
    never executes under them. This test runs the real per-repo path (same
    graph/registry fixtures as the single-repo --json tests) for one
    configured repository and checks the parsed array element carries real
    `AnalyzeSummary` fields.
    """
    _setup_graph(tmp_path)
    repo = _make_repo(tmp_path / "repo-a")
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        f"repositories:\n  - path: {repo}\n    connection: demo\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(cfg_dir / "config.yaml"),
            "analyze",
            "--all",
            "--json",
            "--state-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert len(payload) == 1
    summary = payload[0]
    for key in (
        "files_scanned",
        "consumers_written",
        "edges_written",
        "cross_connection_dropped",
        "scan_run_id",
        "default_schema",
    ):
        assert key in summary
    assert summary["consumers_written"] == 1
    assert summary["default_schema"] == "public"
