"""Shared per-command execution logic reused across CLI commands.

``run_index`` and ``run_analyze_one`` are the heavy-lifting cores of
``pretensor index`` and ``pretensor analyze`` respectively. They live here
(rather than in ``index.py`` / ``analyze.py``) because ``quickstart.py`` and
``init.py`` also need to invoke them directly, without going through the
Typer command layer.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pretensor.cli import constants as cli_constants
from pretensor.cli.dbt_enrichment import (
    apply_dbt_enrichment_cli,
    preload_dbt_manifest,
)
from pretensor.cli.paths import (
    graph_file_for_connection,
    keystore_path,
    unified_graph_path,
)
from pretensor.config import EmbeddingsConfig, PretensorConfig
from pretensor.connectors.inspect import inspect
from pretensor.core.builder import GraphBuilder
from pretensor.core.dsn_crypto import DSNEncryptor
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.enrichment.analyze.pipeline import (
    EmptyGraphError,
    run_analyze_enrichment,
)
from pretensor.enrichment.analyze.summary import AnalyzeSummary
from pretensor.introspection.models.dsn import registry_dialect_for
from pretensor.mcp.service_registry import (
    load_registry,
    open_store_for_entry,
    release_store,
    resolve_registry_entry,
)
from pretensor.observability import log_timed_operation
from pretensor.skills.generator import SkillGenerator
from pretensor.staleness.snapshot_store import SnapshotStore
from pretensor.visibility.config import load_visibility_config, merge_profile_into_base
from pretensor.visibility.filter import VisibilityFilter

_EXIT_ERROR = 1
_MAX_DISPLAY_ROWS = 20

_PROFILE_INDEX = os.environ.get("PRETENSOR_PROFILE_INDEX", "").lower() not in (
    "",
    "0",
    "false",
    "no",
)
logger = logging.getLogger(__name__)


def run_index(
    *,
    console: Console,
    cli_config: object,
    dsn: str,
    connection_name: str,
    config: object,
    state_dir: Path,
    unified: bool,
    embeddings: bool,
    skills_target: str,
    visibility_file: Path | None,
    profile: str | None,
    dbt_manifest: Path | None,
    dbt_sources: Path | None,
) -> None:
    """Core indexing logic shared by DSN, --source, and --all paths."""
    from pretensor.cli.config_file import PretensorCliConfig
    from pretensor.introspection.models.config import ConnectionConfig as _CC

    assert isinstance(config, _CC)
    assert isinstance(cli_config, PretensorCliConfig)

    total_started = time.perf_counter()
    sd = Path(state_dir)
    vis_path = (
        Path(visibility_file) if visibility_file is not None else sd / "visibility.yml"
    )
    try:
        vis_cfg = merge_profile_into_base(load_visibility_config(vis_path), profile)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    visibility_filter = (
        VisibilityFilter.from_config(vis_cfg)
        if vis_cfg.hidden_schemas
        or vis_cfg.hidden_tables
        or vis_cfg.hidden_columns
        or vis_cfg.allowed_schemas
        or vis_cfg.allowed_tables
        else None
    )
    preloaded_manifest = None
    preloaded_sources: Path | None = None
    if dbt_manifest is not None:
        preloaded_manifest, preloaded_sources = preload_dbt_manifest(
            dbt_manifest, dbt_sources, console=console
        )

    try:
        sd.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        console.print(
            f"[red]Cannot create state directory {sd}:[/red] {e}\n"
            "Check write permissions or use --state-dir to specify a writable path."
        )
        raise typer.Exit(1) from e

    console.print(f"[bold blue]Introspecting[/bold blue] '{connection_name}'...")
    _t_inspect = time.perf_counter() if _PROFILE_INDEX else 0.0
    try:
        with log_timed_operation(
            logger,
            event="index.inspect",
            connection_name=connection_name,
            dialect=str(config.type),
        ):
            snapshot = inspect(config)
    except Exception as e:
        console.print(
            f"[red]Cannot connect to database:[/red] {e}\n"
            "Check that the DSN is correct and the database is reachable."
        )
        raise typer.Exit(1) from e
    if _PROFILE_INDEX:
        console.print(
            f"[dim][profile] inspect: {time.perf_counter() - _t_inspect:.2f}s[/dim]"
        )

    graph_path = (
        unified_graph_path(sd)
        if unified
        else graph_file_for_connection(sd, connection_name)
    )
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    if unified:
        if graph_path.exists():
            try:
                store = KuzuStore(graph_path)
            except Exception as e:
                console.print(
                    f"[red]Cannot open graph file {graph_path}:[/red] {e}\n"
                    "The file may be corrupt. Delete it and re-run `pretensor index`."
                )
                raise typer.Exit(1) from e
            try:
                store.ensure_schema()
                store.clear_connection_subgraph(connection_name)
            finally:
                store.close()
    else:
        if graph_path.exists():
            try:
                graph_path.unlink()
            except PermissionError as e:
                console.print(
                    f"[red]Cannot remove existing graph file {graph_path}:[/red] {e}"
                )
                raise typer.Exit(1) from e

    try:
        store = KuzuStore(graph_path)
    except Exception as e:
        console.print(
            f"[red]Cannot create graph file {graph_path}:[/red] {e}\n"
            "Check disk space and permissions."
        )
        raise typer.Exit(1) from e
    try:
        _t_build = time.perf_counter() if _PROFILE_INDEX else 0.0
        with log_timed_operation(
            logger,
            event="index.graph_build",
            connection_name=connection_name,
            replace_mode="connection" if unified else "full",
        ):
            run_config = PretensorConfig(
                graph=cli_config.graph,
                embeddings=EmbeddingsConfig(index_tables=embeddings),
            )
            GraphBuilder().build(
                snapshot,
                store,
                replace_mode="connection" if unified else "full",
                visibility_filter=visibility_filter,
                config=run_config,
            )
        if _PROFILE_INDEX:
            console.print(
                f"[dim][profile] build (incl. intelligence layer): {time.perf_counter() - _t_build:.2f}s[/dim]"
            )
        if preloaded_manifest is not None:
            apply_dbt_enrichment_cli(
                manifest=preloaded_manifest,
                sources_path=preloaded_sources,
                store=store,
                connection_name=connection_name,
                console=console,
            )
        tc_rows = store.query_all_rows(
            """
            MATCH (t:SchemaTable {connection_name: $cn})
            RETURN count(*)
            """,
            {"cn": connection_name},
        )
        table_count = int(tc_rows[0][0]) if tc_rows and tc_rows[0][0] is not None else 0
    finally:
        store.close()

    try:
        reg = GraphRegistry(sd / cli_constants.REGISTRY_FILENAME).load()
    except Exception as e:
        console.print(
            f"[red]Cannot read registry file:[/red] {e}\n"
            "The registry.json may be corrupt. Delete it and re-run `pretensor index`."
        )
        raise typer.Exit(1) from e
    indexed_at = datetime.now(timezone.utc)
    # Encrypt the DSN at rest by default. ``DSNEncryptor`` auto-creates the
    # keystore (0o600) on first use, so a fresh state dir no longer stores the
    # plaintext password in registry.json.
    enc = DSNEncryptor(keystore_path(sd))
    reg.upsert(
        connection_name=connection_name,
        database=snapshot.database,
        dsn=dsn,
        graph_path=graph_path,
        unified_graph_path=graph_path if unified else None,
        indexed_at=indexed_at,
        encrypt_dsn=True,
        encryptor=enc,
        dialect=registry_dialect_for(config.type),
        table_count=table_count,
        dbt_manifest_path=str(dbt_manifest) if dbt_manifest is not None else None,
    )
    try:
        reg.save()
    except PermissionError as e:
        console.print(
            f"[red]Cannot write registry file:[/red] {e}\n"
            "Check write permissions on the state directory."
        )
        raise typer.Exit(1) from e

    snap_store = SnapshotStore(sd)
    snap_store.save(connection_name, snapshot)

    entry = reg.get(connection_name)
    if entry is not None:
        skill_store = KuzuStore(graph_path)
        try:
            skill_paths = SkillGenerator.write_for_index(
                store=skill_store,
                entry=entry,
                skills_target=skills_target,
            )
        finally:
            skill_store.close()
        for p in skill_paths:
            console.print(f"[green]Skill written:[/green] {p}")

    console.print(f"[green]Graph written:[/green] {graph_path}")
    console.print(f"[green]Registry updated:[/green] {reg.path}")
    logger.info(
        "index completed in %.2fms",
        (time.perf_counter() - total_started) * 1000,
        extra={
            "event": "index.total",
            "status": "ok",
            "duration_ms": (time.perf_counter() - total_started) * 1000,
            "connection_name": connection_name,
            "unified": unified,
            "table_count": table_count,
        },
    )


def run_analyze_one(
    *,
    console: Console,
    repo_path: Path,
    connection: str,
    service: str | None,
    state_dir: Path,
    graph_dir: Path | None,
    include: list[str],
    exclude: list[str],
    max_file_bytes: int,
    min_confidence: float,
    default_schema: str | None,
    dry_run: bool,
    as_json: bool,
    json_sink: list[dict] | None = None,
) -> int:
    """Analyze one repository. Returns a process exit code."""
    service_name = service or repo_path.name

    reg = load_registry(state_dir)
    entry = resolve_registry_entry(reg, connection)
    if entry is None:
        _empty_graph_error(console, connection)
        return _EXIT_ERROR

    store = open_store_for_entry(entry)
    try:
        summary = run_analyze_enrichment(
            repo_path,
            store,
            entry.connection_name,
            service_name=service_name,
            includes=include,
            excludes=exclude,
            max_file_bytes=max_file_bytes,
            min_confidence=min_confidence,
            default_schema=default_schema,
            dialect=entry.dialect,
            database=entry.database,
            dry_run=dry_run,
        )
    except EmptyGraphError:
        _empty_graph_error(console, entry.connection_name)
        return _EXIT_ERROR
    finally:
        release_store(store)

    if as_json:
        payload = dataclasses.asdict(summary)
        if json_sink is not None:
            json_sink.append(payload)
        else:
            typer.echo(json.dumps(payload, indent=2))
        return 0

    _render_summary(console, summary, service_name=service_name, dry_run=dry_run)
    return 0


def _empty_graph_error(console: Console, connection: str) -> None:
    """Print the hard-error hint; no auto-index is performed (by design)."""
    console.print(
        f"[red]The graph for connection '{connection}' is empty.[/red] Run:\n"
        f"  pretensor index --connection {connection}\n"
        "first, then re-run pretensor analyze."
    )


def _render_summary(
    console: Console,
    summary: AnalyzeSummary,
    *,
    service_name: str,
    dry_run: bool,
) -> None:
    rows = sorted(summary.rows, key=lambda r: r.confidence, reverse=True)
    shown = rows[:_MAX_DISPLAY_ROWS]

    title = (
        "pretensor analyze (dry run — no writes)" if dry_run else "pretensor analyze"
    )
    table = Table(title=title)
    table.add_column("service")
    table.add_column("table")
    table.add_column("op")
    table.add_column("file:line")
    table.add_column("confidence", justify="right")
    for r in shown:
        line = (
            f"{r.file_path}:{r.line_start}"
            if r.line_start == r.line_end
            else f"{r.file_path}:{r.line_start}-{r.line_end}"
        )
        table.add_row(r.service_name, r.table, r.op, line, f"{r.confidence:.2f}")
    console.print(table)

    if len(rows) > len(shown):
        console.print(f"[dim]… showing top {len(shown)} of {len(rows)} edges[/dim]")

    verb = "would write" if dry_run else "wrote"
    console.print(
        f"Scanned {summary.files_scanned} files, "
        f"found {summary.sql_candidates_found} SQL candidates, "
        f"{verb} {summary.consumers_written} consumers / {summary.edges_written} edges "
        f"({summary.cross_connection_dropped} dropped: cross-connection). "
        f"Unqualified refs resolved against schema '{summary.default_schema}'."
    )
