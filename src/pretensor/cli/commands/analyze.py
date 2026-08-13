"""``pretensor analyze`` — scan a repo for SQL and link code to graph tables."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pretensor.cli import constants as cli_constants
from pretensor.cli.config_file import get_cli_config, resolve_aliased_path_option
from pretensor.enrichment.analyze.pipeline import (
    EmptyGraphError,
    run_analyze_enrichment,
)
from pretensor.enrichment.analyze.summary import AnalyzeSummary
from pretensor.mcp.service_registry import (
    load_registry,
    open_store_for_entry,
    release_store,
    resolve_registry_entry,
)

__all__ = ["register_analyze_command"]

_EXIT_ERROR = 1
_MAX_DISPLAY_ROWS = 20


def register_analyze_command(app: typer.Typer, *, console: Console) -> None:
    """Register ``pretensor analyze`` onto ``app``."""

    @app.command("analyze")
    def analyze_command(
        path: Path = typer.Argument(
            Path("."),
            help="Repository root to scan (default: current directory).",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        connection: str = typer.Option(
            ...,
            "--connection",
            help="Required. Connection whose tables SQL refs resolve against.",
        ),
        service: str | None = typer.Option(
            None,
            "--service",
            help="Consumer-service label (default: scanned directory basename).",
        ),
        include: list[str] = typer.Option(
            [],
            "--include",
            help="Gitignore-style glob to include (repeatable).",
        ),
        exclude: list[str] = typer.Option(
            [],
            "--exclude",
            help="Gitignore-style glob to exclude (repeatable).",
        ),
        max_file_bytes: int = typer.Option(
            1_000_000,
            "--max-file-bytes",
            help="Skip files larger than this many bytes.",
        ),
        min_confidence: float = typer.Option(
            0.6,
            "--min-confidence",
            help="Minimum confidence to emit a CONSUMES edge.",
        ),
        default_schema: str = typer.Option(
            "public",
            "--default-schema",
            help="Schema assumed for unqualified table names in scanned SQL.",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Parse and print the plan only; make no graph writes.",
        ),
        as_json: bool = typer.Option(
            False,
            "--json",
            help="Emit the machine-readable summary as JSON (suppresses the table).",
        ),
        state_dir: Path = typer.Option(
            cli_constants.DEFAULT_STATE_DIR,
            "--state-dir",
            help="Directory containing registry.json.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        graph_dir: Path | None = typer.Option(
            None,
            "--graph-dir",
            hidden=True,
            help="Deprecated alias for --state-dir.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        ctx: typer.Context = typer.Option(None, hidden=True),
    ) -> None:
        """Scan a repository for SQL string literals and link them to graph tables."""
        cli_config = get_cli_config(ctx)
        state_dir = resolve_aliased_path_option(
            ctx,
            primary_param="state_dir",
            primary_value=state_dir,
            alias_param="graph_dir",
            alias_value=graph_dir,
            config_value=cli_config.state_dir,
        )

        service_name = service or path.name

        reg = load_registry(state_dir)
        entry = resolve_registry_entry(reg, connection)
        if entry is None:
            _empty_graph_error(console, connection)
            raise typer.Exit(_EXIT_ERROR)

        store = open_store_for_entry(entry)
        try:
            summary = run_analyze_enrichment(
                path,
                store,
                entry.connection_name,
                service_name=service_name,
                includes=include,
                excludes=exclude,
                max_file_bytes=max_file_bytes,
                min_confidence=min_confidence,
                default_schema=default_schema,
                dry_run=dry_run,
            )
        except EmptyGraphError:
            _empty_graph_error(console, entry.connection_name)
            raise typer.Exit(_EXIT_ERROR) from None
        finally:
            release_store(store)

        if as_json:
            typer.echo(json.dumps(dataclasses.asdict(summary), indent=2))
            return

        _render_summary(console, summary, service_name=service_name, dry_run=dry_run)


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
        f"({summary.cross_connection_dropped} dropped: cross-connection)."
    )
