"""``pretensor analyze`` — scan a repo for SQL and link code to graph tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from click.core import ParameterSource
from rich.console import Console

from pretensor.cli import constants as cli_constants
from pretensor.cli.commands._command_runners import run_analyze_one
from pretensor.cli.config_file import get_cli_config, resolve_aliased_path_option

__all__ = ["register_analyze_command"]

_EXIT_ERROR = 1


def register_analyze_command(app: typer.Typer, *, console: Console) -> None:
    """Register ``pretensor analyze`` onto ``app``."""

    @app.command("analyze")
    def analyze_command(
        path: Path = typer.Argument(
            Path("."),
            help="Repository root to scan (default: current directory). "
            "Ignored with --all.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        connection: str | None = typer.Option(
            None,
            "--connection",
            help="Connection whose tables SQL refs resolve against. Required unless --all.",
        ),
        all_repos: bool = typer.Option(
            False,
            "--all",
            help="Analyze every repository listed under `repositories:` in config.",
        ),
        service: str | None = typer.Option(
            None,
            "--service",
            help="Consumer-service label (default: scanned directory basename). "
            "Ignored with --all.",
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
        default_schema: str | None = typer.Option(
            None,
            "--default-schema",
            help=(
                "Schema assumed for unqualified table names in scanned SQL. "
                "Default: derived from the connection (indexed schemas and "
                "dialect), falling back to 'public'."
            ),
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

        if all_repos:
            _warn_ignored_with_all(console, ctx, as_json=as_json)
            if not cli_config.repositories:
                console.print(
                    "[red]No repositories configured.[/red] "
                    "Run `pretensor init` to link one, or pass a path and --connection."
                )
                raise typer.Exit(_EXIT_ERROR)
            worst = 0
            json_summaries: list[dict] | None = [] if as_json else None
            for repo in cli_config.repositories:
                if not as_json:
                    console.print(f"[bold blue]Analyzing[/bold blue] {repo.path}")
                code = run_analyze_one(
                    console=console,
                    repo_path=repo.path,
                    connection=repo.connection,
                    service=repo.service,
                    state_dir=state_dir,
                    graph_dir=graph_dir,
                    include=include,
                    exclude=exclude,
                    max_file_bytes=max_file_bytes,
                    min_confidence=min_confidence,
                    default_schema=default_schema,
                    dry_run=dry_run,
                    as_json=as_json,
                    json_sink=json_summaries,
                )
                worst = max(worst, code)
            if as_json:
                typer.echo(json.dumps(json_summaries, indent=2))
            raise typer.Exit(worst)

        if connection is None:
            console.print("[red]--connection is required[/red] unless you pass --all.")
            raise typer.Exit(_EXIT_ERROR)

        code = run_analyze_one(
            console=console,
            repo_path=path,
            connection=connection,
            service=service,
            state_dir=state_dir,
            graph_dir=graph_dir,
            include=include,
            exclude=exclude,
            max_file_bytes=max_file_bytes,
            min_confidence=min_confidence,
            default_schema=default_schema,
            dry_run=dry_run,
            as_json=as_json,
        )
        raise typer.Exit(code)


def _warn_ignored_with_all(
    console: Console, ctx: typer.Context | None, *, as_json: bool
) -> None:
    """Warn once if the user explicitly passed args that --all ignores.

    Routed to stderr when ``as_json`` is set: stdout is the machine-readable
    JSON channel for ``--all --json``, and interleaving a warning line there
    would break ``json.loads`` on the caller's side. Scripted callers still
    see the warning on stderr instead of it being silently dropped.
    """
    if ctx is None:
        return
    ignored: list[str] = []
    for label, param_name in (("path", "path"), ("--service", "service")):
        try:
            source = ctx.get_parameter_source(param_name)
        except Exception:
            source = None
        if source is not None and source is not ParameterSource.DEFAULT:
            ignored.append(label)
    if not ignored:
        return
    message = (
        f"[yellow]Ignoring {', '.join(ignored)}: each repository under "
        "--all uses its own path/service from config.[/yellow]"
    )
    if as_json:
        Console(file=sys.stderr).print(message)
    else:
        console.print(message)
