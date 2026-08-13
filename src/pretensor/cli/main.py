"""Typer entrypoint for ``pretensor``."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import typer
from rich.console import Console

from pretensor.benchmark import register_benchmark_command
from pretensor.cli.commands.analyze import register_analyze_command
from pretensor.cli.commands.connections import register_connection_commands
from pretensor.cli.commands.export import register_export_command
from pretensor.cli.commands.index import register_index_command
from pretensor.cli.commands.list import register_list_command
from pretensor.cli.commands.quickstart import register_quickstart_command
from pretensor.cli.commands.reindex import register_reindex_command
from pretensor.cli.commands.semantic import register_semantic_commands
from pretensor.cli.commands.serve import register_serve_command
from pretensor.cli.commands.sync_grants import register_sync_grants_command
from pretensor.cli.commands.validate import register_validate_command
from pretensor.cli.config_file import CliConfigError, load_cli_config
from pretensor.cli.plugin import discover_cli_plugins
from pretensor.observability import LogFormat, LogLevel, configure_logging

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if not value:
        return
    try:
        ver = _pkg_version("pretensor")
    except PackageNotFoundError:
        ver = "unknown"
    typer.echo(f"pretensor {ver}")
    raise typer.Exit()


@app.callback()
def configure_root_logging(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the pretensor version and exit.",
    ),
    log_level: LogLevel = typer.Option(
        "warning",
        "--log-level",
        help="Logging level: debug, info, warning, error.",
    ),
    log_format: LogFormat = typer.Option(
        "text",
        "--log-format",
        help="Log output format: text or json.",
    ),
    log_file: Path | None = typer.Option(
        None,
        "--log-file",
        help="Optional file path for duplicate log output.",
        dir_okay=False,
        writable=True,
        resolve_path=True,
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional path to pretensor CLI YAML config.",
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Configure process-wide logging for all CLI subcommands."""
    configure_logging(level=log_level, log_format=log_format, log_file=log_file)
    try:
        cli_config = load_cli_config(config)
    except CliConfigError as exc:
        Console().print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    ctx.obj = {"config": cli_config}


register_serve_command(app)
register_sync_grants_command(app)
console = Console()

register_connection_commands(app)

register_benchmark_command(app)
register_index_command(app, console=console)
register_analyze_command(app, console=console)
register_list_command(app, console=console)
register_quickstart_command(app, console=console)
register_reindex_command(app, console=console)
register_export_command(app, console=console)
register_semantic_commands(app, console=console)
register_validate_command(app, console=console)

discover_cli_plugins(app)
