"""Typer entrypoint for ``pretensor``."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

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

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def configure_root_logging(
    ctx: typer.Context,
    log_level: LogLevel = typer.Option(
        "info",
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

register_index_command(app, console=console)
register_list_command(app, console=console)
register_quickstart_command(app, console=console)
register_reindex_command(app, console=console)
register_export_command(app, console=console)
register_semantic_commands(app, console=console)
register_validate_command(app, console=console)

discover_cli_plugins(app)
