"""``add`` and ``remove`` cross-DB registry commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from pretensor.cli import constants as cli_constants
from pretensor.cli.config_file import (
    get_cli_config,
    resolve_optional_str_option,
    resolve_path_option,
)
from pretensor.cli.paths import default_connection_name, unified_graph_path
from pretensor.core.dsn_crypto import DSNEncryptor
from pretensor.core.registry import GraphRegistry
from pretensor.introspection.models.dsn import (
    connection_config_from_url,
    registry_dialect_for,
)


def _encryptor_for(state_dir: Path) -> DSNEncryptor:
    return DSNEncryptor(state_dir / "keystore")


def register_add_remove_commands(app: typer.Typer) -> None:
    @app.command("add")
    def add_command(
        dsn: str = typer.Argument(
            ..., help="Database DSN URL (postgresql:// or snowflake://)."
        ),
        name: str | None = typer.Option(
            None,
            "--name",
            "-n",
            help="Logical connection name (default: DB name from DSN).",
        ),
        state_dir: Path = typer.Option(
            cli_constants.DEFAULT_STATE_DIR,
            "--state-dir",
            help="State directory for registry and graphs.",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
        ctx: typer.Context = typer.Option(None, hidden=True),
    ) -> None:
        """Register a DSN (encrypted); run `pretensor index` to build the graph."""
        cli_config = get_cli_config(ctx)
        state_dir = resolve_path_option(
            ctx,
            param_name="state_dir",
            cli_value=state_dir,
            config_value=cli_config.state_dir,
        )
        name = resolve_optional_str_option(
            ctx,
            param_name="name",
            cli_value=name,
            config_value=cli_config.connection_defaults.name,
        )
        console = Console()
        connection_name = name or default_connection_name(dsn)
        enc = _encryptor_for(state_dir)
        reg_path = state_dir / cli_constants.REGISTRY_FILENAME
        reg = GraphRegistry(reg_path).load()
        graph_path = unified_graph_path(state_dir)
        cfg = connection_config_from_url(dsn, connection_name)
        reg.upsert(
            connection_name=connection_name,
            database=cfg.database or connection_name,
            dsn=dsn,
            graph_path=graph_path,
            unified_graph_path=graph_path,
            encrypt_dsn=True,
            encryptor=enc,
            indexed_at=None,
            dialect=registry_dialect_for(cfg.type),
        )
        reg.save()
        console.print(
            f"[green]Registered[/green] connection {connection_name!r} (DSN encrypted)."
        )
        console.print(
            f"Run: pretensor index {dsn!r} --name {connection_name} --unified"
        )

    @app.command("remove")
    def remove_command(
        name: str = typer.Argument(
            ..., help="Connection name to remove from registry."
        ),
        state_dir: Path = typer.Option(
            cli_constants.DEFAULT_STATE_DIR,
            "--state-dir",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        ctx: typer.Context = typer.Option(None, hidden=True),
    ) -> None:
        """Remove a connection from the registry (does not delete the unified graph file)."""
        cli_config = get_cli_config(ctx)
        state_dir = resolve_path_option(
            ctx,
            param_name="state_dir",
            cli_value=state_dir,
            config_value=cli_config.state_dir,
        )
        console = Console()
        reg_path = state_dir / cli_constants.REGISTRY_FILENAME
        reg = GraphRegistry(reg_path).load()
        if reg.get(name) is None:
            console.print(f"[red]Unknown connection[/red] {name!r}")
            raise typer.Exit(1)
        reg.remove(name)
        reg.save()
        console.print(f"[green]Removed[/green] {name!r} from registry.")
