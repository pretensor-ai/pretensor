"""``pretensor list`` command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from pretensor.cli import constants as cli_constants
from pretensor.cli.config_file import get_cli_config, resolve_path_option
from pretensor.core.registry import GraphRegistry


def register_list_command(app: typer.Typer, *, console: Console) -> None:
    @app.command("list")
    def list_command(
        state_dir: Path = typer.Option(
            cli_constants.DEFAULT_STATE_DIR,
            "--state-dir",
            help="Directory containing registry.json.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        ctx: typer.Context = typer.Option(
            None,
            hidden=True,
        ),
    ) -> None:
        """List indexed connections from the registry."""
        cli_config = get_cli_config(ctx)
        state_dir = resolve_path_option(
            ctx,
            param_name="state_dir",
            cli_value=state_dir,
            config_value=cli_config.state_dir,
        )
        path = state_dir / cli_constants.REGISTRY_FILENAME
        if not path.exists():
            console.print("[dim]No registry found.[/dim]")
            raise typer.Exit(0)

        try:
            reg = GraphRegistry(path).load()
        except Exception as e:
            console.print(
                f"[red]Cannot read registry file {path}:[/red] {e}\n"
                "The registry.json may be corrupt. Delete it and re-run `pretensor index`."
            )
            raise typer.Exit(1) from e
        entries = reg.list_entries()
        if not entries:
            console.print("[dim]Registry is empty.[/dim]")
            return

        for entry in entries:
            ts = entry.last_indexed_at.isoformat()
            tc = entry.table_count
            tc_s = f"  tables={tc}" if tc is not None else ""
            console.print(
                f"[bold]{entry.connection_name}[/bold]  "
                f"db={entry.database}  indexed={ts}{tc_s}\n"
                f"  graph: {entry.unified_graph_path or entry.graph_path}"
            )
