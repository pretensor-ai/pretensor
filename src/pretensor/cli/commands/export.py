"""``pretensor export`` command."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from pretensor.cli import constants as cli_constants
from pretensor.cli.config_file import get_cli_config, resolve_path_option
from pretensor.core.portable_export import export_graph_payload
from pretensor.core.registry import DatabaseNotFoundError, GraphRegistry
from pretensor.core.store import KuzuStore


def register_export_command(app: typer.Typer, *, console: Console) -> None:
    @app.command("export")
    def export_command(
        database: str = typer.Option(
            ...,
            "--database",
            "-d",
            help="Registry connection name to export (see `pretensor list`).",
        ),
        output: Path = typer.Option(
            ...,
            "--output",
            "-o",
            help="Target JSON path for portable graph export.",
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
        state_dir: Path = typer.Option(
            cli_constants.DEFAULT_STATE_DIR,
            "--state-dir",
            help="Directory containing registry.json and graph files.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        ctx: typer.Context = typer.Option(
            None,
            hidden=True,
        ),
    ) -> None:
        """Dump one indexed graph to portable pretty-printed JSON."""
        cli_config = get_cli_config(ctx)
        state_dir = resolve_path_option(
            ctx,
            param_name="state_dir",
            cli_value=state_dir,
            config_value=cli_config.state_dir,
        )
        reg_path = state_dir / cli_constants.REGISTRY_FILENAME
        if not reg_path.exists():
            console.print("[red]No registry found.[/red] Run `pretensor index` first.")
            raise typer.Exit(1)
        try:
            reg = GraphRegistry(reg_path).load()
            entry = reg.require(database)
        except DatabaseNotFoundError:
            console.print(
                f"[red]Unknown connection {database!r}.[/red] "
                "Use `pretensor list` to discover available names."
            )
            raise typer.Exit(1)
        except Exception as exc:
            console.print(
                f"[red]Cannot read registry file {reg_path}:[/red] {exc}\n"
                "The registry.json may be corrupt. Delete it and re-run `pretensor index`."
            )
            raise typer.Exit(1) from exc
        graph_path = Path(entry.unified_graph_path or entry.graph_path)
        if not graph_path.exists():
            console.print(
                f"[red]Graph file missing:[/red] {graph_path}\n"
                "Re-run `pretensor index` before exporting."
            )
            raise typer.Exit(1)
        try:
            store = KuzuStore(graph_path)
        except Exception as exc:
            console.print(
                f"[red]Cannot open graph file {graph_path}:[/red] {exc}\n"
                "The file may be corrupt. Re-run `pretensor index`."
            )
            raise typer.Exit(1) from exc
        try:
            payload = export_graph_payload(
                store,
                connection_name=entry.connection_name,
                database_name=entry.database,
                graph_path=graph_path,
            )
        finally:
            store.close()
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except PermissionError as exc:
            console.print(f"[red]Cannot write export file {output}:[/red] {exc}")
            raise typer.Exit(1) from exc
        except OSError as exc:
            console.print(f"[red]Cannot write export file {output}:[/red] {exc}")
            raise typer.Exit(1) from exc
        stats = payload["stats"]
        console.print(f"[green]Graph exported:[/green] {output}")
        console.print(
            "[dim]"
            f"node_types={stats['node_types']} edge_types={stats['edge_types']} "
            f"nodes={stats['node_count']} edges={stats['edge_count']}"
            "[/dim]"
        )
