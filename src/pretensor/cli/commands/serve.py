"""``pretensor serve`` — MCP stdio server + client config snippet."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

from pretensor.cli.config_file import (
    get_cli_config,
    resolve_optional_path_option,
    resolve_optional_str_option,
    resolve_path_option,
)
from pretensor.config import PretensorConfig
from pretensor.mcp.server import print_mcp_config, run_server
from pretensor.mcp.service import mcp_config_json
from pretensor.visibility.config import default_visibility_path

console = Console()


def register_serve_command(app: typer.Typer) -> None:
    """Attach ``serve`` to the root Typer app."""

    @app.command("serve")
    def serve_command(
        graph_dir: Path = typer.Option(
            Path(".pretensor"),
            "--graph-dir",
            help="State directory (registry.json, graphs/, search index).",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        visibility_file: Path | None = typer.Option(
            None,
            "--visibility",
            help="Path to visibility.yml (default: <graph-dir>/visibility.yml).",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        profile: str | None = typer.Option(
            None,
            "--profile",
            help="Named visibility profile from visibility.yml (serve-time filter).",
        ),
        config_only: bool = typer.Option(
            False,
            "--config-only",
            help="Print mcpServers JSON to stdout and exit (for piping).",
        ),
        print_config: bool = typer.Option(
            True,
            "--print-config/--no-print-config",
            help="When starting the server, also print the mcpServers JSON (to stderr by default).",
        ),
        ctx: typer.Context = typer.Option(None, hidden=True),
    ) -> None:
        """Run the Pretensor MCP server on stdio (JSON-RPC over stdout).

        The MCP protocol uses stdout for JSON-RPC. The Claude/Cursor ``mcpServers``
        snippet is printed to **stderr** by default so it does not corrupt the
        protocol stream. Use ``--config-only`` to print the snippet to stdout and exit.
        """
        cli_config = get_cli_config(ctx)
        graph_dir = resolve_path_option(
            ctx,
            param_name="graph_dir",
            cli_value=graph_dir,
            config_value=cli_config.state_dir,
        ).resolve()
        visibility_file = resolve_optional_path_option(
            ctx,
            param_name="visibility_file",
            cli_value=visibility_file,
            config_value=(
                cli_config.visibility.path
                if cli_config.visibility.path is not None
                else default_visibility_path(graph_dir)
            ),
        )
        profile = resolve_optional_str_option(
            ctx,
            param_name="profile",
            cli_value=profile,
            config_value=cli_config.visibility.profile,
        )
        vis_path = Path(visibility_file) if visibility_file is not None else None
        if profile is not None and not str(profile).strip():
            profile = None
        if config_only:
            try:
                sys.stdout.write(mcp_config_json(graph_dir) + "\n")
                sys.stdout.flush()
            except Exception as e:
                console.print(f"[red]Failed to generate config JSON:[/red] {e}")
                raise typer.Exit(1) from e
            return
        if print_config:
            try:
                print_mcp_config(graph_dir, stream=sys.stderr)
            except Exception:
                pass
            Console(file=sys.stderr).print(
                "[dim]MCP JSON-RPC on stdout — stderr above is config only.[/dim]",
            )
        try:
            run_server(
                graph_dir,
                visibility_path=vis_path,
                profile=profile,
                config=PretensorConfig(graph=cli_config.graph),
            )
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e
        except PermissionError as e:
            console.print(
                f"[red]Permission denied accessing state directory:[/red] {e}\n"
                "Check read permissions on the --graph-dir path."
            )
            raise typer.Exit(1) from e
        except Exception as e:
            console.print(f"[red]MCP server error:[/red] {e}")
            raise typer.Exit(1) from e
