"""``pretensor sync-grants`` — generate ``visibility.yml`` profiles from DB grants."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from pretensor.cli.config_file import (
    get_cli_config,
    resolve_optional_str_option,
    resolve_path_option,
)
from pretensor.cli.paths import default_connection_name
from pretensor.connectors.registry import get_connector
from pretensor.introspection.models.config import DatabaseType
from pretensor.introspection.models.dsn import (
    connection_config_from_url,
    infer_database_type_from_dsn,
)
from pretensor.visibility.sync_grants import run_sync_grants

console = Console()


def _parse_roles_option(raw: str | None) -> frozenset[str] | None:
    if raw is None or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    names = {p for p in parts if p}
    return frozenset(names) if names else None


def register_sync_grants_command(app: typer.Typer) -> None:
    """Attach ``sync-grants`` to the root Typer app."""

    @app.command("sync-grants")
    def sync_grants_command(
        dsn: str = typer.Option(
            ...,
            "--dsn",
            help="Admin-capable database URL (postgresql:// or snowflake://).",
        ),
        name: str | None = typer.Option(
            None,
            "--name",
            "-n",
            help=(
                "Logical connection name (defaults to the database name from the DSN). "
                "Must match the value used with ``pretensor index`` so generated "
                "``connection::schema.table`` patterns match at serve time."
            ),
        ),
        output: Path = typer.Option(
            Path(".pretensor") / "visibility.yml",
            "--output",
            help="Destination visibility.yml path.",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        roles: str | None = typer.Option(
            None,
            "--roles",
            help="Comma-separated DB role names to sync (omit to sync all grantees).",
        ),
        ctx: typer.Context = typer.Option(None, hidden=True),
    ) -> None:
        """Write or merge ``profiles`` with ``allowed_tables`` from SELECT table grants."""
        cli_config = get_cli_config(ctx)
        configured_output = cli_config.visibility.path
        if configured_output is None and cli_config.state_dir is not None:
            configured_output = cli_config.state_dir / "visibility.yml"
        output = resolve_path_option(
            ctx,
            param_name="output",
            cli_value=output,
            config_value=configured_output,
        )
        name = resolve_optional_str_option(
            ctx,
            param_name="name",
            cli_value=name,
            config_value=cli_config.connection_defaults.name,
        )
        raw_dsn = dsn.strip()
        try:
            db_type = infer_database_type_from_dsn(raw_dsn)
        except ValueError as exc:
            console.print(f"[red]Invalid DSN:[/red] {exc}")
            raise typer.Exit(1) from exc
        if db_type == DatabaseType.BIGQUERY:
            console.print(
                "[red]sync-grants does not support BigQuery in v1 "
                "(IAM grants are out of scope).[/red]",
            )
            raise typer.Exit(1)

        try:
            connection_name = name or default_connection_name(raw_dsn)
        except ValueError as exc:
            console.print(f"[red]Invalid DSN:[/red] {exc}")
            raise typer.Exit(1) from exc
        try:
            cfg = connection_config_from_url(raw_dsn, connection_name)
        except ValueError as exc:
            console.print(f"[red]Invalid DSN:[/red] {exc}")
            raise typer.Exit(1) from exc
        roles_filter = _parse_roles_option(roles)

        try:
            with get_connector(cfg) as connector:
                grantee_count = run_sync_grants(
                    connector,
                    output_path=output.resolve(),
                    connection_name=connection_name,
                    roles_filter=roles_filter,
                )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        except PermissionError as exc:
            console.print(
                f"[red]Cannot write visibility file {output}:[/red] {exc}\n"
                "Check write permissions on the output path."
            )
            raise typer.Exit(1) from exc
        except Exception as exc:
            console.print(
                f"[red]Cannot connect to database:[/red] {exc}\n"
                "Check that the DSN is correct and the database is reachable."
            )
            raise typer.Exit(1) from exc

        if grantee_count == 0:
            console.print(
                "[yellow]No SELECT grants returned by the connector.[/yellow] "
                "The DB role used for --dsn may lack catalog access, or the database "
                "has no non-PUBLIC grants. visibility.yml was still written (merged "
                "with any existing base rules).",
            )
        console.print(
            f"[green]Wrote merged visibility rules to[/green] {output.resolve()} "
            f"({grantee_count} grantee profile(s))",
        )
