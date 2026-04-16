"""``pretensor validate`` — sqlglot + graph SQL validation."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from pretensor.cli import constants as cli_constants
from pretensor.cli.config_file import get_cli_config, resolve_path_option
from pretensor.mcp.service_registry import (
    load_registry,
    open_store_for_entry,
    resolve_registry_entry,
)
from pretensor.validation.query_validator import QueryValidator

__all__ = ["register_validate_command"]


_EXIT_INVALID = 1
_EXIT_BAD_ARGS = 2


def register_validate_command(app: typer.Typer, *, console: Console) -> None:
    """Register ``pretensor validate`` onto ``app``."""

    @app.command("validate")
    def validate_command(
        sql_or_path: str = typer.Argument(
            ...,
            help=(
                "SQL text or path to a .sql file. If the argument names an "
                "existing file, its contents are read."
            ),
        ),
        db: str | None = typer.Option(
            None,
            "--db",
            help="connection_name or logical database; omit if only one is indexed.",
        ),
        dialect: str = typer.Option(
            "postgres",
            "--dialect",
            help="sqlglot dialect (default 'postgres').",
        ),
        state_dir: Path = typer.Option(
            cli_constants.DEFAULT_STATE_DIR,
            "--state-dir",
            help="Directory containing registry.json.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        as_json: bool = typer.Option(
            False,
            "--json",
            help="Emit machine-readable JSON instead of a rich panel.",
        ),
        ctx: typer.Context = typer.Option(None, hidden=True),
    ) -> None:
        """Validate SQL against the indexed graph for one database."""
        cli_config = get_cli_config(ctx)
        state_dir = resolve_path_option(
            ctx,
            param_name="state_dir",
            cli_value=state_dir,
            config_value=cli_config.state_dir,
        )

        maybe_path = Path(sql_or_path)
        if maybe_path.exists() and maybe_path.is_file():
            sql_text = maybe_path.read_text(encoding="utf-8")
        else:
            sql_text = sql_or_path

        if not sql_text.strip():
            console.print("[red]Empty SQL input.[/red]")
            raise typer.Exit(_EXIT_BAD_ARGS)

        reg = load_registry(state_dir)
        entry = resolve_registry_entry(reg, db)
        if entry is None:
            msg = (
                f"no registry entry matches --db {db!r}"
                if db is not None
                else "multiple indexed databases — pass --db"
            )
            console.print(f"[red]{msg}[/red]")
            raise typer.Exit(_EXIT_BAD_ARGS)

        store = open_store_for_entry(entry)
        try:
            validator = QueryValidator(
                store,
                connection_name=entry.connection_name,
                database_key=entry.database,
                dialect=dialect,
            )
            result = validator.validate(sql_text)
        finally:
            store.close()

        if as_json:
            payload: dict[str, object] = {
                "valid": result.valid,
                "dialect": dialect,
                "syntax_errors": result.syntax_errors,
                "missing_tables": result.missing_tables,
                "missing_columns": result.missing_columns,
                "invalid_joins": [
                    {
                        "message": j.message,
                        "left_table": j.left_table,
                        "right_table": j.right_table,
                    }
                    for j in result.invalid_joins
                ],
                "suggestions": result.suggestions,
            }
            typer.echo(json.dumps(payload, indent=2))
        else:
            valid_label = (
                "[green]valid[/green]" if result.valid else "[red]invalid[/red]"
            )
            lines = [
                f"[bold]Dialect:[/bold] {dialect}",
                f"[bold]Valid:[/bold] {valid_label}",
            ]
            if result.syntax_errors:
                lines.append("")
                lines.append("[bold]Syntax errors[/bold]")
                lines.extend(f"  - {e}" for e in result.syntax_errors)
            if result.missing_tables:
                lines.append("")
                lines.append("[bold]Missing tables[/bold]")
                lines.extend(f"  - {t}" for t in result.missing_tables)
            if result.missing_columns:
                lines.append("")
                lines.append("[bold]Missing columns[/bold]")
                lines.extend(f"  - {c}" for c in result.missing_columns)
            if result.invalid_joins:
                lines.append("")
                lines.append("[bold]Invalid joins[/bold]")
                lines.extend(f"  - {j.message}" for j in result.invalid_joins)
            if result.suggestions:
                lines.append("")
                lines.append("[bold]Suggestions[/bold]")
                lines.extend(f"  - {s}" for s in result.suggestions)
            console.print(Panel("\n".join(lines), title="validate_sql"))

        if not result.valid:
            raise typer.Exit(_EXIT_INVALID)
