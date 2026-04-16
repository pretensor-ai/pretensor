"""``pretensor semantic`` command group — compile YAML metrics to SQL."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from pretensor.cli import constants as cli_constants
from pretensor.cli.config_file import get_cli_config, resolve_path_option
from pretensor.introspection.models.semantic import (
    SemanticLayer as SemanticLayerModel,
)
from pretensor.mcp.service_registry import (
    load_registry,
    open_store_for_entry,
    resolve_registry_entry,
)
from pretensor.semantic.compiler import MetricCompileError, MetricSqlCompiler

__all__ = ["register_semantic_commands"]


_EXIT_INVALID = 1
_EXIT_BAD_ARGS = 2


def register_semantic_commands(app: typer.Typer, *, console: Console) -> None:
    """Register ``pretensor semantic compile`` onto ``app``."""
    semantic_app = typer.Typer(
        name="semantic",
        help="Compile and inspect user-authored semantic layers (OSS).",
        no_args_is_help=True,
    )

    @semantic_app.command("compile")
    def compile_command(
        yaml_path: Path = typer.Argument(
            ...,
            help="Path to a semantic layer YAML file.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
        ),
        metric: str = typer.Option(
            ...,
            "--metric",
            help="Name of the metric to compile.",
        ),
        db: str | None = typer.Option(
            None,
            "--db",
            help="connection_name or logical database; omit if only one is indexed.",
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
        """Compile a YAML metric to SQL and validate it against the graph."""
        cli_config = get_cli_config(ctx)
        state_dir = resolve_path_option(
            ctx,
            param_name="state_dir",
            cli_value=state_dir,
            config_value=cli_config.state_dir,
        )

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

        try:
            layer = SemanticLayerModel.from_yaml(
                yaml_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            console.print(f"[red]Failed to parse {yaml_path}:[/red] {exc}")
            raise typer.Exit(_EXIT_BAD_ARGS) from exc

        if layer.connection_name != entry.connection_name and (
            layer.connection_name != entry.database
        ):
            console.print(
                f"[yellow]Warning: YAML connection_name {layer.connection_name!r} "
                f"does not match registry entry "
                f"({entry.connection_name!r} / {entry.database!r}).[/yellow]"
            )

        store = open_store_for_entry(entry)
        try:
            compiler = MetricSqlCompiler(
                store,
                connection_name=entry.connection_name,
                database_key=entry.database,
            )
            try:
                compiled = compiler.compile(layer, metric)
            except MetricCompileError as exc:
                if as_json:
                    typer.echo(json.dumps({"error": str(exc)}))
                else:
                    console.print(f"[red]Compile error:[/red] {exc}")
                raise typer.Exit(_EXIT_BAD_ARGS) from exc

            if as_json:
                payload: dict[str, object] = {
                    "metric": compiled.metric,
                    "entity": compiled.entity,
                    "sql": compiled.sql,
                    "dialect": compiled.dialect,
                    "warnings": compiled.warnings,
                    "valid": compiled.validation.valid,
                    "syntax_errors": compiled.validation.syntax_errors,
                    "missing_tables": compiled.validation.missing_tables,
                    "missing_columns": compiled.validation.missing_columns,
                    "invalid_joins": [
                        {
                            "message": j.message,
                            "left_table": j.left_table,
                            "right_table": j.right_table,
                        }
                        for j in compiled.validation.invalid_joins
                    ],
                    "suggestions": compiled.validation.suggestions,
                }
                typer.echo(json.dumps(payload, indent=2))
            else:
                valid_label = (
                    "[green]valid[/green]"
                    if compiled.validation.valid
                    else "[red]invalid[/red]"
                )
                lines = [
                    f"[bold]Entity:[/bold] {compiled.entity}",
                    f"[bold]Dialect:[/bold] {compiled.dialect}",
                    f"[bold]Valid:[/bold] {valid_label}",
                    "",
                    "[bold]SQL[/bold]",
                    compiled.sql,
                ]
                if compiled.warnings:
                    lines.append("")
                    lines.append("[bold]Warnings[/bold]")
                    lines.extend(f"  - {w}" for w in compiled.warnings)
                if not compiled.validation.valid:
                    lines.append("")
                    lines.append("[bold]Validation errors[/bold]")
                    for err in compiled.validation.syntax_errors:
                        lines.append(f"  - syntax: {err}")
                    for t in compiled.validation.missing_tables:
                        lines.append(f"  - missing_table: {t}")
                    for c in compiled.validation.missing_columns:
                        lines.append(f"  - missing_column: {c}")
                    for j in compiled.validation.invalid_joins:
                        lines.append(f"  - invalid_join: {j.message}")
                if compiled.validation.suggestions:
                    lines.append("")
                    lines.append("[bold]Suggestions[/bold]")
                    lines.extend(f"  - {s}" for s in compiled.validation.suggestions)
                console.print(
                    Panel("\n".join(lines), title=f"metric: {compiled.metric}")
                )

            if not compiled.validation.valid:
                raise typer.Exit(_EXIT_INVALID)
        finally:
            store.close()

    app.add_typer(semantic_app)
