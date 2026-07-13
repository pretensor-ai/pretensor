"""Shared CLI scaffolding for commands that operate over named sources."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from pretensor.introspection.models.dsn import validate_source_env_vars

if TYPE_CHECKING:
    from pretensor.cli.config_file import PretensorCliConfig


def resolve_embeddings_mode(
    embeddings: bool | None,
    console: Console,
    *,
    auto_enable_message: str | None = None,
) -> bool:
    """Verify or auto-detect the embeddings mode.

    Parameters
    ----------
    embeddings:
        Raw CLI value: ``True`` = explicit opt-in, ``False`` = explicit
        opt-out, ``None`` = auto-detect.
    console:
        Rich console for user-facing output.
    auto_enable_message:
        If given, printed when auto-detection enables embeddings.
        Pass ``None`` (default) to suppress the message.

    Returns
    -------
    bool
        Resolved embeddings flag.
    """
    if embeddings:
        from pretensor.intelligence.embeddings import verify_embeddings_extra_installed

        try:
            verify_embeddings_extra_installed()
        except ImportError as e:
            console.print(str(e), style="red", markup=False)
            raise typer.Exit(1) from e
        return True
    if embeddings is None:
        from pretensor.intelligence.embeddings import resolve_embeddings_auto

        resolved = resolve_embeddings_auto(None)
        if resolved and auto_enable_message:
            console.print(auto_enable_message)
        return resolved
    return False


def check_source_exists(
    source: str,
    cli_config: PretensorCliConfig,
    console: Console,
) -> None:
    """Exit 1 with an informative message if *source* is not in ``cli_config.sources``."""
    if source not in cli_config.sources:
        available = ", ".join(cli_config.sources) or "(none)"
        console.print(f"[red]Unknown source {source!r}.[/red] Available: {available}")
        raise typer.Exit(1)


def print_multi_source_summary(
    console: Console,
    results: list[tuple[str, bool, str]],
) -> None:
    """Print the tick/cross summary table and exit 1 if any source failed."""
    console.print(f"\n[bold]{'─' * 40}[/bold]")
    console.print("[bold]Summary:[/bold]")
    failed = 0
    for src_name, ok, msg in results:
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {status} {src_name}: {msg}")
        if not ok:
            failed += 1
    if failed:
        raise typer.Exit(1)


def run_for_all_sources(
    *,
    cli_config: PretensorCliConfig,
    console: Console,
    logger: logging.Logger,
    source_banner: str,
    action_verb: str,
    run_one: Callable[[str], None],
) -> None:
    """Iterate all configured sources, skip ones with missing env vars, exit 1 on any failure.

    Parameters
    ----------
    cli_config:
        Loaded CLI config (must have a non-empty ``sources`` mapping).
    console:
        Rich console for user-facing output.
    logger:
        Module-level logger for structured warnings/errors.
    source_banner:
        Label shown in the per-source header, e.g. ``"Source:"`` or
        ``"Reindexing source:"``.
    action_verb:
        Verb used in error messages, e.g. ``"indexing"`` or ``"reindexing"``.
    run_one:
        Callable ``(src_name) -> None`` that performs the per-source
        action; should raise ``typer.Exit`` on failure.
    """
    if not cli_config.sources:
        console.print(
            "[red]No sources defined in config.[/red] "
            "Add a `sources:` section to .pretensor/config.yaml."
        )
        raise typer.Exit(1)
    results: list[tuple[str, bool, str]] = []
    for src_name, src_cfg in cli_config.sources.items():
        console.print(
            f"\n[bold]{'─' * 40}[/bold]\n"
            f"[bold blue]{source_banner}[/bold blue] {src_name}\n"
        )
        missing_vars = validate_source_env_vars(src_cfg)
        if missing_vars:
            msg = f"missing env vars: {', '.join(missing_vars)}"
            console.print(f"[yellow]Skipping {src_name}:[/yellow] {msg}")
            logger.warning("Skipping source %s: %s", src_name, msg)
            results.append((src_name, False, f"skipped ({msg})"))
            continue
        try:
            run_one(src_name)
            results.append((src_name, True, "ok"))
        except typer.Exit:
            logger.warning("Source %s exited with failure", src_name)
            results.append((src_name, False, "failed"))
        except Exception as e:
            logger.exception("Error %s source %s", action_verb, src_name)
            console.print(f"[red]Error {action_verb} {src_name}:[/red] {e}")
            results.append((src_name, False, str(e)))
    print_multi_source_summary(console, results)
