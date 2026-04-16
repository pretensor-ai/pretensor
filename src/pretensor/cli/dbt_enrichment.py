"""Shared dbt manifest / sources path handling for CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.manifest import DbtManifest, DbtManifestError
from pretensor.enrichment.dbt.pipeline import run_dbt_enrichment_from_manifest


def resolve_dbt_sources_path(
    manifest_path: Path, explicit: Path | None
) -> Path | None:
    """Return path to ``sources.json`` for freshness signals.

    If ``explicit`` is set, it must exist as a file. If omitted, use
    ``<manifest_dir>/sources.json`` when that file exists.

    Args:
        manifest_path: Resolved manifest path (must exist when called for defaulting).
        explicit: User-provided ``--dbt-sources`` or ``None``.

    Returns:
        Path to read, or ``None`` when no sources file should be used.

    Raises:
        FileNotFoundError: When ``explicit`` is set but is not a file.
    """
    if explicit is not None:
        if not explicit.is_file():
            msg = f"dbt sources file not found or not a file: {explicit}"
            raise FileNotFoundError(msg)
        return explicit
    default = manifest_path.parent / "sources.json"
    return default if default.is_file() else None


def validate_dbt_manifest_path(path: Path) -> None:
    """Ensure ``--dbt-manifest`` points at an existing file before heavy work."""
    if not path.is_file():
        msg = f"dbt manifest not found or not a file: {path}"
        raise FileNotFoundError(msg)


def preload_dbt_manifest(
    manifest_path: Path, sources_explicit: Path | None, *, console: Console
) -> tuple[DbtManifest, Path | None]:
    """Validate and parse the manifest (and ``sources.json``) up front.

    Running this before any graph mutations means a malformed ``manifest.json``
    aborts the command *before* the DB is re-introspected and patched, instead
    of leaving a half-updated graph that then fails during enrichment.

    Raises:
        typer.Exit: With exit code 1 if either file is missing or the manifest
            cannot be parsed.
    """
    try:
        validate_dbt_manifest_path(manifest_path)
        sources = resolve_dbt_sources_path(manifest_path, sources_explicit)
        manifest = DbtManifest.load(manifest_path)
    except DbtManifestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    return manifest, sources


def apply_dbt_enrichment_cli(
    *,
    manifest: DbtManifest,
    sources_path: Path | None,
    store: KuzuStore,
    connection_name: str,
    console: Console,
) -> None:
    """Run dbt enrichment on an already-parsed manifest and print a summary line."""
    summary = run_dbt_enrichment_from_manifest(
        manifest, sources_path, store, connection_name
    )

    console.print(
        "[green]dbt enrichment:[/green] "
        f"lineage_edges={summary.lineage_edges} "
        f"tables_enriched={summary.tables_enriched} "
        f"tags_set={summary.tags_set} "
        f"exposures_marked={summary.exposures_marked} "
        f"freshness_rows={summary.freshness_rows_applied} "
        f"tests_counted={summary.tests_counted}"
    )
