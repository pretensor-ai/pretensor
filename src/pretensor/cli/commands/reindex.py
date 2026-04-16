"""``pretensor reindex`` command (snapshot diff + graph patch)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from pretensor.cli import constants as cli_constants
from pretensor.cli.config_file import (
    get_cli_config,
    resolve_optional_str_option,
    resolve_path_option,
)
from pretensor.cli.dbt_enrichment import (
    apply_dbt_enrichment_cli,
    preload_dbt_manifest,
)
from pretensor.cli.paths import default_connection_name, keystore_path
from pretensor.config import PretensorConfig
from pretensor.connectors.inspect import inspect
from pretensor.connectors.snapshot import diff_snapshots
from pretensor.core.dsn_crypto import DSNEncryptor
from pretensor.core.registry import DatabaseNotFoundError, GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.intelligence.discovery import RelationshipDiscovery
from pretensor.intelligence.metric_templates import MetricTemplateBuilder
from pretensor.intelligence.pipeline import run_intelligence_layer_sync
from pretensor.introspection.models.dsn import (
    connection_config_from_registry_dsn,
    connection_config_from_source,
    connection_config_from_url,
    dsn_from_source,
    registry_dialect_for,
    validate_source_env_vars,
)
from pretensor.observability import log_timed_operation
from pretensor.skills.generator import SkillGenerator
from pretensor.staleness.graph_patcher import GraphPatcher
from pretensor.staleness.impact_analyzer import ImpactAnalyzer
from pretensor.staleness.snapshot_store import SnapshotStore

logger = logging.getLogger(__name__)


def register_reindex_command(app: typer.Typer, *, console: Console) -> None:
    @app.command("reindex")
    def reindex_command(
        dsn: str | None = typer.Argument(
            None,
            help=(
                "Database DSN URL (must match the indexed connection unless --dialect is set). "
                "Mutually exclusive with --source and --all."
            ),
        ),
        source: str | None = typer.Option(
            None,
            "--source",
            "-s",
            help="Named source from .pretensor/config.yaml sources: block.",
        ),
        all_sources: bool = typer.Option(
            False,
            "--all",
            help="Reindex all sources defined in .pretensor/config.yaml.",
        ),
        dialect: str | None = typer.Option(
            None,
            "--dialect",
            help="Override connector for this run: postgres, postgresql, snowflake, or bigquery.",
        ),
        database: str | None = typer.Option(
            None,
            "--database",
            "-d",
            help="Registry connection name (defaults to database name from the DSN).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Show schema diff and planned mutations without writing the graph.",
        ),
        recompute_intelligence: bool = typer.Option(
            False,
            "--recompute-intelligence",
            help="After patching, re-run relationship discovery and clustering/join paths.",
        ),
        state_dir: Path = typer.Option(
            cli_constants.DEFAULT_STATE_DIR,
            "--state-dir",
            help="Directory for registry.json, snapshots, and graph files.",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
        skills_target: str = typer.Option(
            "claude",
            "--skills-target",
            help=(
                "Where to write the generated skill after a successful patch: "
                "claude, cursor, all, or a file path."
            ),
        ),
        dbt_manifest: Path | None = typer.Option(
            None,
            "--dbt-manifest",
            help=(
                "Path to dbt target/manifest.json. Run `dbt compile` or `dbt run` first "
                "so the manifest exists; enrichment runs after the graph is updated."
            ),
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        dbt_sources: Path | None = typer.Option(
            None,
            "--dbt-sources",
            help=(
                "Path to dbt sources.json (e.g. from `dbt source freshness`). "
                "Defaults to target/sources.json next to the manifest when that file exists."
            ),
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        ctx: typer.Context = typer.Option(
            None,
            hidden=True,
        ),
    ) -> None:
        """Re-introspect the DB, diff against the last snapshot, and patch the Kuzu graph."""
        cli_config = get_cli_config(ctx)

        # --- mutual exclusion -------------------------------------------------
        modes = sum([dsn is not None, source is not None, all_sources])
        if modes > 1:
            console.print(
                "[red]DSN argument, --source, and --all are mutually exclusive.[/red]"
            )
            raise typer.Exit(1)

        state_dir = resolve_path_option(
            ctx,
            param_name="state_dir",
            cli_value=state_dir,
            config_value=cli_config.state_dir,
        )

        # --- --all: iterate all configured sources ----------------------------
        if all_sources:
            if not cli_config.sources:
                console.print(
                    "[red]No sources defined in config.[/red] "
                    "Add a `sources:` section to .pretensor/config.yaml."
                )
                raise typer.Exit(1)
            results: list[tuple[str, bool, str]] = []
            for src_name in cli_config.sources:
                console.print(
                    f"\n[bold]{'─' * 40}[/bold]\n"
                    f"[bold blue]Reindexing source:[/bold blue] {src_name}\n"
                )
                src_cfg = cli_config.sources[src_name]
                missing_vars = validate_source_env_vars(src_cfg)
                if missing_vars:
                    msg = f"missing env vars: {', '.join(missing_vars)}"
                    console.print(f"[yellow]Skipping {src_name}:[/yellow] {msg}")
                    logger.warning("Skipping source %s: %s", src_name, msg)
                    results.append((src_name, False, f"skipped ({msg})"))
                    continue
                try:
                    _run_single_reindex(
                        console=console,
                        cli_config=cli_config,
                        ctx=ctx,
                        dsn=None,
                        source_name=src_name,
                        dialect=dialect,
                        database=src_name,
                        dry_run=dry_run,
                        recompute_intelligence=recompute_intelligence,
                        state_dir=state_dir,
                        skills_target=skills_target,
                        dbt_manifest=dbt_manifest,
                        dbt_sources=dbt_sources,
                    )
                    results.append((src_name, True, "ok"))
                except typer.Exit:
                    logger.warning("Source %s exited with failure", src_name)
                    results.append((src_name, False, "failed"))
                except Exception as e:
                    logger.exception("Error reindexing source %s", src_name)
                    console.print(f"[red]Error reindexing {src_name}:[/red] {e}")
                    results.append((src_name, False, str(e)))

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
            return

        # --- --source: single named source ------------------------------------
        if source is not None:
            if source not in cli_config.sources:
                available = ", ".join(cli_config.sources) or "(none)"
                console.print(
                    f"[red]Unknown source {source!r}.[/red] "
                    f"Available: {available}"
                )
                raise typer.Exit(1)
            _run_single_reindex(
                console=console,
                cli_config=cli_config,
                ctx=ctx,
                dsn=None,
                source_name=source,
                dialect=dialect,
                database=source,
                dry_run=dry_run,
                recompute_intelligence=recompute_intelligence,
                state_dir=state_dir,
                skills_target=skills_target,
                dbt_manifest=dbt_manifest,
                dbt_sources=dbt_sources,
            )
            return

        # --- DSN argument (original path) or no mode (use registry) -----------
        if dsn is None:
            # No DSN, no source, no --all: need at least a database name
            database = resolve_optional_str_option(
                ctx,
                param_name="database",
                cli_value=database,
                config_value=cli_config.connection_defaults.name,
            )
            if not database:
                console.print(
                    "[red]Provide a DSN argument, --source <name>, --all, "
                    "or --database <name>.[/red]"
                )
                raise typer.Exit(1)

        _run_single_reindex(
            console=console,
            cli_config=cli_config,
            ctx=ctx,
            dsn=dsn,
            source_name=None,
            dialect=dialect,
            database=database,
            dry_run=dry_run,
            recompute_intelligence=recompute_intelligence,
            state_dir=state_dir,
            skills_target=skills_target,
            dbt_manifest=dbt_manifest,
            dbt_sources=dbt_sources,
        )


def _run_single_reindex(
    *,
    console: Console,
    cli_config: object,
    ctx: typer.Context | None,
    dsn: str | None,
    source_name: str | None,
    dialect: str | None,
    database: str | None,
    dry_run: bool,
    recompute_intelligence: bool,
    state_dir: Path,
    skills_target: str,
    dbt_manifest: Path | None,
    dbt_sources: Path | None,
) -> None:
    """Core reindex logic shared by DSN, --source, and --all paths."""
    from pretensor.cli.config_file import PretensorCliConfig

    assert isinstance(cli_config, PretensorCliConfig)

    database = resolve_optional_str_option(
        ctx,
        param_name="database",
        cli_value=database,
        config_value=cli_config.connection_defaults.name,
    )
    dialect = resolve_optional_str_option(
        ctx,
        param_name="dialect",
        cli_value=dialect,
        config_value=cli_config.connection_defaults.dialect,
    )
    total_started = time.perf_counter()

    # Resolve connection_name and DSN
    if source_name is not None and source_name in cli_config.sources:
        src_cfg = cli_config.sources[source_name]
        connection_name = source_name
        try:
            dsn_resolved = dsn_from_source(source_name, src_cfg)
        except ValueError as e:
            console.print(f"[red]Cannot build DSN from source:[/red] {e}")
            raise typer.Exit(1) from e
    elif dsn is not None:
        connection_name = database or default_connection_name(dsn)
        dsn_resolved = dsn
    else:
        connection_name = database or ""
        dsn_resolved = None

    preloaded_manifest = None
    preloaded_sources: Path | None = None
    if dbt_manifest is not None:
        preloaded_manifest, preloaded_sources = preload_dbt_manifest(
            dbt_manifest, dbt_sources, console=console
        )
    reg_path = state_dir / cli_constants.REGISTRY_FILENAME
    if not reg_path.exists():
        console.print("[red]No registry found.[/red] Run `pretensor index` first.")
        raise typer.Exit(1)
    reg = GraphRegistry(reg_path).load()
    ks = keystore_path(state_dir)
    try:
        entry = reg.require(connection_name)
    except DatabaseNotFoundError:
        console.print(
            f"[red]Unknown connection {connection_name!r}.[/red] "
            "Use `pretensor list` or `--database`."
        )
        raise typer.Exit(1)

    # For source-based reindex, use source config directly; otherwise use registry DSN
    if source_name is not None and source_name in cli_config.sources:
        src_cfg = cli_config.sources[source_name]
        try:
            config = connection_config_from_source(source_name, src_cfg)
        except ValueError as e:
            console.print(f"[red]Invalid source config:[/red] {e}")
            raise typer.Exit(1) from e
        dsn_for_registry = dsn_resolved or ""
    else:
        decryptor = DSNEncryptor(ks) if entry.dsn_encrypted else None
        dsn_for_reindex = entry.plaintext_dsn(decryptor)
        dsn_for_registry = dsn_resolved or dsn_for_reindex

        if dialect is not None:
            config = connection_config_from_url(
                dsn_for_reindex, connection_name, dialect_override=dialect
            )
        else:
            config = connection_config_from_registry_dsn(
                dsn_for_reindex, connection_name, entry.dialect
            )

    snap_store = SnapshotStore(state_dir)
    old_snapshot = snap_store.load(connection_name)
    if old_snapshot is None:
        console.print(
            "[red]No saved snapshot for this connection.[/red] "
            "Run `pretensor index` once, then use `reindex`."
        )
        raise typer.Exit(1)

    console.print(f"[bold blue]Re-introspecting[/bold blue] '{connection_name}'...")
    try:
        with log_timed_operation(
            logger,
            event="reindex.inspect",
            connection_name=connection_name,
            dialect=str(config.type),
        ):
            new_snapshot = inspect(config)
    except Exception as e:
        console.print(
            f"[red]Cannot connect to database:[/red] {e}\n"
            "Check that the database is reachable and the DSN is correct."
        )
        raise typer.Exit(1) from e

    changes = diff_snapshots(old_snapshot, new_snapshot)
    graph_path = Path(entry.unified_graph_path or entry.graph_path)
    if not graph_path.exists():
        console.print(f"[red]Graph file missing:[/red] {graph_path}")
        raise typer.Exit(1)

    try:
        store = KuzuStore(graph_path)
    except Exception as e:
        console.print(
            f"[red]Cannot open graph file {graph_path}:[/red] {e}\n"
            "The file may be corrupt. Run `pretensor index` to rebuild it."
        )
        raise typer.Exit(1) from e
    try:
        analyzer = ImpactAnalyzer(store)
        impact = analyzer.analyze(
            changes,
            connection_name=connection_name,
            database_key=str(entry.database),
        )
        if changes:
            console.print(
                f"\n[bold]Schema changes[/bold] for {connection_name!r} "
                f"({len(changes)} item(s)):"
            )
            for ch in changes:
                col = f".{ch.column_name}" if ch.column_name else ""
                console.print(
                    f"  {ch.change_type.value} {ch.target.value}: "
                    f"{ch.schema_name}.{ch.table_name}{col}"
                    + (f" — {ch.details}" if ch.details else "")
                )
            console.print(f"\n[dim]{impact.summary}[/dim]\n")
        else:
            console.print("[green]No schema changes since last snapshot.[/green]")

        patcher = GraphPatcher(store)
        with log_timed_operation(
            logger,
            event="reindex.apply_patch",
            connection_name=connection_name,
            dry_run=dry_run,
            change_count=len(changes),
        ):
            patch = patcher.apply(changes, new_snapshot, dry_run=dry_run)
        if changes and dry_run:
            console.print("[bold]Planned graph updates (dry-run):[/bold]")
            console.print(
                f"  tables +{patch.tables_added} / -{patch.tables_removed}, "
                f"columns +{patch.columns_added} / -{patch.columns_removed} / "
                f"~{patch.columns_updated}, "
                f"FK edges removed (column touch): {patch.fk_edges_removed}, "
                f"inferred removed: {patch.inferred_joins_removed}"
            )
            console.print("\n[yellow]Run without --dry-run to apply.[/yellow]")
            return

        if changes and not dry_run:
            MetricTemplateBuilder.mark_stale_for_database(
                store, str(new_snapshot.database)
            )

        if not changes:
            if preloaded_manifest is not None and not dry_run:
                apply_dbt_enrichment_cli(
                    manifest=preloaded_manifest,
                    sources_path=preloaded_sources,
                    store=store,
                    connection_name=connection_name,
                    console=console,
                )
            indexed_at = datetime.now(timezone.utc)
            enc = DSNEncryptor(ks) if ks.exists() and entry.dsn_encrypted else None
            reg.upsert(
                connection_name=connection_name,
                database=new_snapshot.database,
                dsn=dsn_for_registry,
                graph_path=graph_path,
                unified_graph_path=Path(entry.unified_graph_path)
                if entry.unified_graph_path
                else None,
                indexed_at=indexed_at,
                encrypt_dsn=bool(entry.dsn_encrypted),
                encryptor=enc,
                dialect=registry_dialect_for(config.type),
                table_count=entry.table_count,
                dbt_manifest_path=(
                    str(dbt_manifest) if dbt_manifest is not None
                    else entry.dbt_manifest_path
                ),
                llm_enrichment_ran=entry.llm_enrichment_ran,
            )
            reg.save()
            snap_store.save(connection_name, new_snapshot)
            console.print("[dim]Registry timestamp refreshed.[/dim]")
            return

        if recompute_intelligence and not dry_run:
            with log_timed_operation(
                logger,
                event="reindex.relationship_discovery",
                connection_name=connection_name,
            ):
                RelationshipDiscovery(store).discover(new_snapshot)
            with log_timed_operation(
                logger,
                event="reindex.intelligence_sync",
                connection_name=connection_name,
            ):
                run_intelligence_layer_sync(
                    store,
                    new_snapshot.database,
                    config=PretensorConfig(graph=cli_config.graph),
                )

        if preloaded_manifest is not None and not dry_run:
            apply_dbt_enrichment_cli(
                manifest=preloaded_manifest,
                sources_path=preloaded_sources,
                store=store,
                connection_name=connection_name,
                console=console,
            )

        snap_store.save(connection_name, new_snapshot)
        indexed_at = datetime.now(timezone.utc)
        tc_rows = store.query_all_rows(
            """
            MATCH (t:SchemaTable {connection_name: $cn})
            RETURN count(*)
            """,
            {"cn": connection_name},
        )
        table_count = (
            int(tc_rows[0][0]) if tc_rows and tc_rows[0][0] is not None else 0
        )
        enc = DSNEncryptor(ks) if ks.exists() and entry.dsn_encrypted else None
        reg.upsert(
            connection_name=connection_name,
            database=new_snapshot.database,
            dsn=dsn_for_registry,
            graph_path=graph_path,
            unified_graph_path=Path(entry.unified_graph_path)
            if entry.unified_graph_path
            else None,
            indexed_at=indexed_at,
            encrypt_dsn=bool(entry.dsn_encrypted),
            encryptor=enc,
            dialect=registry_dialect_for(config.type),
            table_count=table_count,
            dbt_manifest_path=(
                str(dbt_manifest) if dbt_manifest is not None
                else entry.dbt_manifest_path
            ),
            llm_enrichment_ran=entry.llm_enrichment_ran,
        )
        reg.save()

        skill_store = KuzuStore(graph_path)
        try:
            refreshed = reg.get(connection_name)
            if refreshed is not None:
                SkillGenerator.write_for_index(
                    store=skill_store,
                    entry=refreshed,
                    skills_target=skills_target,
                )
        finally:
            skill_store.close()

        console.print(f"[green]Patched graph:[/green] {graph_path}")
        console.print(
            f"[green]Snapshot saved:[/green] {snap_store.path_for(connection_name)}"
        )
        logger.info(
            "reindex completed in %.2fms",
            (time.perf_counter() - total_started) * 1000,
            extra={
                "event": "reindex.total",
                "status": "ok",
                "duration_ms": (time.perf_counter() - total_started) * 1000,
                "connection_name": connection_name,
                "dry_run": dry_run,
                "change_count": len(changes),
                "recompute_intelligence": recompute_intelligence,
            },
        )
    finally:
        store.close()
