"""``pretensor index`` command."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from pretensor.cli import constants as cli_constants
from pretensor.cli.config_file import (
    get_cli_config,
    resolve_optional_path_option,
    resolve_optional_str_option,
    resolve_path_option,
)
from pretensor.cli.dbt_enrichment import (
    apply_dbt_enrichment_cli,
    preload_dbt_manifest,
)
from pretensor.cli.paths import (
    default_connection_name,
    graph_file_for_connection,
    keystore_path,
    unified_graph_path,
)
from pretensor.connectors.inspect import inspect
from pretensor.core.builder import GraphBuilder
from pretensor.core.dsn_crypto import DSNEncryptor
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore
from pretensor.introspection.models.dsn import (
    connection_config_from_source,
    connection_config_from_url,
    dsn_from_source,
    registry_dialect_for,
    validate_source_env_vars,
)
from pretensor.observability import log_timed_operation
from pretensor.skills.generator import SkillGenerator
from pretensor.staleness.snapshot_store import SnapshotStore
from pretensor.visibility.config import load_visibility_config, merge_profile_into_base
from pretensor.visibility.filter import VisibilityFilter

_PROFILE_INDEX = os.environ.get("PRETENSOR_PROFILE_INDEX", "").lower() not in (
    "",
    "0",
    "false",
    "no",
)
logger = logging.getLogger(__name__)


def register_index_command(app: typer.Typer, *, console: Console) -> None:
    @app.command("index")
    def index_command(
        dsn: str | None = typer.Argument(
            None,
            help=(
                "Database DSN URL (postgresql://, postgres://, snowflake://, or bigquery://). "
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
            help="Index all sources defined in .pretensor/config.yaml.",
        ),
        dialect: str | None = typer.Option(
            None,
            "--dialect",
            help=(
                "Force connector: postgres, postgresql, snowflake, or bigquery "
                "(overrides URL scheme). Only used with DSN argument."
            ),
        ),
        name: str | None = typer.Option(
            None,
            "--name",
            "-n",
            help="Logical connection name (defaults to the database name from the DSN).",
        ),
        state_dir: Path = typer.Option(
            cli_constants.DEFAULT_STATE_DIR,
            "--state-dir",
            help="Directory for registry.json and graph files.",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
        unified: bool = typer.Option(
            False,
            "--unified",
            help="Merge into a single Kuzu file (graphs/unified.kuzu) for cross-DB linking.",
        ),
        skills_target: str = typer.Option(
            "claude",
            "--skills-target",
            help=(
                "Where to write the generated skill: claude (.claude/skills/), "
                "cursor (.cursor/rules/), all (both), or a file path."
            ),
        ),
        visibility_file: Path | None = typer.Option(
            None,
            "--visibility",
            help="Path to visibility.yml (default: <state-dir>/visibility.yml).",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        profile: str | None = typer.Option(
            None,
            "--profile",
            help="Named visibility profile from visibility.yml (merged at index time).",
        ),
        dbt_manifest: Path | None = typer.Option(
            None,
            "--dbt-manifest",
            help=(
                "Path to dbt target/manifest.json. Run `dbt compile` or `dbt run` first "
                "so the manifest exists; then lineage, descriptions/tags, and exposure "
                "signals are merged into the graph after indexing."
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
        """Introspect a database and write a Kuzu graph with explicit FK edges."""
        cli_config = get_cli_config(ctx)

        # --- mutual exclusion -------------------------------------------------
        modes = sum([dsn is not None, source is not None, all_sources])
        if modes == 0:
            console.print(
                "[red]Provide a DSN argument, --source <name>, or --all.[/red]"
            )
            raise typer.Exit(1)
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
        visibility_file = resolve_optional_path_option(
            ctx,
            param_name="visibility_file",
            cli_value=visibility_file,
            config_value=cli_config.visibility.path,
        )
        profile = resolve_optional_str_option(
            ctx,
            param_name="profile",
            cli_value=profile,
            config_value=cli_config.visibility.profile,
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
            for src_name, src_cfg in cli_config.sources.items():
                console.print(
                    f"\n[bold]{'─' * 40}[/bold]\n"
                    f"[bold blue]Source:[/bold blue] {src_name}\n"
                )
                missing_vars = validate_source_env_vars(src_cfg)
                if missing_vars:
                    msg = f"missing env vars: {', '.join(missing_vars)}"
                    console.print(f"[yellow]Skipping {src_name}:[/yellow] {msg}")
                    logger.warning("Skipping source %s: %s", src_name, msg)
                    results.append((src_name, False, f"skipped ({msg})"))
                    continue
                try:
                    src_config = connection_config_from_source(src_name, src_cfg)
                    src_dsn = dsn_from_source(src_name, src_cfg)
                    _run_index(
                        console=console,
                        cli_config=cli_config,
                        dsn=src_dsn,
                        connection_name=src_name,
                        config=src_config,
                        state_dir=state_dir,
                        unified=unified,
                        skills_target=skills_target,
                        visibility_file=visibility_file,
                        profile=profile,
                        dbt_manifest=dbt_manifest,
                        dbt_sources=dbt_sources,
                    )
                    results.append((src_name, True, "ok"))
                except typer.Exit:
                    logger.warning("Source %s exited with failure", src_name)
                    results.append((src_name, False, "failed"))
                except Exception as e:
                    logger.exception("Error indexing source %s", src_name)
                    console.print(f"[red]Error indexing {src_name}:[/red] {e}")
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
            src_cfg = cli_config.sources[source]
            try:
                config = connection_config_from_source(source, src_cfg)
            except ValueError as e:
                console.print(f"[red]Invalid source config:[/red] {e}")
                raise typer.Exit(1) from e
            try:
                dsn_str = dsn_from_source(source, src_cfg)
            except ValueError as e:
                console.print(f"[red]Cannot build DSN from source:[/red] {e}")
                raise typer.Exit(1) from e
            connection_name = source
            _run_index(
                console=console,
                cli_config=cli_config,
                dsn=dsn_str,
                connection_name=connection_name,
                config=config,
                state_dir=state_dir,
                unified=unified,
                skills_target=skills_target,
                visibility_file=visibility_file,
                profile=profile,
                dbt_manifest=dbt_manifest,
                dbt_sources=dbt_sources,
            )
            return

        # --- DSN argument (original path) -------------------------------------
        assert dsn is not None
        name = resolve_optional_str_option(
            ctx,
            param_name="name",
            cli_value=name,
            config_value=cli_config.connection_defaults.name,
        )
        dialect = resolve_optional_str_option(
            ctx,
            param_name="dialect",
            cli_value=dialect,
            config_value=cli_config.connection_defaults.dialect,
        )
        try:
            connection_name = name or default_connection_name(dsn)
        except ValueError as e:
            console.print(f"[red]Invalid DSN:[/red] {e}")
            raise typer.Exit(1) from e
        try:
            config = connection_config_from_url(
                dsn, connection_name, dialect_override=dialect
            )
        except ValueError as e:
            console.print(f"[red]Invalid DSN:[/red] {e}")
            raise typer.Exit(1) from e
        _run_index(
            console=console,
            cli_config=cli_config,
            dsn=dsn,
            connection_name=connection_name,
            config=config,
            state_dir=state_dir,
            unified=unified,
            skills_target=skills_target,
            visibility_file=visibility_file,
            profile=profile,
            dbt_manifest=dbt_manifest,
            dbt_sources=dbt_sources,
        )


def _run_index(
    *,
    console: Console,
    cli_config: object,
    dsn: str,
    connection_name: str,
    config: object,
    state_dir: Path,
    unified: bool,
    skills_target: str,
    visibility_file: Path | None,
    profile: str | None,
    dbt_manifest: Path | None,
    dbt_sources: Path | None,
) -> None:
    """Core indexing logic shared by DSN, --source, and --all paths."""
    from pretensor.introspection.models.config import ConnectionConfig as _CC

    assert isinstance(config, _CC)

    total_started = time.perf_counter()
    sd = Path(state_dir)
    vis_path = (
        Path(visibility_file)
        if visibility_file is not None
        else sd / "visibility.yml"
    )
    try:
        vis_cfg = merge_profile_into_base(load_visibility_config(vis_path), profile)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    visibility_filter = (
        VisibilityFilter.from_config(vis_cfg)
        if vis_cfg.hidden_schemas
        or vis_cfg.hidden_tables
        or vis_cfg.hidden_columns
        or vis_cfg.allowed_schemas
        or vis_cfg.allowed_tables
        else None
    )
    preloaded_manifest = None
    preloaded_sources: Path | None = None
    if dbt_manifest is not None:
        preloaded_manifest, preloaded_sources = preload_dbt_manifest(
            dbt_manifest, dbt_sources, console=console
        )

    try:
        sd.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        console.print(
            f"[red]Cannot create state directory {sd}:[/red] {e}\n"
            "Check write permissions or use --state-dir to specify a writable path."
        )
        raise typer.Exit(1) from e

    console.print(f"[bold blue]Introspecting[/bold blue] '{connection_name}'...")
    _t_inspect = time.perf_counter() if _PROFILE_INDEX else 0.0
    try:
        with log_timed_operation(
            logger,
            event="index.inspect",
            connection_name=connection_name,
            dialect=str(config.type),
        ):
            snapshot = inspect(config)
    except Exception as e:
        console.print(
            f"[red]Cannot connect to database:[/red] {e}\n"
            "Check that the DSN is correct and the database is reachable."
        )
        raise typer.Exit(1) from e
    if _PROFILE_INDEX:
        console.print(
            f"[dim][profile] inspect: {time.perf_counter() - _t_inspect:.2f}s[/dim]"
        )

    graph_path = (
        unified_graph_path(sd)
        if unified
        else graph_file_for_connection(sd, connection_name)
    )
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    if unified:
        if graph_path.exists():
            try:
                store = KuzuStore(graph_path)
            except Exception as e:
                console.print(
                    f"[red]Cannot open graph file {graph_path}:[/red] {e}\n"
                    "The file may be corrupt. Delete it and re-run `pretensor index`."
                )
                raise typer.Exit(1) from e
            try:
                store.ensure_schema()
                store.clear_connection_subgraph(connection_name)
            finally:
                store.close()
    else:
        if graph_path.exists():
            try:
                graph_path.unlink()
            except PermissionError as e:
                console.print(
                    f"[red]Cannot remove existing graph file {graph_path}:[/red] {e}"
                )
                raise typer.Exit(1) from e

    try:
        store = KuzuStore(graph_path)
    except Exception as e:
        console.print(
            f"[red]Cannot create graph file {graph_path}:[/red] {e}\n"
            "Check disk space and permissions."
        )
        raise typer.Exit(1) from e
    try:
        _t_build = time.perf_counter() if _PROFILE_INDEX else 0.0
        with log_timed_operation(
            logger,
            event="index.graph_build",
            connection_name=connection_name,
            replace_mode="connection" if unified else "full",
        ):
            GraphBuilder().build(
                snapshot,
                store,
                replace_mode="connection" if unified else "full",
                visibility_filter=visibility_filter,
            )
        if _PROFILE_INDEX:
            console.print(
                f"[dim][profile] build (incl. intelligence layer): {time.perf_counter() - _t_build:.2f}s[/dim]"
            )
        if preloaded_manifest is not None:
            apply_dbt_enrichment_cli(
                manifest=preloaded_manifest,
                sources_path=preloaded_sources,
                store=store,
                connection_name=connection_name,
                console=console,
            )
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
    finally:
        store.close()

    try:
        reg = GraphRegistry(sd / cli_constants.REGISTRY_FILENAME).load()
    except Exception as e:
        console.print(
            f"[red]Cannot read registry file:[/red] {e}\n"
            "The registry.json may be corrupt. Delete it and re-run `pretensor index`."
        )
        raise typer.Exit(1) from e
    indexed_at = datetime.now(timezone.utc)
    enc: DSNEncryptor | None = None
    encrypt = keystore_path(sd).exists()
    if encrypt:
        enc = DSNEncryptor(keystore_path(sd))
    reg.upsert(
        connection_name=connection_name,
        database=snapshot.database,
        dsn=dsn,
        graph_path=graph_path,
        unified_graph_path=graph_path if unified else None,
        indexed_at=indexed_at,
        encrypt_dsn=encrypt,
        encryptor=enc,
        dialect=registry_dialect_for(config.type),
        table_count=table_count,
        dbt_manifest_path=str(dbt_manifest) if dbt_manifest is not None else None,
    )
    try:
        reg.save()
    except PermissionError as e:
        console.print(
            f"[red]Cannot write registry file:[/red] {e}\n"
            "Check write permissions on the state directory."
        )
        raise typer.Exit(1) from e

    snap_store = SnapshotStore(sd)
    snap_store.save(connection_name, snapshot)

    entry = reg.get(connection_name)
    if entry is not None:
        skill_store = KuzuStore(graph_path)
        try:
            skill_paths = SkillGenerator.write_for_index(
                store=skill_store,
                entry=entry,
                skills_target=skills_target,
            )
        finally:
            skill_store.close()
        for p in skill_paths:
            console.print(f"[green]Skill written:[/green] {p}")

    console.print(f"[green]Graph written:[/green] {graph_path}")
    console.print(f"[green]Registry updated:[/green] {reg.path}")
    logger.info(
        "index completed in %.2fms",
        (time.perf_counter() - total_started) * 1000,
        extra={
            "event": "index.total",
            "status": "ok",
            "duration_ms": (time.perf_counter() - total_started) * 1000,
            "connection_name": connection_name,
            "unified": unified,
            "table_count": table_count,
        },
    )
