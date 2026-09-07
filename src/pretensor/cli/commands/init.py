"""``pretensor init`` — interactive first-run setup."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import typer
from rich.console import Console
from ruamel.yaml import YAMLError

from pretensor.cli import constants as cli_constants
from pretensor.cli.commands._command_runners import run_analyze_one, run_index
from pretensor.cli.config_file import get_cli_config, record_repository, record_source
from pretensor.cli.init.inference import (
    DIALECTS,
    InferredDsn,
    InferredRepo,
    count_supported_files,
)
from pretensor.cli.init.mcp_clients import McpClient, register_client
from pretensor.cli.init.steps import (
    ask_add_another_database,
    ask_compute_embeddings,
    choose_clients,
    collect_dsn,
    collect_repository,
    repo_skip_message,
)
from pretensor.cli.init.wizard import InitPlan, build_plan, render_plan
from pretensor.cli.paths import default_connection_name
from pretensor.cli.prompts import Prompter, RichPrompter, is_noninteractive
from pretensor.connectors.registry import get_connector
from pretensor.core.registry import GraphRegistry
from pretensor.intelligence.embeddings import (
    embeddings_extra_installed,
    resolve_embeddings_auto,
)
from pretensor.introspection.models.dsn import connection_config_from_url, redact_dsn
from pretensor.mcp import print_mcp_config

__all__ = ["register_init_command"]

_EXIT_ERROR = 1

_NONINTERACTIVE_GUIDANCE = """\
Setup needs a terminal, and this one is not interactive.

Supply the values as flags instead:

  pretensor init --dsn postgresql://USER:PASSWORD@HOST:5432/DBNAME --yes

Or run the individual commands:

  pretensor index <DSN>
  pretensor serve --config-only
"""

_SAMPLE_GUIDANCE = """\
Sample database setup lives in `pretensor quickstart`.

Run:

  pretensor quickstart

It spins up Docker Postgres, indexes it, and prints the MCP config, all in \
one step.
"""

_NEXT_QUESTIONS = """
Try asking your agent:
  "how do I join orders to customers?"
  "what breaks if I drop users.email?"
  "which services read the payments table?"
"""

# The express summary opens with a three-way choice rather than a literal
# "[Y/n/c]" text prompt: it reuses `Prompter.choose`, the same primitive
# already exercised by `choose_clients`/tests, instead of adding new
# single-letter parsing/validation logic for a shape the protocol doesn't
# have. Options read as full words for clarity; the index is what matters.
_EXPRESS_OPTIONS = (
    "Yes, proceed with everything above",
    "No, exit without writing anything",
    "Customize each step",
)
_EXPRESS_ACCEPT = 0
_EXPRESS_DECLINE = 1

# The keys accepted by --client, and the label to show for each in messages.
_CLIENT_LABELS: dict[str, str] = {
    "claude-code": "Claude Code",
    "claude-desktop": "Claude Desktop",
    "cursor": "Cursor",
}


def register_init_command(app: typer.Typer, *, console: Console) -> None:
    """Attach ``init`` to the root Typer app."""

    @app.command("init")
    def init_command(
        dsn: str | None = typer.Option(None, "--dsn", help="Database URL to index."),
        dialect: str | None = typer.Option(
            None,
            "--dialect",
            help=(
                "Force the database dialect "
                f"({', '.join(DIALECTS)}) instead of inferring it from the DSN."
            ),
        ),
        name: str | None = typer.Option(
            None,
            "--name",
            help="Connection name (default: database name from the DSN).",
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
        repo: Path | None = typer.Option(
            None, "--repo", help="Code repository to link with `analyze`."
        ),
        no_repo: bool = typer.Option(
            False, "--no-repo", help="Skip repository linking."
        ),
        sample: bool = typer.Option(
            False,
            "--sample",
            help="Use the bundled sample database instead of your own.",
        ),
        yes: bool = typer.Option(
            False, "--yes", help="Accept every default and never prompt."
        ),
        no_mcp_write: bool = typer.Option(
            False, "--no-mcp-write", help="Do not write into MCP client configs."
        ),
        client: list[str] = typer.Option(
            [],
            "--client",
            help=(
                "Register with this MCP client explicitly, even under --yes or "
                "a non-interactive run: naming it is the consent that "
                "detecting it alone is not. Repeatable. One of: "
                + ", ".join(_CLIENT_LABELS)
                + "."
            ),
        ),
        ctx: typer.Context = typer.Option(None, hidden=True),
    ) -> None:
        """Set up a database connection, link code, and register with your MCP client."""
        if sample:
            # The sample database is set up entirely by `pretensor quickstart`;
            # there is nothing for `init` to build a plan, render, confirm, or
            # preflight around. Short-circuit before any of that.
            console.print(_SAMPLE_GUIDANCE)
            raise typer.Exit(0)

        if client and no_mcp_write:
            console.print(
                "--client and --no-mcp-write conflict: --no-mcp-write disables "
                "all MCP registration, so naming a client to register with has "
                "no effect."
            )
            raise typer.Exit(_EXIT_ERROR)

        unknown_clients = [key for key in client if key not in _CLIENT_LABELS]
        if unknown_clients:
            console.print(
                f"Unknown --client value(s): {', '.join(unknown_clients)}. "
                f"Valid keys: {', '.join(_CLIENT_LABELS)}."
            )
            raise typer.Exit(_EXIT_ERROR)

        normalized_dialect: str | None = None
        if dialect is not None:
            normalized_dialect = dialect.strip().lower()
            if normalized_dialect not in DIALECTS:
                console.print(
                    f"[red]Invalid dialect:[/red] {dialect!r}. "
                    f"Valid dialects: {', '.join(DIALECTS)}"
                )
                raise typer.Exit(_EXIT_ERROR)

        plan = build_plan(
            env=os.environ,
            cwd=Path.cwd(),
            home=Path.home(),
            state_dir=state_dir,
        )
        if dsn:
            plan.dsn = dsn
            plan.dsn_env_var = None
            if not name:
                # A flag-supplied DSN overrides whatever ``build_plan``
                # inferred the name from (typically an env-var DSN); without
                # this, an explicit --dsn silently indexes under the old
                # (stale) connection name.
                plan.name = default_connection_name(dsn)
        if name:
            plan.name = name
        if repo is not None:
            plan.repo = repo
        if no_repo:
            plan.repo = None
        # --client names the clients the user consents to registering with,
        # among those actually detected. A named key that was not detected
        # gets a warning and falls back to the printed config block, same as
        # an unnamed detected client under --yes.
        requested_clients: tuple[McpClient, ...] = tuple(
            c for c in plan.clients if c.key in client
        )
        detected_keys = {c.key for c in plan.clients}
        for key in client:
            if key not in detected_keys:
                console.print(
                    f"{_CLIENT_LABELS[key]} was not detected; printing the "
                    "config block instead."
                )

        auto_accept = yes or is_noninteractive()
        if no_mcp_write:
            plan.clients = ()
        elif auto_accept:
            # Under --yes / non-interactive, only a client the user
            # explicitly named via --client is consent to write; merely
            # *detecting* one installed is not. Every other detected
            # client falls back to printing the mcpServers block.
            plan.clients = requested_clients

        if plan.dsn is None and auto_accept:
            console.print(_NONINTERACTIVE_GUIDANCE)
            raise typer.Exit(_EXIT_ERROR)

        prompter = RichPrompter(console)

        if not _confirm_existing_state(
            console, prompter, plan.state_dir, auto_accept=auto_accept
        ):
            console.print("Nothing was written.")
            raise typer.Exit(0)

        # ``None`` means "let run_index resolve it automatically" (the
        # unchanged default). It is only ever set to True/False by the
        # guided flow's embeddings question below.
        embeddings: bool | None = None

        if not auto_accept:
            express_choice: int | None = None
            if plan.dsn is not None:
                # Summary-first express confirmation: only offered when
                # inference produced at least a DSN. The plan is rendered
                # once, up front, and the whole interaction is a single
                # three-way choice.
                console.print(render_plan(plan))
                express_choice = prompter.choose(
                    "Proceed with these?", _EXPRESS_OPTIONS
                )

            if express_choice == _EXPRESS_DECLINE:
                console.print("Nothing was written.")
                raise typer.Exit(0)

            if express_choice == _EXPRESS_ACCEPT:
                # Accept everything inferred, no further questions. Named
                # clients (--client) already have explicit flag consent and
                # stay registered; a merely-detected client does not gain
                # consent just because the user said "Y" here, same as the
                # rule --yes already applies -- it falls back to printing
                # the mcpServers block.
                plan.clients = requested_clients
                if plan.repo is not None and count_supported_files(plan.repo) == 0:
                    # Mirrors the same precheck the guided flow already does
                    # below: a repo with nothing to scan (typically supplied
                    # via --repo, since `infer_repo` never offers one of
                    # these) must not count as "linked". Left set, `plan.repo`
                    # would wrongly make the repo-add loop below believe a
                    # first repository had linked and offer to add another
                    # one on top of it.
                    console.print(repo_skip_message(plan.repo))
                    plan.repo = None
            else:
                # express_choice is None (no DSN was inferred, so the
                # summary above was never shown) or the user chose
                # "Customize": walk through the same per-item questions as
                # before, pre-filled with whatever was inferred.
                inferred_dsn = (
                    InferredDsn(dsn=plan.dsn, env_var=plan.dsn_env_var)
                    if plan.dsn is not None
                    else None
                )
                collected = collect_dsn(
                    prompter, inferred_dsn, dialect=normalized_dialect
                )
                plan.dsn = collected.dsn
                plan.dsn_env_var = collected.env_var
                if not name:
                    # Re-derive the name from whatever DSN the user just
                    # confirmed or supplied (declining the inferred DSN and
                    # typing a different one must not keep the old name).
                    plan.name = default_connection_name(plan.dsn)

                if not no_repo:
                    inferred_repo: InferredRepo | None = None
                    if plan.repo is not None:
                        count = count_supported_files(plan.repo)
                        if count == 0:
                            console.print(repo_skip_message(plan.repo))
                            plan.repo = None
                        else:
                            inferred_repo = InferredRepo(
                                path=plan.repo, file_count=count
                            )
                    plan.repo = collect_repository(prompter, inferred_repo, Path.cwd())

                if not no_mcp_write and plan.clients:
                    # Explicitly named clients already have consent
                    # (--client); only ask to confirm the ones the user did
                    # not name.
                    to_confirm = tuple(c for c in plan.clients if c.key not in client)
                    confirmed = (
                        choose_clients(prompter, to_confirm) if to_confirm else ()
                    )
                    plan.clients = requested_clients + confirmed

                console.print(render_plan(plan))
                if not prompter.confirm("Proceed with these?", default=True):
                    console.print("Nothing was written.")
                    raise typer.Exit(0)

                if embeddings_extra_installed():
                    embeddings = ask_compute_embeddings(prompter)
        else:
            console.print(render_plan(plan))

        assert plan.dsn is not None
        connection_name = plan.name or default_connection_name(plan.dsn)
        _execute(
            console=console,
            ctx=ctx,
            plan=plan,
            embeddings=embeddings,
            dialect_override=normalized_dialect,
        )

        if not auto_accept:
            cli_config = get_cli_config(ctx)

            # Repo linking happens once, for the first connection, as
            # today; this only offers to link *additional* repositories
            # against that same connection. Unlike the first repository
            # (whose failure propagates via `typer.Exit` inside `_execute`),
            # a loop-linked repository's failure can't raise mid-loop
            # without also cutting off further "link another?" and
            # "add another database?" offers -- so track the worst exit
            # code seen and raise once every loop has finished, mirroring
            # `_execute`'s own fail-after-finishing contract for analyze.
            repo_loop_exit_code = 0
            if plan.repo is not None:
                while True:
                    extra_repo = collect_repository(prompter, None, Path.cwd())
                    if extra_repo is None:
                        break
                    linked, code = _link_repository(
                        console=console,
                        repo_path=extra_repo,
                        connection_name=connection_name,
                        state_dir=plan.state_dir,
                    )
                    if linked is not None:
                        console.print(
                            f"[bold green]Linked code[/bold green] from {linked}."
                        )
                    elif code != 0:
                        repo_loop_exit_code = max(repo_loop_exit_code, code)

            while ask_add_another_database(prompter):
                collected = collect_dsn(prompter, None, dialect=normalized_dialect)
                extra_connection_name = _connect_and_index(
                    console=console,
                    cli_config=cli_config,
                    dsn=collected.dsn,
                    name=default_connection_name(collected.dsn),
                    state_dir=plan.state_dir,
                    embeddings=embeddings,
                    dialect_override=normalized_dialect,
                )
                console.print(
                    f"\n[bold green]Indexed[/bold green] `{extra_connection_name}`."
                )

            if repo_loop_exit_code != 0:
                raise typer.Exit(repo_loop_exit_code)


def _confirm_existing_state(
    console: Console,
    prompter: Prompter,
    state_dir: Path,
    *,
    auto_accept: bool,
) -> bool:
    """Summarize any existing registry state and confirm before proceeding.

    Returns False when the user declines and the caller should stop without
    writing anything. Under ``--yes``/non-interactive, the summary is still
    printed but no confirmation is asked.
    """
    registry_path = state_dir / cli_constants.REGISTRY_FILENAME
    if not registry_path.exists():
        return True
    try:
        reg = GraphRegistry(registry_path).load()
    except Exception:
        # A corrupt registry is not this preflight's problem to solve; let
        # the normal index/analyze flow surface it.
        return True
    entries = reg.list_entries()
    if not entries:
        return True

    console.print("[bold]Existing connections:[/bold]")
    for entry in entries:
        console.print(f"  {entry.connection_name} ({entry.database}, {entry.dialect})")

    if auto_accept:
        return True
    return prompter.confirm("Continue and add or update a connection?", default=True)


def _connect_and_index(
    *,
    console: Console,
    cli_config: object,
    dsn: str,
    name: str | None,
    state_dir: Path,
    embeddings: bool | None,
    dialect_override: str | None = None,
) -> str:
    """Test connectivity and index one connection. Returns its connection name.

    Shared by `_execute` (the first connection) and the interactive
    "Add another database?" loop in `init_command` (every connection after
    it), so both go through the same connect/index path.
    """
    connection_name = name or default_connection_name(dsn)
    try:
        config = connection_config_from_url(
            dsn, connection_name, dialect_override=dialect_override
        )
    except ValueError as e:
        console.print(f"[red]Invalid DSN:[/red] {e}")
        raise typer.Exit(_EXIT_ERROR) from e

    with console.status(f"Testing connection to {redact_dsn(dsn)}..."):
        try:
            connector = get_connector(config)
            connector.connect()
        except Exception as e:
            console.print(f"[red]Cannot connect:[/red] {redact_dsn(dsn)}\n{e}")
            raise typer.Exit(_EXIT_ERROR) from e
        connector.disconnect()

    run_index(
        console=console,
        cli_config=cli_config,
        dsn=dsn,
        connection_name=connection_name,
        config=config,
        state_dir=state_dir,
        unified=False,
        # AUTO by default, same as `pretensor index`: on iff the
        # [embeddings] extra is installed and the kill switch is unset. The
        # guided flow's embeddings question overrides this explicitly when
        # asked (see `ask_compute_embeddings`).
        embeddings=embeddings
        if embeddings is not None
        else resolve_embeddings_auto(None),
        skills_target="claude",
        visibility_file=None,
        profile=None,
        dbt_manifest=None,
        dbt_sources=None,
    )
    return connection_name


def _link_repository(
    *,
    console: Console,
    repo_path: Path,
    connection_name: str,
    state_dir: Path,
) -> tuple[Path | None, int]:
    """Analyze *repo_path* against *connection_name* and record it if linked.

    Returns ``(repo_path, 0)`` when linking succeeded, ``(None, 0)`` when
    there was nothing to scan (not a failure), and ``(None, exit_code)``
    when analysis failed. Shared by `_execute` (the first repository) and
    the "Link another code repository?" loop in `init_command`.
    """
    if count_supported_files(repo_path) == 0:
        console.print(repo_skip_message(repo_path))
        return None, 0

    exit_code = run_analyze_one(
        console=console,
        repo_path=repo_path,
        connection=connection_name,
        service=None,
        state_dir=state_dir,
        graph_dir=None,
        include=[],
        exclude=[],
        max_file_bytes=1_000_000,
        min_confidence=0.6,
        default_schema=None,
        dry_run=False,
        as_json=False,
    )
    if exit_code != 0:
        console.print(
            "[yellow]Repository linking failed.[/yellow] Retry with:\n"
            f"  pretensor analyze {repo_path} --connection {connection_name}"
        )
        return None, exit_code

    config_path = state_dir / "config.yaml"
    try:
        record_repository(
            config_path,
            path=repo_path,
            service=repo_path.name,
            connection=connection_name,
        )
    except (YAMLError, OSError, ValueError) as e:
        # Indexing and analysis already succeeded; a config.yaml we can't
        # safely rewrite must not fail the command.
        console.print(
            f"Could not record the repository in {config_path}: {e}. "
            "Link it later with:\n"
            f"  pretensor analyze {repo_path} --connection {connection_name}"
        )
    return repo_path, exit_code


def _execute(
    *,
    console: Console,
    ctx: typer.Context | None,
    plan: InitPlan,
    embeddings: bool | None = None,
    dialect_override: str | None = None,
) -> None:
    """Run the planned actions.

    Indexing must precede analysis: ``run_analyze_enrichment`` raises
    ``EmptyGraphError`` when the connection has no ``SchemaTable`` rows.
    """
    assert plan.dsn is not None
    cli_config = get_cli_config(ctx)

    # 1. Index -----------------------------------------------------------
    connection_name = _connect_and_index(
        console=console,
        cli_config=cli_config,
        dsn=plan.dsn,
        name=plan.name,
        state_dir=plan.state_dir,
        embeddings=embeddings,
        dialect_override=dialect_override,
    )

    # 1b. Record env-sourced DSN reference --------------------------------
    # Only when the DSN came from a single environment variable (tracked as
    # `plan.dsn_env_var`): the literal `${VAR}` reference is safe to commit,
    # so `sources:` gets it. A typed/pasted/assembled DSN has no single env
    # var to reference; the encrypted registry entry from indexing above
    # already covers that case, so nothing is written to `sources:`.
    if plan.dsn_env_var:
        config_path = plan.state_dir / "config.yaml"
        url_reference = f"${{{plan.dsn_env_var}}}"
        try:
            record_source(
                config_path,
                name=connection_name,
                url_reference=url_reference,
            )
        except (YAMLError, OSError, ValueError) as e:
            # Indexing already succeeded; a config.yaml we can't safely
            # rewrite must not fail the command.
            console.print(
                f"Could not record the source in {config_path}: {e}. "
                "Add it later under `sources:`:\n"
                f"  {connection_name}:\n"
                f'    url: "{url_reference}"'
            )

    # 2. Analyze -----------------------------------------------------------
    analyzed_repo: Path | None = None
    analyze_exit_code = 0
    if plan.repo is not None:
        analyzed_repo, analyze_exit_code = _link_repository(
            console=console,
            repo_path=plan.repo,
            connection_name=connection_name,
            state_dir=plan.state_dir,
        )

    # 3. Register ------------------------------------------------------------
    register_lines: list[str] = []
    registration_failed = False
    if plan.clients:
        for client in plan.clients:
            try:
                register_lines.append(register_client(client, plan.state_dir))
            except (ValueError, json.JSONDecodeError, OSError) as e:
                registration_failed = True
                console.print(f"{client.label}: {e}")
    else:
        console.print("\n[bold]MCP server config[/bold] (paste into your client):")
        print_mcp_config(plan.state_dir)

    # 4. Summary -----------------------------------------------------------
    _print_summary(
        console,
        connection_name=connection_name,
        repo=analyzed_repo,
        register_lines=register_lines,
    )

    if analyze_exit_code != 0:
        # Indexing (and registration) succeeded; only analyze failed. Report
        # that via the exit code without undoing the work already done.
        raise typer.Exit(analyze_exit_code)
    if registration_failed:
        raise typer.Exit(_EXIT_ERROR)


def _print_summary(
    console: Console,
    *,
    connection_name: str,
    repo: Path | None,
    register_lines: Sequence[str],
) -> None:
    console.print()
    console.print(f"[bold green]Indexed[/bold green] `{connection_name}`.")
    if repo is not None:
        console.print(f"[bold green]Linked code[/bold green] from {repo}.")
    for line in register_lines:
        console.print(f"[bold green]{line}[/bold green]")
    console.print(_NEXT_QUESTIONS)
