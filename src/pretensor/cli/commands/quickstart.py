"""``pretensor quickstart`` — clone-to-running in one command."""

from __future__ import annotations

import shutil
import subprocess
import time
from importlib.resources import files
from pathlib import Path

import typer
from rich.console import Console

from pretensor.cli import constants as cli_constants
from pretensor.cli.commands.index import _run_index
from pretensor.cli.config_file import get_cli_config
from pretensor.introspection.models.dsn import connection_config_from_url
from pretensor.mcp import print_mcp_config

QUICKSTART_DSN = "postgresql://postgres:postgres@localhost:55432/pagila"
QUICKSTART_NAME = "pagila"
HEALTH_TIMEOUT_SECONDS = 60


def _compose_path() -> Path:
    # The compose file ships inside the wheel as a package data file alongside
    # the pagila SQL fixtures it mounts. Resolving via ``importlib.resources``
    # works in editable installs and in pip-installed wheels alike.
    return Path(str(files("pretensor.quickstart") / "docker-compose.yml"))


_QUICKSTART_ASSET_NAMES = ("docker-compose.yml", "pagila_ddl.sql", "pagila_data.sql")


def _stage_compose_assets(state_dir: Path) -> Path:
    # Snap-confined Docker (the default `snap install docker` on Ubuntu)
    # can't read files under hidden directories in $HOME. A `pipx install`
    # puts the package under ~/.local/pipx/venvs/..., so invoking compose
    # straight against the site-packages copy fails there. Copying the
    # compose file and SQL fixtures next to the state dir sidesteps that
    # (and makes the compose file user-inspectable).
    src_dir = files("pretensor.quickstart")
    dest_dir = state_dir / "quickstart"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in _QUICKSTART_ASSET_NAMES:
        (dest_dir / name).write_bytes((src_dir / name).read_bytes())
    return dest_dir / "docker-compose.yml"


def _wait_healthy(console: Console, container: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Health.Status}}", container],
            capture_output=True,
            text=True,
        )
        status = proc.stdout.strip()
        if status == "healthy":
            return
        if proc.returncode != 0 and "No such" in proc.stderr:
            raise RuntimeError(f"container {container} not running")
        time.sleep(1)
    raise RuntimeError(
        f"container {container} did not become healthy within {timeout}s"
    )


def register_quickstart_command(app: typer.Typer, *, console: Console) -> None:
    @app.command("quickstart")
    def quickstart_command(
        no_docker: bool = typer.Option(
            False,
            "--no-docker",
            help=(
                "Skip the docker compose step; assume Postgres is already "
                f"reachable at {QUICKSTART_DSN}."
            ),
        ),
        down: bool = typer.Option(
            False,
            "--down",
            help="Tear down the quickstart container and exit.",
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
        ctx: typer.Context = typer.Option(None, hidden=True),
    ) -> None:
        """Spin up a sample Postgres, index it, and print MCP config — one command."""
        cli_config = get_cli_config(ctx)
        compose_file = _compose_path()

        if down:
            staged_compose = _stage_compose_assets(state_dir)
            _compose_down(console, staged_compose)
            return

        if not no_docker:
            if shutil.which("docker") is None:
                console.print(
                    "[red]`docker` is not on PATH.[/red]\n"
                    "Install Docker Desktop, or pass --no-docker if you have "
                    "Postgres reachable at the quickstart DSN already."
                )
                raise typer.Exit(1)
            if not compose_file.is_file():
                console.print(
                    f"[red]Compose file not found:[/red] {compose_file}\n"
                    "Run `pretensor quickstart` from a clone of the pretensor repo."
                )
                raise typer.Exit(1)
            staged_compose = _stage_compose_assets(state_dir)
            _compose_up(console, staged_compose)
            try:
                _wait_healthy(
                    console, "pretensor-quickstart-db", HEALTH_TIMEOUT_SECONDS
                )
            except RuntimeError as e:
                console.print(f"[red]Postgres did not start:[/red] {e}")
                raise typer.Exit(1) from e
            console.print("[green]Postgres is up.[/green]")

        try:
            config = connection_config_from_url(QUICKSTART_DSN, QUICKSTART_NAME)
        except ValueError as e:
            console.print(f"[red]Invalid quickstart DSN:[/red] {e}")
            raise typer.Exit(1) from e

        from pretensor.intelligence.embeddings import resolve_embeddings_auto

        _run_index(
            console=console,
            cli_config=cli_config,
            dsn=QUICKSTART_DSN,
            connection_name=QUICKSTART_NAME,
            config=config,
            state_dir=state_dir,
            unified=False,
            # AUTO, same as `pretensor index`: on iff the [embeddings]
            # extra is installed and the kill switch is unset.
            embeddings=resolve_embeddings_auto(None),
            skills_target="claude",
            visibility_file=None,
            profile=None,
            dbt_manifest=None,
            dbt_sources=None,
        )

        console.print("\n[bold]MCP server config[/bold] (paste into your client):")
        print_mcp_config(state_dir)
        console.print(
            "\n[bold green]Done.[/bold green] Try a query: "
            f"`pretensor list` or start the server with `pretensor serve "
            f"--state-dir {state_dir}`."
        )


def _print_permission_hint_if_relevant(console: Console, output: str) -> None:
    if "permission denied" in output.lower():
        console.print(
            "[yellow]Hint:[/yellow] snap-confined Docker (the default "
            "`snap install docker` on Ubuntu) can't read files under hidden "
            "directories in $HOME. That includes the staged state dir when "
            "quickstart runs from $HOME itself (~/.pretensor/...) or from "
            "any dotfile-prefixed path. Rerun from a non-hidden project "
            "directory (e.g. `mkdir ~/pretensor-demo && cd ~/pretensor-demo`), "
            "or install Docker from apt/a PPA instead of the snap."
        )


def _compose_up(console: Console, compose_file: Path) -> None:
    console.print(f"[bold blue]Starting Postgres[/bold blue] via {compose_file}...")
    proc = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "up", "-d"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        output = proc.stderr or proc.stdout
        console.print(f"[red]docker compose up failed:[/red]\n{output}")
        _print_permission_hint_if_relevant(console, output)
        raise typer.Exit(1)


def _compose_down(console: Console, compose_file: Path) -> None:
    if not compose_file.is_file():
        console.print(f"[yellow]Compose file not found:[/yellow] {compose_file}")
        raise typer.Exit(1)
    console.print(f"[bold blue]Tearing down[/bold blue] {compose_file}...")
    proc = subprocess.run(
        ["docker", "compose", "-f", str(compose_file), "down", "-v"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        output = proc.stderr or proc.stdout
        console.print(f"[red]docker compose down failed:[/red]\n{output}")
        _print_permission_hint_if_relevant(console, output)
        raise typer.Exit(1)
    console.print("[green]Quickstart container removed.[/green]")
