"""Express-path assembly and step sequencing for ``pretensor init``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pretensor.cli.init.inference import infer_dsn_from_env, infer_repo
from pretensor.cli.init.mcp_clients import McpClient, detect_clients
from pretensor.cli.paths import default_connection_name
from pretensor.introspection.models.dsn import redact_dsn

__all__ = ["InitPlan", "build_plan", "render_plan"]


@dataclass(slots=True)
class InitPlan:
    """Everything the wizard intends to do, before it does any of it."""

    dsn: str | None
    dsn_env_var: str | None
    name: str | None
    state_dir: Path
    repo: Path | None
    clients: tuple[McpClient, ...]


def build_plan(
    *,
    env: Mapping[str, str],
    cwd: Path,
    home: Path,
    state_dir: Path,
) -> InitPlan:
    """Infer as much of the setup as possible without asking anything."""
    found_dsn = infer_dsn_from_env(env)
    found_repo = infer_repo(cwd)
    return InitPlan(
        dsn=found_dsn.dsn if found_dsn else None,
        dsn_env_var=found_dsn.env_var if found_dsn else None,
        name=default_connection_name(found_dsn.dsn) if found_dsn else None,
        state_dir=state_dir,
        repo=found_repo.path if found_repo else None,
        clients=detect_clients(home=home, cwd=cwd),
    )


def render_plan(plan: InitPlan) -> str:
    """Render the express-path summary. Never shows a password."""
    rows: list[tuple[str, str]] = []
    if plan.dsn:
        source = f"   (from {plan.dsn_env_var})" if plan.dsn_env_var else ""
        rows.append(("Database", f"{redact_dsn(plan.dsn)}{source}"))
    if plan.name:
        rows.append(("Name", plan.name))
    if plan.repo:
        rows.append(("Code repo", str(plan.repo)))
    if plan.clients:
        rows.append(("MCP client", ", ".join(c.label for c in plan.clients)))
    rows.append(("State dir", str(plan.state_dir)))
    return "\n".join(f"  {label:<12}{value}" for label, value in rows)
