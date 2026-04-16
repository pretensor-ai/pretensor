"""Wire connection-management Typer commands onto the root app."""

from __future__ import annotations

import typer

from pretensor.cli.commands.connections.add_remove import register_add_remove_commands


def register_connection_commands(app: typer.Typer) -> None:
    """Attach ``add`` and ``remove`` to ``app``."""
    register_add_remove_commands(app)
