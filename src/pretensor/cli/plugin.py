"""CLI plugin discovery via Python entry_points.

Plugins declare themselves under the ``pretensor.cli_plugins`` entry-point
group in their ``pyproject.toml``::

    [project.entry-points."pretensor.cli_plugins"]
    my-plugin = "my_package.cli_plugin:register"

Each entry point must be a callable with the signature::

    def register(app: typer.Typer) -> None: ...
"""

from __future__ import annotations

import importlib.metadata
import logging

import typer

__all__ = ["discover_cli_plugins"]

_ENTRY_POINT_GROUP = "pretensor.cli_plugins"

logger = logging.getLogger(__name__)


def discover_cli_plugins(app: typer.Typer) -> None:
    """Scan the ``pretensor.cli_plugins`` entry-point group and register commands.

    Each entry point must resolve to a callable that accepts a single
    ``typer.Typer`` argument.  Errors during load or registration are logged
    and skipped so a broken plugin never prevents the CLI from starting.

    Args:
        app: The root Typer application to register plugin commands into.
    """
    eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
    for ep in eps:
        try:
            register = ep.load()
        except Exception:
            logger.warning(
                "pretensor: failed to load CLI plugin %r — skipping", ep.name, exc_info=True
            )
            continue
        try:
            register(app)
        except Exception:
            logger.warning(
                "pretensor: failed to register CLI plugin %r — skipping", ep.name, exc_info=True
            )
