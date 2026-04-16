"""Unit tests for CLI plugin discovery (pretensor.cli.plugin)."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import Any
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from pretensor.cli.plugin import discover_cli_plugins

_runner = CliRunner()


def _make_ep(name: str, load_result: Any) -> MagicMock:
    ep = MagicMock(spec=EntryPoint)
    ep.name = name
    ep.load.return_value = load_result
    return ep


def _make_failing_ep(name: str) -> MagicMock:
    ep = MagicMock(spec=EntryPoint)
    ep.name = name
    ep.load.side_effect = ImportError("no such module")
    return ep


def test_no_plugins_installed() -> None:
    """When the entry_point group is empty, discover_cli_plugins is a no-op."""
    app = typer.Typer(no_args_is_help=True, add_completion=False)

    @app.command("dummy")
    def _dummy() -> None:
        """Dummy command."""

    with patch("pretensor.cli.plugin.importlib.metadata.entry_points", return_value=[]):
        discover_cli_plugins(app)

    result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dummy" in result.stdout


def test_plugin_register_called_with_app() -> None:
    """Each discovered entry point's register() is invoked with the Typer app."""
    app = typer.Typer()
    register_fn = MagicMock()
    ep = _make_ep("my-plugin", register_fn)

    with patch("pretensor.cli.plugin.importlib.metadata.entry_points", return_value=[ep]):
        discover_cli_plugins(app)

    register_fn.assert_called_once_with(app)


def test_plugin_adds_command() -> None:
    """A plugin that adds a Typer sub-command is visible in --help."""
    app = typer.Typer(no_args_is_help=True, add_completion=False)

    def register(a: typer.Typer) -> None:
        @a.command("plugin-cmd")
        def _cmd() -> None:
            """Plugin command."""

    ep = _make_ep("cloud-plugin", register)

    with patch("pretensor.cli.plugin.importlib.metadata.entry_points", return_value=[ep]):
        discover_cli_plugins(app)

    result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "plugin-cmd" in result.stdout


def test_broken_load_is_skipped(caplog: Any) -> None:
    """An entry point that fails to load must not crash the CLI."""
    app = typer.Typer()
    ep = _make_failing_ep("broken-plugin")

    with patch("pretensor.cli.plugin.importlib.metadata.entry_points", return_value=[ep]):
        discover_cli_plugins(app)

    assert any("broken-plugin" in r.message for r in caplog.records)


def test_broken_register_is_skipped(caplog: Any) -> None:
    """An entry point whose register() raises must not crash the CLI."""
    app = typer.Typer()

    def bad_register(a: typer.Typer) -> None:
        _ = a
        raise RuntimeError("registration failed")

    ep = _make_ep("bad-register-plugin", bad_register)

    with patch("pretensor.cli.plugin.importlib.metadata.entry_points", return_value=[ep]):
        discover_cli_plugins(app)

    assert any("bad-register-plugin" in r.message for r in caplog.records)


def test_multiple_plugins_all_registered() -> None:
    """All plugins in the group are registered, not just the first."""
    app = typer.Typer(no_args_is_help=True, add_completion=False)

    calls: list[str] = []

    def make_register(label: str):  # type: ignore[return]
        def register(a: typer.Typer) -> None:
            _ = a
            calls.append(label)

        return register

    eps = [_make_ep(f"plugin-{i}", make_register(f"plugin-{i}")) for i in range(3)]

    with patch("pretensor.cli.plugin.importlib.metadata.entry_points", return_value=eps):
        discover_cli_plugins(app)

    assert calls == ["plugin-0", "plugin-1", "plugin-2"]


def test_main_app_has_no_plugin_commands_by_default() -> None:
    """With no plugins installed, ``pretensor --help`` shows only built-in commands."""
    from pretensor.cli.main import app as main_app

    with patch("pretensor.cli.plugin.importlib.metadata.entry_points", return_value=[]):
        result = _runner.invoke(main_app, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.stdout
