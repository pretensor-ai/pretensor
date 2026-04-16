"""CLI tests for the root ``pretensor`` Typer app (main.py)."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from pretensor.cli.main import app

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _ANSI_ESCAPE_RE.sub("", text)).strip()


def test_root_help_lists_all_commands() -> None:
    """``pretensor --help`` shows all built-in commands."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)

    for cmd in ("index", "reindex", "list", "export", "serve", "add", "remove"):
        assert cmd in plain, f"Command {cmd!r} missing from --help"


def test_root_no_args_shows_help() -> None:
    """Invoking ``pretensor`` with no arguments shows help text (no crash)."""
    result = CliRunner().invoke(app, [])
    # no_args_is_help=True causes Typer to print help and exit 0 (newer) or 2 (older).
    assert result.exit_code in (0, 2)
    assert "pretensor" in result.stdout.lower() or "commands" in result.stdout.lower()


def test_unknown_command_exits_nonzero() -> None:
    """An unknown subcommand causes a non-zero exit."""
    result = CliRunner().invoke(app, ["does-not-exist"])
    assert result.exit_code != 0


def test_index_subcommand_in_help() -> None:
    """``pretensor index --help`` returns exit 0."""
    result = CliRunner().invoke(app, ["index", "--help"])
    assert result.exit_code == 0


def test_list_subcommand_in_help() -> None:
    """``pretensor list --help`` returns exit 0."""
    result = CliRunner().invoke(app, ["list", "--help"])
    assert result.exit_code == 0


def test_reindex_subcommand_in_help() -> None:
    """``pretensor reindex --help`` returns exit 0."""
    result = CliRunner().invoke(app, ["reindex", "--help"])
    assert result.exit_code == 0


def test_add_subcommand_in_help() -> None:
    """``pretensor add --help`` returns exit 0."""
    result = CliRunner().invoke(app, ["add", "--help"])
    assert result.exit_code == 0


def test_remove_subcommand_in_help() -> None:
    """``pretensor remove --help`` returns exit 0."""
    result = CliRunner().invoke(app, ["remove", "--help"])
    assert result.exit_code == 0


def test_sync_grants_subcommand_in_help() -> None:
    """``pretensor sync-grants --help`` returns exit 0."""
    result = CliRunner().invoke(app, ["sync-grants", "--help"])
    assert result.exit_code == 0


def test_export_subcommand_in_help() -> None:
    """``pretensor export --help`` returns exit 0."""
    result = CliRunner().invoke(app, ["export", "--help"])
    assert result.exit_code == 0


def test_root_help_includes_logging_flags() -> None:
    """Root help documents global logging flags."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--log-level" in plain
    assert "--log-format" in plain
    assert "--log-file" in plain
