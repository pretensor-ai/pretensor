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

    for cmd in (
        "benchmark",
        "index",
        "reindex",
        "list",
        "export",
        "serve",
        "add",
        "remove",
    ):
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


def test_benchmark_subcommand_in_help() -> None:
    """``pretensor benchmark --help`` returns exit 0."""
    result = CliRunner().invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0


def test_root_help_includes_logging_flags() -> None:
    """Root help documents global logging flags."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--log-level" in plain
    assert "--log-format" in plain
    assert "--log-file" in plain


def test_short_help_flag() -> None:
    """`pretensor -h` is equivalent to --help."""
    result = CliRunner().invoke(app, ["-h"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--log-level" in plain


def test_default_log_level_is_warning() -> None:
    """--log-level defaults to 'warning' in the help text."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "warning" in result.stdout.lower()


def test_version_flag_prints_version() -> None:
    """``pretensor --version`` prints the installed package version and exits 0."""
    from importlib.metadata import version as pkg_version

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "pretensor" in result.stdout.lower()
    assert pkg_version("pretensor") in result.stdout


def test_no_args_hint_shown_when_nothing_indexed(monkeypatch, tmp_path) -> None:
    """Bare ``pretensor`` prints the fresh-install hint when nothing is set up."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, [])
    assert "No connections indexed yet" in result.stdout
    assert "pretensor init" in result.stdout
    assert result.exit_code == 2


def test_no_args_hint_hidden_when_state_dir_exists(monkeypatch, tmp_path) -> None:
    """The hint never prints once a state dir already exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".pretensor").mkdir()
    result = CliRunner().invoke(app, [])
    assert "No connections indexed yet" not in result.stdout
    assert result.exit_code == 2


def test_no_args_with_malformed_config_still_shows_help(monkeypatch, tmp_path) -> None:
    """A broken ``.pretensor/config.yaml`` must not block a bare invocation.

    Bare ``pretensor`` never reaches ``load_cli_config`` (no subcommand is
    going to run), so a malformed config file must not surface a config
    error; it should show help and exit with the same code as a normal bare
    invocation. Since the config file's existence means a state dir already
    exists, the fresh-install hint must not print either.
    """
    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / ".pretensor"
    state_dir.mkdir()
    (state_dir / "config.yaml").write_text("just a plain string, not a mapping\n")

    result = CliRunner().invoke(app, [])

    # Same exit code as a normal bare invocation (see the two tests above).
    assert result.exit_code == 2
    assert "No connections indexed yet" not in result.stdout
    assert "Config file root must be a mapping" not in result.stdout
    assert "Invalid YAML" not in result.stdout
    assert "Traceback" not in result.stdout
    assert "commands" in result.stdout.lower()


def test_root_help_includes_version_flag() -> None:
    """``pretensor --help`` documents --version."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--version" in _normalize(result.stdout)
