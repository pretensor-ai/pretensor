"""CLI tests for ``pretensor quickstart``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from pretensor.cli.main import app

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _ANSI_ESCAPE_RE.sub("", text)).strip()


def test_quickstart_help_lists_flags() -> None:
    result = CliRunner().invoke(app, ["quickstart", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--no-docker" in plain
    assert "--down" in plain
    assert "--state-dir" in plain


def test_quickstart_no_docker_skips_compose_and_indexes(tmp_path: Path) -> None:
    """``--no-docker`` runs no docker commands and calls _run_index once."""
    with (
        patch("pretensor.cli.commands.quickstart.subprocess.run") as mock_sub,
        patch("pretensor.cli.commands.quickstart._run_index") as mock_idx,
        patch("pretensor.cli.commands.quickstart.print_mcp_config"),
    ):
        result = CliRunner().invoke(
            app,
            ["quickstart", "--no-docker", "--state-dir", str(tmp_path)],
        )

    assert result.exit_code == 0, result.stdout
    mock_sub.assert_not_called()
    mock_idx.assert_called_once()
    kwargs = mock_idx.call_args.kwargs
    assert kwargs["dsn"] == "postgresql://postgres:postgres@localhost:55432/pagila"
    assert kwargs["connection_name"] == "pagila"
    assert kwargs["state_dir"] == tmp_path


def test_quickstart_down_invokes_compose_down(tmp_path: Path) -> None:
    """``--down`` calls ``docker compose down -v`` and skips indexing."""

    def _fake_run(args: list[str], **_: Any) -> MagicMock:
        m = MagicMock()
        m.returncode = 0
        m.stdout = ""
        m.stderr = ""
        return m

    with (
        patch(
            "pretensor.cli.commands.quickstart.subprocess.run",
            side_effect=_fake_run,
        ) as mock_sub,
        patch("pretensor.cli.commands.quickstart._run_index") as mock_idx,
    ):
        result = CliRunner().invoke(
            app, ["quickstart", "--down", "--state-dir", str(tmp_path)]
        )

    assert result.exit_code == 0, result.stdout
    mock_idx.assert_not_called()
    cmd = mock_sub.call_args_list[0].args[0]
    assert cmd[:3] == ["docker", "compose", "-f"]
    assert cmd[-2:] == ["down", "-v"]


def test_quickstart_errors_when_docker_missing(tmp_path: Path) -> None:
    """Without docker on PATH the command exits 1 with a helpful message."""
    with (
        patch("pretensor.cli.commands.quickstart.shutil.which", return_value=None),
        patch("pretensor.cli.commands.quickstart._run_index"),
    ):
        result = CliRunner().invoke(
            app, ["quickstart", "--state-dir", str(tmp_path)]
        )
    assert result.exit_code == 1
    assert "docker" in _normalize(result.stdout).lower()


def test_quickstart_prints_mcp_config_after_index(tmp_path: Path) -> None:
    """``print_mcp_config`` is called once with the state-dir root."""
    with (
        patch("pretensor.cli.commands.quickstart.subprocess.run"),
        patch("pretensor.cli.commands.quickstart._run_index"),
        patch(
            "pretensor.cli.commands.quickstart.print_mcp_config"
        ) as mock_print,
    ):
        result = CliRunner().invoke(
            app, ["quickstart", "--no-docker", "--state-dir", str(tmp_path)]
        )

    assert result.exit_code == 0, result.stdout
    mock_print.assert_called_once()
    (graph_dir,) = mock_print.call_args.args
    assert graph_dir == tmp_path
