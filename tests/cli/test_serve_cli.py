"""CLI tests for ``pretensor serve`` command."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from pretensor.cli.main import app

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _ANSI_ESCAPE_RE.sub("", text)).strip()


def test_serve_config_only_outputs_valid_json(tmp_path: Path) -> None:
    """``--config-only`` prints mcpServers JSON to stdout and exits 0."""
    result = CliRunner().invoke(
        app, ["serve", "--config-only", "--graph-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert "mcpServers" in data
    assert "pretensor" in data["mcpServers"]


def test_serve_config_only_includes_graph_dir(tmp_path: Path) -> None:
    """The ``--config-only`` JSON embeds the resolved graph-dir in the args list."""
    result = CliRunner().invoke(
        app, ["serve", "--config-only", "--graph-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    args: list[Any] = data["mcpServers"]["pretensor"]["args"]
    assert "--graph-dir" in args
    graph_dir_idx = args.index("--graph-dir")
    assert args[graph_dir_idx + 1] == str(tmp_path.resolve())


def test_serve_help_shows_options() -> None:
    """``pretensor serve --help`` is available and shows key flags."""
    result = CliRunner().invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--graph-dir" in plain
    assert "--config-only" in plain
    assert "--visibility" in plain
    assert "--profile" in plain


def test_serve_starts_server_without_config_only(tmp_path: Path) -> None:
    """Without ``--config-only``, ``run_server`` is called once."""
    with (
        patch("pretensor.cli.commands.serve.run_server") as mock_run,
        patch("pretensor.cli.commands.serve.print_mcp_config"),
    ):
        result = CliRunner().invoke(
            app,
            ["serve", "--no-print-config", "--graph-dir", str(tmp_path)],
        )
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_serve_passes_visibility_path(tmp_path: Path) -> None:
    """``--visibility`` is forwarded to ``run_server`` as ``visibility_path``."""
    vis_file = tmp_path / "vis.yml"
    vis_file.write_text("", encoding="utf-8")

    captured: dict[str, Any] = {}

    def _fake_run_server(
        graph_dir: Path, *, visibility_path: Any, profile: Any, config: Any
    ) -> None:
        captured["visibility_path"] = visibility_path
        captured["profile"] = profile
        captured["config"] = config

    with (
        patch(
            "pretensor.cli.commands.serve.run_server", side_effect=_fake_run_server
        ),
        patch("pretensor.cli.commands.serve.print_mcp_config"),
    ):
        result = CliRunner().invoke(
            app,
            [
                "serve",
                "--no-print-config",
                "--graph-dir",
                str(tmp_path),
                "--visibility",
                str(vis_file),
            ],
        )
    assert result.exit_code == 0
    assert captured["visibility_path"] is not None
    assert captured["config"] is not None


def test_serve_value_error_exits_1(tmp_path: Path) -> None:
    """A ``ValueError`` from ``run_server`` prints an error and exits with code 1."""
    with (
        patch(
            "pretensor.cli.commands.serve.run_server",
            side_effect=ValueError("bad config"),
        ),
        patch("pretensor.cli.commands.serve.print_mcp_config"),
    ):
        result = CliRunner().invoke(
            app,
            ["serve", "--no-print-config", "--graph-dir", str(tmp_path)],
        )
    assert result.exit_code == 1
    assert "bad config" in _normalize(result.stdout)


def test_serve_profile_empty_string_normalised(tmp_path: Path) -> None:
    """An empty string ``--profile`` is treated as None (no profile)."""
    captured: dict[str, Any] = {}

    def _fake_run_server(
        graph_dir: Path, *, visibility_path: Any, profile: Any, config: Any
    ) -> None:
        captured["profile"] = profile
        captured["config"] = config

    with (
        patch(
            "pretensor.cli.commands.serve.run_server", side_effect=_fake_run_server
        ),
        patch("pretensor.cli.commands.serve.print_mcp_config"),
    ):
        result = CliRunner().invoke(
            app,
            [
                "serve",
                "--no-print-config",
                "--graph-dir",
                str(tmp_path),
                "--profile",
                "  ",
            ],
        )
    assert result.exit_code == 0
    assert captured["profile"] is None
    assert captured["config"] is not None


def test_serve_print_config_writes_to_stderr(tmp_path: Path) -> None:
    """When ``--print-config`` is active, ``print_mcp_config`` is called."""
    with (
        patch("pretensor.cli.commands.serve.run_server"),
        patch("pretensor.cli.commands.serve.print_mcp_config") as mock_print,
    ):
        result = CliRunner().invoke(
            app,
            ["serve", "--print-config", "--graph-dir", str(tmp_path)],
        )
    assert result.exit_code == 0
    mock_print.assert_called_once()


def test_serve_no_print_config_skips_stderr(tmp_path: Path) -> None:
    """When ``--no-print-config``, ``print_mcp_config`` is NOT called."""
    with (
        patch("pretensor.cli.commands.serve.run_server"),
        patch("pretensor.cli.commands.serve.print_mcp_config") as mock_print,
    ):
        CliRunner().invoke(
            app,
            ["serve", "--no-print-config", "--graph-dir", str(tmp_path)],
        )
    mock_print.assert_not_called()
