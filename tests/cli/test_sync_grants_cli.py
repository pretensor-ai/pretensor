"""CLI-level tests for ``pretensor sync-grants`` (F1 --name, F4 empty-grants warn)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from typer.testing import CliRunner

from pretensor.cli.main import app

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _normalize(text: str) -> str:
    """Collapse whitespace so assertions survive Rich's terminal-width wrapping."""
    return _WHITESPACE_RE.sub(" ", _strip_ansi(text)).strip()


def test_sync_grants_name_flag_overrides_derived_connection_name(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """F1: ``--name custom`` must reach ``run_sync_grants`` as ``connection_name``.

    Without this wiring, the DSN-derived default would silently produce
    ``<dsn_db>::schema.table`` patterns that never match the serve-time
    ``custom::…`` patterns emitted by ``pretensor index --name custom``.
    """
    captured: dict[str, Any] = {}

    def _fake_run_sync_grants(connector: Any, **kwargs: Any) -> int:
        _ = connector
        captured.update(kwargs)
        return 1

    connector_cm = MagicMock()
    connector_cm.__enter__.return_value = MagicMock()
    connector_cm.__exit__.return_value = None

    monkeypatch.setattr(
        "pretensor.cli.commands.sync_grants.run_sync_grants", _fake_run_sync_grants
    )
    monkeypatch.setattr(
        "pretensor.cli.commands.sync_grants.get_connector",
        lambda cfg: connector_cm,
    )

    out_path = tmp_path / "visibility.yml"
    result = CliRunner().invoke(
        app,
        [
            "sync-grants",
            "--dsn",
            "postgresql://u:p@localhost/shopdb",
            "--name",
            "warehouse",
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, _normalize(result.stdout)
    assert captured["connection_name"] == "warehouse"


def test_sync_grants_empty_grants_emits_warning(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """F4: an empty-grants run must print a yellow warning the user can act on."""
    def _fake_run_sync_grants(connector: Any, **_kwargs: Any) -> int:
        _ = connector
        return 0

    connector_cm = MagicMock()
    connector_cm.__enter__.return_value = MagicMock()
    connector_cm.__exit__.return_value = None

    monkeypatch.setattr(
        "pretensor.cli.commands.sync_grants.run_sync_grants", _fake_run_sync_grants
    )
    monkeypatch.setattr(
        "pretensor.cli.commands.sync_grants.get_connector",
        lambda cfg: connector_cm,
    )

    result = CliRunner().invoke(
        app,
        [
            "sync-grants",
            "--dsn",
            "postgresql://u:p@localhost/shopdb",
            "--output",
            str(tmp_path / "visibility.yml"),
        ],
    )
    assert result.exit_code == 0, _normalize(result.stdout)
    plain = _normalize(result.stdout)
    assert "No SELECT grants returned" in plain
    # The success line still appears and includes the grantee count.
    assert "0 grantee profile(s)" in plain


def test_sync_grants_name_flag_appears_in_help() -> None:
    """F1: ``--name`` / ``-n`` must be documented in ``--help`` so users find it."""
    result = CliRunner().invoke(app, ["sync-grants", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "--name" in plain
    assert "-n" in plain


def test_sync_grants_bigquery_dsn_exits_1(tmp_path: Path) -> None:
    """BigQuery DSNs are unsupported and must exit 1 with an informative message."""
    result = CliRunner().invoke(
        app,
        [
            "sync-grants",
            "--dsn",
            "bigquery://project/dataset",
            "--output",
            str(tmp_path / "vis.yml"),
        ],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "BigQuery" in plain or "bigquery" in plain.lower()


def test_sync_grants_roles_option_forwarded(tmp_path: Path, monkeypatch: Any) -> None:
    """``--roles`` is parsed and forwarded to ``run_sync_grants`` as a frozenset."""
    captured: dict[str, Any] = {}

    def _fake_run_sync_grants(connector: Any, **kwargs: Any) -> int:
        _ = connector
        captured.update(kwargs)
        return 2

    connector_cm = MagicMock()
    connector_cm.__enter__.return_value = MagicMock()
    connector_cm.__exit__.return_value = None

    monkeypatch.setattr(
        "pretensor.cli.commands.sync_grants.run_sync_grants", _fake_run_sync_grants
    )
    monkeypatch.setattr(
        "pretensor.cli.commands.sync_grants.get_connector",
        lambda cfg: connector_cm,
    )

    result = CliRunner().invoke(
        app,
        [
            "sync-grants",
            "--dsn",
            "postgresql://u:p@localhost/shopdb",
            "--roles",
            "analyst,engineer",
            "--output",
            str(tmp_path / "vis.yml"),
        ],
    )
    assert result.exit_code == 0, _normalize(result.stdout)
    assert captured["roles_filter"] == frozenset({"analyst", "engineer"})


def test_sync_grants_value_error_exits_1(tmp_path: Path, monkeypatch: Any) -> None:
    """A ``ValueError`` from ``get_connector`` exits 1 with the error message."""
    monkeypatch.setattr(
        "pretensor.cli.commands.sync_grants.get_connector",
        lambda cfg: (_ for _ in ()).throw(ValueError("connector not supported")),
    )

    result = CliRunner().invoke(
        app,
        [
            "sync-grants",
            "--dsn",
            "postgresql://u:p@localhost/shopdb",
            "--output",
            str(tmp_path / "vis.yml"),
        ],
    )
    assert result.exit_code == 1
    assert "connector not supported" in _normalize(result.stdout)


def test_sync_grants_import_error_exits_1(tmp_path: Path, monkeypatch: Any) -> None:
    """An ``ImportError`` (missing optional driver) exits 1 with the error message."""
    monkeypatch.setattr(
        "pretensor.cli.commands.sync_grants.get_connector",
        lambda cfg: (_ for _ in ()).throw(ImportError("psycopg2 not installed")),
    )

    result = CliRunner().invoke(
        app,
        [
            "sync-grants",
            "--dsn",
            "postgresql://u:p@localhost/shopdb",
            "--output",
            str(tmp_path / "vis.yml"),
        ],
    )
    assert result.exit_code == 1
    assert "psycopg2 not installed" in _normalize(result.stdout)
