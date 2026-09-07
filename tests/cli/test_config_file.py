"""Tests for ``pretensor`` CLI config file support."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import typer
from ruamel.yaml import YAMLError
from typer.testing import CliRunner

from pretensor.cli.config_file import (
    CliConfigError,
    load_cli_config,
    record_repository,
    record_source,
    resolve_aliased_path_option,
)
from pretensor.cli.main import app

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")


def _write_minimal_registry(state_dir: Path) -> None:
    reg_path = state_dir / "registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(
        json.dumps({"version": 1, "entries": {}}, indent=2),
        encoding="utf-8",
    )


def test_list_uses_default_config_yaml_state_dir(tmp_path: Path) -> None:
    state_dir = tmp_path / "state-from-config"
    _write_minimal_registry(state_dir)
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(
        f"state_dir: {state_dir}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["--config", str(cfg_dir / "config.yaml"), "list"],
    )

    assert result.exit_code == 0
    assert "Registry is empty" in result.stdout


def test_list_cli_flag_overrides_config_state_dir(tmp_path: Path) -> None:
    cfg_state_dir = tmp_path / "cfg-state"
    cfg_state_dir.mkdir(parents=True, exist_ok=True)
    explicit_state_dir = tmp_path / "explicit-state"
    _write_minimal_registry(explicit_state_dir)
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        f"state_dir: {cfg_state_dir}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "list",
            "--state-dir",
            str(explicit_state_dir),
        ],
    )
    assert result.exit_code == 0
    assert "Registry is empty" in result.stdout


def test_main_errors_for_missing_explicit_config_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    result = CliRunner().invoke(app, ["--config", str(missing), "list"])
    assert result.exit_code == 1
    assert "Config file not found" in result.stdout


def test_main_errors_for_invalid_graph_key_in_config(tmp_path: Path) -> None:
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        "graph:\n  unknown_key: 1\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["--config", str(config_path), "list"])
    assert result.exit_code == 1
    assert "Unknown `graph` config key" in _ANSI_ESCAPE_RE.sub("", result.stdout)


def test_index_uses_connection_defaults_from_config(tmp_path: Path) -> None:
    cfg_state_dir = tmp_path / "cfg-state"
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"state_dir: {cfg_state_dir}",
                "connection_defaults:",
                "  name: config-conn",
                "  dialect: postgres",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _fake_cfg_from_url(
        dsn: str, name: str, *, dialect_override: str | None = None
    ) -> Any:
        captured["name"] = name
        captured["dialect"] = dialect_override
        raise ValueError("stop after asserting defaults")

    with patch(
        "pretensor.cli.commands.index.connection_config_from_url",
        side_effect=_fake_cfg_from_url,
    ):
        result = CliRunner().invoke(
            app,
            ["--config", str(config_path), "index", "postgresql://u:p@localhost/db1"],
        )
    assert result.exit_code == 1
    assert captured["name"] == "config-conn"
    assert captured["dialect"] == "postgres"


def test_index_cli_name_overrides_connection_default(tmp_path: Path) -> None:
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        "connection_defaults:\n  name: config-conn\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _fake_cfg_from_url(
        dsn: str, name: str, *, dialect_override: str | None = None
    ) -> Any:
        captured["name"] = name
        raise ValueError("stop after asserting override")

    with patch(
        "pretensor.cli.commands.index.connection_config_from_url",
        side_effect=_fake_cfg_from_url,
    ):
        result = CliRunner().invoke(
            app,
            [
                "--config",
                str(config_path),
                "index",
                "postgresql://u:p@localhost/db1",
                "--name",
                "cli-conn",
            ],
        )
    assert result.exit_code == 1
    assert captured["name"] == "cli-conn"


def test_reindex_uses_state_dir_from_config(tmp_path: Path) -> None:
    cfg_state_dir = tmp_path / "cfg-state"
    cfg_state_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        f"state_dir: {cfg_state_dir}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["--config", str(config_path), "reindex", "postgresql://u:p@localhost/db1"],
    )
    assert result.exit_code == 1
    assert "No registry found" in result.stdout


def test_sync_grants_uses_config_defaults_for_output_and_name(tmp_path: Path) -> None:
    state_dir = tmp_path / "from-config"
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"state_dir: {state_dir}",
                "connection_defaults:",
                "  name: cfg-conn",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    class _ConnectorCtx:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
            return None

    def _fake_run_sync_grants(connector: Any, **kwargs: Any) -> int:
        _ = connector
        captured.update(kwargs)
        return 0

    with (
        patch(
            "pretensor.cli.commands.sync_grants.get_connector",
            return_value=_ConnectorCtx(),
        ),
        patch(
            "pretensor.cli.commands.sync_grants.run_sync_grants",
            side_effect=_fake_run_sync_grants,
        ),
    ):
        result = CliRunner().invoke(
            app,
            [
                "--config",
                str(config_path),
                "sync-grants",
                "--dsn",
                "postgresql://u:p@localhost/db1",
            ],
        )
    assert result.exit_code == 0
    assert captured["connection_name"] == "cfg-conn"
    assert captured["output_path"] == (state_dir / "visibility.yml").resolve()


def test_add_uses_config_defaults(tmp_path: Path) -> None:
    cfg_state_dir = tmp_path / "cfg-state"
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"state_dir: {cfg_state_dir}",
                "connection_defaults:",
                "  name: cfg-add-name",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class _FakeDSNEncryptor:
        def __init__(self, path: Path) -> None:
            captured["keystore_path"] = path

        def encrypt(self, value: str) -> str:
            return f"enc:{value}"

        def decrypt(self, value: str) -> str:
            return value.removeprefix("enc:")

    def _fake_connection_config_from_url(dsn: str, name: str) -> Any:
        captured["connection_name"] = name

        class _Cfg:
            database = "cfg-db"
            type = "postgres"

        return _Cfg()

    with (
        patch(
            "pretensor.cli.commands.connections.add_remove.DSNEncryptor",
            _FakeDSNEncryptor,
        ),
        patch(
            "pretensor.cli.commands.connections.add_remove.connection_config_from_url",
            side_effect=_fake_connection_config_from_url,
        ),
        patch(
            "pretensor.cli.commands.connections.add_remove.registry_dialect_for",
            return_value="postgres",
        ),
    ):
        result = CliRunner().invoke(
            app,
            ["--config", str(config_path), "add", "postgresql://u:p@localhost/db1"],
        )
    assert result.exit_code == 0
    assert captured["connection_name"] == "cfg-add-name"
    assert captured["keystore_path"] == cfg_state_dir / "keystore"


# ── Source config parsing tests ──────────────────────────────────────────


def test_load_sources_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "sources:",
                "  my_pg:",
                "    dialect: postgres",
                "    host: localhost",
                "    port: 5432",
                "    user: app",
                "    password: secret",
                "    database: mydb",
                "  my_sf:",
                "    dialect: snowflake",
                "    account: xy12345.us-east-1.aws",
                "    user: bob",
                "    password: pw",
                "    database: ANALYTICS",
                "    schema: PUBLIC",
                "    warehouse: COMPUTE_WH",
                "    role: ANALYST",
                "  my_bq:",
                "    dialect: bigquery",
                "    project: my-gcp-project",
                "    dataset: reporting",
                "    location: US",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_cli_config(config_path)
    assert len(cfg.sources) == 3

    pg = cfg.sources["my_pg"]
    assert pg.dialect == "postgres"
    assert pg.host == "localhost"
    assert pg.port == 5432
    assert pg.user == "app"
    assert pg.password == "secret"
    assert pg.database == "mydb"

    sf = cfg.sources["my_sf"]
    assert sf.dialect == "snowflake"
    assert sf.account == "xy12345.us-east-1.aws"
    assert sf.warehouse == "COMPUTE_WH"
    assert sf.role == "ANALYST"
    assert sf.schema == "PUBLIC"

    bq = cfg.sources["my_bq"]
    assert bq.dialect == "bigquery"
    assert bq.project == "my-gcp-project"
    assert bq.dataset == "reporting"
    assert bq.location == "US"


def test_sources_secrets_merge(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "\n".join(
            [
                "sources:",
                "  my_pg:",
                "    dialect: postgres",
                "    host: localhost",
                "    user: app",
                "    database: mydb",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cfg_dir / "sources.secrets.yaml").write_text(
        "\n".join(
            [
                "my_pg:",
                "  password: from-secrets-file",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_cli_config(cfg_dir / "config.yaml")
    assert cfg.sources["my_pg"].password == "from-secrets-file"


def test_sources_secrets_override(tmp_path: Path) -> None:
    """Secrets file overrides values from main config."""
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "\n".join(
            [
                "sources:",
                "  db:",
                "    dialect: postgres",
                "    host: localhost",
                "    password: original",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cfg_dir / "sources.secrets.yaml").write_text(
        "db:\n  password: overridden\n",
        encoding="utf-8",
    )
    cfg = load_cli_config(cfg_dir / "config.yaml")
    assert cfg.sources["db"].password == "overridden"


def test_source_missing_dialect_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sources:\n  bad:\n    host: localhost\n",
        encoding="utf-8",
    )
    with pytest.raises(CliConfigError, match="requires a `dialect`"):
        load_cli_config(config_path)


def test_source_unknown_key_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sources:\n  bad:\n    dialect: postgres\n    bogus_key: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(CliConfigError, match="Unknown key"):
        load_cli_config(config_path)


def test_source_invalid_port_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sources:\n  bad:\n    dialect: postgres\n    host: x\n    port: not_a_number\n",
        encoding="utf-8",
    )
    with pytest.raises(CliConfigError, match="port.*must be an integer"):
        load_cli_config(config_path)


def test_source_url_reference_parsed(tmp_path: Path) -> None:
    """A ``url:`` source with an env-var reference parses without a `dialect`."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sources:\n  mydb:\n    url: ${DATABASE_URL}\n",
        encoding="utf-8",
    )
    cfg = load_cli_config(config_path)
    src = cfg.sources["mydb"]
    assert src.url == "${DATABASE_URL}"
    assert src.dialect is None
    assert src.host is None


def test_source_url_with_dialect_override_parsed(tmp_path: Path) -> None:
    """``dialect:`` is a valid override alongside ``url:``."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sources:\n  mydb:\n    url: ${DATABASE_URL}\n    dialect: postgres\n",
        encoding="utf-8",
    )
    cfg = load_cli_config(config_path)
    src = cfg.sources["mydb"]
    assert src.url == "${DATABASE_URL}"
    assert src.dialect == "postgres"


def test_source_url_and_host_conflict_raises(tmp_path: Path) -> None:
    """``url:`` is mutually exclusive with the field-based form."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sources:\n  mydb:\n    url: ${DATABASE_URL}\n    host: localhost\n",
        encoding="utf-8",
    )
    with pytest.raises(CliConfigError, match="mydb.*url.*cannot be combined"):
        load_cli_config(config_path)


def test_source_url_and_password_conflict_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sources:\n  mydb:\n    url: ${DATABASE_URL}\n    password: secret\n",
        encoding="utf-8",
    )
    with pytest.raises(CliConfigError, match="password"):
        load_cli_config(config_path)


def test_source_empty_url_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "sources:\n  mydb:\n    url: ''\n",
        encoding="utf-8",
    )
    with pytest.raises(CliConfigError, match="non-empty string"):
        load_cli_config(config_path)


def test_source_url_and_secrets_overlay_conflict_raises_with_hint(
    tmp_path: Path,
) -> None:
    """A url source whose secrets overlay adds a field key must raise, and the
    message must point at both files so the user can self-serve the fix."""
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "sources:\n  app:\n    url: ${DATABASE_URL}\n",
        encoding="utf-8",
    )
    (cfg_dir / "sources.secrets.yaml").write_text(
        "app:\n  password: x\n",
        encoding="utf-8",
    )
    with pytest.raises(CliConfigError, match="sources.secrets.yaml") as excinfo:
        load_cli_config(cfg_dir / "config.yaml")
    assert "password" in str(excinfo.value)
    assert "—" not in str(excinfo.value)


def test_empty_sources_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("sources:\n", encoding="utf-8")
    cfg = load_cli_config(config_path)
    assert cfg.sources == {}


def test_no_sources_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("state_dir: .\n", encoding="utf-8")
    cfg = load_cli_config(config_path)
    assert cfg.sources == {}


def test_index_source_flag_resolves_source_config(tmp_path: Path) -> None:
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        "\n".join(
            [
                "sources:",
                "  my_pg:",
                "    dialect: postgres",
                "    host: localhost",
                "    port: 5432",
                "    user: app",
                "    password: secret",
                "    database: testdb",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _fake_inspect(config: Any) -> Any:
        captured["host"] = config.host
        captured["name"] = config.name
        captured["type"] = config.type
        raise ValueError("stop after capturing")

    with patch(
        "pretensor.cli.commands._command_runners.inspect",
        side_effect=_fake_inspect,
    ):
        result = CliRunner().invoke(
            app,
            ["--config", str(config_path), "index", "--source", "my_pg"],
        )
    assert result.exit_code == 1
    assert captured["host"] == "localhost"
    assert captured["name"] == "my_pg"


def test_index_source_unknown_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        "sources:\n  known:\n    dialect: postgres\n    host: x\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["--config", str(config_path), "index", "--source", "nonexistent"],
    )
    assert result.exit_code == 1
    assert "Unknown source" in result.stdout


def test_index_dsn_and_source_mutually_exclusive(tmp_path: Path) -> None:
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        "sources:\n  s1:\n    dialect: postgres\n    host: x\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config_path),
            "index",
            "postgresql://u@h/db",
            "--source",
            "s1",
        ],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.stdout


def test_index_no_args_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text("sources: {}\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["--config", str(config_path), "index"],
    )
    assert result.exit_code == 1
    assert "Provide a DSN" in result.stdout


def test_index_all_no_sources_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text("sources: {}\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["--config", str(config_path), "index", "--all"],
    )
    assert result.exit_code == 1
    assert "No sources defined" in result.stdout


def test_index_dsn_still_works(tmp_path: Path) -> None:
    """Backwards compat: bare DSN positional argument still works."""
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def _fake_cfg_from_url(
        dsn: str, name: str, *, dialect_override: str | None = None
    ) -> Any:
        captured["dsn"] = dsn
        captured["name"] = name
        raise ValueError("stop after asserting")

    with patch(
        "pretensor.cli.commands.index.connection_config_from_url",
        side_effect=_fake_cfg_from_url,
    ):
        result = CliRunner().invoke(
            app,
            ["--config", str(config_path), "index", "postgresql://u:p@h/db"],
        )
    assert result.exit_code == 1
    assert captured["dsn"] == "postgresql://u:p@h/db"


def test_serve_uses_visibility_defaults_from_config(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    vis_path = tmp_path / "custom-visibility.yml"
    config_path = tmp_path / "pretensor.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"state_dir: {state_dir}",
                "visibility:",
                f"  path: {vis_path}",
                "  profile: analyst",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _fake_run_server(
        graph_dir: Path,
        *,
        visibility_path: Path | None,
        profile: str | None,
        config: Any | None = None,
    ) -> None:
        captured["graph_dir"] = graph_dir
        captured["visibility_path"] = visibility_path
        captured["profile"] = profile
        captured["config"] = config

    with (
        patch("pretensor.cli.commands.serve.run_server", side_effect=_fake_run_server),
        patch("pretensor.cli.commands.serve.print_mcp_config"),
    ):
        result = CliRunner().invoke(
            app,
            ["--config", str(config_path), "serve", "--no-print-config"],
        )
    assert result.exit_code == 0
    assert captured["graph_dir"] == state_dir.resolve()
    assert captured["visibility_path"] == vis_path
    assert captured["profile"] == "analyst"
    assert captured["config"] is not None


# ── private_key_path source config tests ─────────────────────────────────


def test_source_private_key_path_accepted(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "sources:",
                "  sf:",
                "    dialect: snowflake",
                "    account: xy12345.us-east-1.aws",
                "    user: bob",
                "    database: MYDB",
                "    private_key_path: /home/bob/.snowflake/rsa_key.p8",
                "    private_key_passphrase: hunter2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_cli_config(config_path)
    sf = cfg.sources["sf"]
    assert sf.private_key_path == "/home/bob/.snowflake/rsa_key.p8"
    assert sf.private_key_passphrase == "hunter2"
    assert sf.password is None


def test_source_private_key_path_via_secrets_overlay(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "\n".join(
            [
                "sources:",
                "  sf:",
                "    dialect: snowflake",
                "    account: xy12345.us-east-1.aws",
                "    user: bob",
                "    database: MYDB",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cfg_dir / "sources.secrets.yaml").write_text(
        "\n".join(
            [
                "sf:",
                "  private_key_path: /run/secrets/snowflake_key.p8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_cli_config(cfg_dir / "config.yaml")
    assert cfg.sources["sf"].private_key_path == "/run/secrets/snowflake_key.p8"


def test_source_private_key_path_wrong_name_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "sources:",
                "  sf:",
                "    dialect: snowflake",
                "    account: xy12345",
                "    private_key_file: /should/fail",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CliConfigError, match="Unknown key"):
        load_cli_config(config_path)


# ── resolve_aliased_path_option ──────────────────────────────────────────


_alias_app = typer.Typer()


@_alias_app.command("resolve")
def _resolve_alias_command(
    primary: Path = typer.Option(Path("/default"), "--primary"),
    alias: Path | None = typer.Option(None, "--alias"),
    config: Path | None = typer.Option(None, "--config-value"),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    resolved = resolve_aliased_path_option(
        ctx,
        primary_param="primary",
        primary_value=primary,
        alias_param="alias",
        alias_value=alias,
        config_value=config,
    )
    typer.echo(str(resolved))


def test_resolve_aliased_path_option_defaults_when_nothing_passed() -> None:
    result = CliRunner().invoke(_alias_app, [])
    assert result.exit_code == 0
    assert result.stdout.strip() == "/default"


def test_resolve_aliased_path_option_alias_wins_when_only_alias_passed() -> None:
    result = CliRunner().invoke(_alias_app, ["--alias", "/from-alias"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "/from-alias"


def test_resolve_aliased_path_option_primary_wins_when_only_primary_passed() -> None:
    result = CliRunner().invoke(_alias_app, ["--primary", "/from-primary"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "/from-primary"


def test_resolve_aliased_path_option_primary_wins_when_both_passed() -> None:
    """When both the canonical flag and its alias are passed, the canonical one wins."""
    result = CliRunner().invoke(
        _alias_app,
        ["--primary", "/from-primary", "--alias", "/from-alias"],
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "/from-primary"


def test_resolve_aliased_path_option_config_wins_over_default() -> None:
    """When neither flag is passed, the config value wins over the default."""
    result = CliRunner().invoke(_alias_app, ["--config-value", "/from-config"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "/from-config"


# ── Repository config parsing tests ─────────────────────────────────────────


def test_repositories_block_parsed(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "repositories:\n  - path: ./app\n    service: app\n    connection: main\n",
        encoding="utf-8",
    )
    cfg = load_cli_config(cfg_dir / "config.yaml")
    assert len(cfg.repositories) == 1
    repo = cfg.repositories[0]
    assert repo.path == (cfg_dir / "app").resolve()
    assert repo.service == "app"
    assert repo.connection == "main"


def test_repository_service_defaults_to_directory_name(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "repositories:\n  - path: ./billing\n    connection: main\n",
        encoding="utf-8",
    )
    cfg = load_cli_config(cfg_dir / "config.yaml")
    assert cfg.repositories[0].service == "billing"


def test_repository_unknown_key_rejected(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "repositories:\n  - path: ./app\n    connection: main\n    bogus: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(CliConfigError, match="bogus"):
        load_cli_config(cfg_dir / "config.yaml")


def test_repository_requires_connection(tmp_path: Path) -> None:
    cfg_dir = tmp_path / ".pretensor"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "repositories:\n  - path: ./app\n", encoding="utf-8"
    )
    with pytest.raises(CliConfigError, match="connection"):
        load_cli_config(cfg_dir / "config.yaml")


# ── record_repository tests ─────────────────────────────────────────────────


def test_record_repository_creates_fresh_file(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"
    repo_path = tmp_path / "app"
    repo_path.mkdir()

    record_repository(config_path, path=repo_path, service="app", connection="main")

    assert config_path.exists()
    cfg = load_cli_config(config_path)
    assert len(cfg.repositories) == 1
    repo = cfg.repositories[0]
    assert repo.path == repo_path.resolve()
    assert repo.service == "app"
    assert repo.connection == "main"


def test_record_repository_preserves_existing_comment(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    repo_path = tmp_path / "app"
    repo_path.mkdir()
    config_path.write_text(
        "# hand-written comment\nstate_dir: .\n",
        encoding="utf-8",
    )

    record_repository(config_path, path=repo_path, service="app", connection="main")

    text = config_path.read_text(encoding="utf-8")
    assert "# hand-written comment" in text
    cfg = load_cli_config(config_path)
    assert len(cfg.repositories) == 1
    assert cfg.repositories[0].connection == "main"


def test_record_repository_updates_instead_of_duplicating(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"
    repo_path = tmp_path / "app"
    repo_path.mkdir()

    record_repository(config_path, path=repo_path, service="app", connection="first")
    record_repository(config_path, path=repo_path, service="app2", connection="second")

    cfg = load_cli_config(config_path)
    assert len(cfg.repositories) == 1
    repo = cfg.repositories[0]
    assert repo.service == "app2"
    assert repo.connection == "second"


def test_record_repository_creates_backup_when_file_existed(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    repo_path = tmp_path / "app"
    repo_path.mkdir()
    config_path.write_text("state_dir: .\n", encoding="utf-8")

    record_repository(config_path, path=repo_path, service="app", connection="main")

    backups = list(config_path.parent.glob("config.yaml.bak.*"))
    assert len(backups) == 1


def test_record_repository_no_backup_when_file_is_new(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"
    repo_path = tmp_path / "app"
    repo_path.mkdir()

    record_repository(config_path, path=repo_path, service="app", connection="main")

    backups = list(config_path.parent.glob("config.yaml.bak.*"))
    assert backups == []


def test_record_repository_malformed_file_raises_without_touching_it(
    tmp_path: Path,
) -> None:
    """A malformed pre-existing file must raise and never be backed up or written.

    Parsing happens before the backup is taken (mirroring
    ``mcp_clients.register_client``), so a bad file is left completely
    untouched rather than stranding a useless `.bak` copy.
    """
    config_path = tmp_path / ".pretensor" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    repo_path = tmp_path / "app"
    repo_path.mkdir()
    original = "repositories:\n  - path: [unclosed\n"
    config_path.write_text(original, encoding="utf-8")

    with pytest.raises(YAMLError):
        record_repository(config_path, path=repo_path, service="app", connection="main")

    assert config_path.read_text(encoding="utf-8") == original


# ── record_source tests ─────────────────────────────────────────────────────


def test_record_source_creates_fresh_file(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"

    record_source(config_path, name="mydb", url_reference="${DATABASE_URL}")

    assert config_path.exists()
    cfg = load_cli_config(config_path)
    assert list(cfg.sources) == ["mydb"]
    src = cfg.sources["mydb"]
    assert src.url == "${DATABASE_URL}"
    assert src.dialect is None


def test_record_source_writes_literal_reference_not_resolved_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The literal ``${VAR}`` string is written, never a resolved DSN."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:secretpw@h:5432/db")
    config_path = tmp_path / ".pretensor" / "config.yaml"

    record_source(config_path, name="mydb", url_reference="${DATABASE_URL}")

    text = config_path.read_text(encoding="utf-8")
    assert "${DATABASE_URL}" in text
    assert "secretpw" not in text
    assert "postgresql://u:" not in text


def test_record_source_with_dialect(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"

    record_source(
        config_path, name="mydb", url_reference="${DATABASE_URL}", dialect="postgres"
    )

    cfg = load_cli_config(config_path)
    assert cfg.sources["mydb"].dialect == "postgres"


def test_record_source_updates_instead_of_duplicating(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"

    record_source(config_path, name="mydb", url_reference="${DATABASE_URL}")
    record_source(config_path, name="mydb", url_reference="${POSTGRES_URL}")

    cfg = load_cli_config(config_path)
    assert list(cfg.sources) == ["mydb"]
    assert cfg.sources["mydb"].url == "${POSTGRES_URL}"


def test_record_source_clears_field_keys_from_existing_entry(tmp_path: Path) -> None:
    """Rewriting a field-based entry as url-based must clear the old field keys.

    Otherwise the written entry has both forms and the next `load_cli_config`
    raises the mutual-exclusion `CliConfigError`, breaking every subsequent
    CLI command.
    """
    config_path = tmp_path / ".pretensor" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "sources:",
                "  app:",
                "    dialect: postgres",
                "    host: localhost",
                "    user: alice",
                "    password: secret",
                "    database: appdb",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record_source(config_path, name="app", url_reference="${DATABASE_URL}")

    # The rewritten config must still load cleanly.
    cfg = load_cli_config(config_path)
    src = cfg.sources["app"]
    assert src.url == "${DATABASE_URL}"
    assert src.host is None
    assert src.user is None
    assert src.password is None
    assert src.database is None

    # And none of the old field keys survive in the raw YAML.
    text = config_path.read_text(encoding="utf-8")
    assert "host:" not in text
    assert "password:" not in text
    assert "secret" not in text

    # The user's prior field-based config is recoverable from the backup.
    backups = list(config_path.parent.glob("config.yaml.bak.*"))
    assert len(backups) == 1
    backup_text = backups[0].read_text(encoding="utf-8")
    assert "host: localhost" in backup_text
    assert "password: secret" in backup_text


def test_record_source_clears_field_keys_but_keeps_existing_dialect(
    tmp_path: Path,
) -> None:
    """`dialect` is not a field key: it survives a field-to-url rewrite untouched
    when `record_source` is not given an explicit `dialect`."""
    config_path = tmp_path / ".pretensor" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "sources:\n  app:\n    dialect: postgres\n    host: localhost\n",
        encoding="utf-8",
    )

    record_source(config_path, name="app", url_reference="${DATABASE_URL}")

    cfg = load_cli_config(config_path)
    assert cfg.sources["app"].dialect == "postgres"
    assert cfg.sources["app"].host is None


def test_record_source_creates_backup_when_file_existed(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("state_dir: .\n", encoding="utf-8")

    record_source(config_path, name="mydb", url_reference="${DATABASE_URL}")

    backups = list(config_path.parent.glob("config.yaml.bak.*"))
    assert len(backups) == 1


def test_record_source_no_backup_when_file_is_new(tmp_path: Path) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"

    record_source(config_path, name="mydb", url_reference="${DATABASE_URL}")

    backups = list(config_path.parent.glob("config.yaml.bak.*"))
    assert backups == []


def test_record_source_malformed_file_raises_without_touching_it(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / ".pretensor" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    original = "sources:\n  bad: [unclosed\n"
    config_path.write_text(original, encoding="utf-8")

    with pytest.raises(YAMLError):
        record_source(config_path, name="mydb", url_reference="${DATABASE_URL}")

    assert config_path.read_text(encoding="utf-8") == original


def test_record_source_preserves_existing_repositories_block(tmp_path: Path) -> None:
    """Writing a source must not clobber an existing ``repositories:`` block."""
    config_path = tmp_path / ".pretensor" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "repositories:\n  - path: /app\n    service: app\n    connection: main\n",
        encoding="utf-8",
    )

    record_source(config_path, name="mydb", url_reference="${DATABASE_URL}")

    cfg = load_cli_config(config_path)
    assert len(cfg.repositories) == 1
    assert cfg.sources["mydb"].url == "${DATABASE_URL}"
