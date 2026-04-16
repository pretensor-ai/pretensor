"""Tests for ``pretensor`` CLI config file support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pretensor.cli.config_file import CliConfigError, load_cli_config
from pretensor.cli.main import app


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
    assert "Unknown `graph` config key" in result.stdout


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

    def _fake_cfg_from_url(dsn: str, name: str, *, dialect_override: str | None = None) -> Any:
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

    def _fake_cfg_from_url(dsn: str, name: str, *, dialect_override: str | None = None) -> Any:
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
        "pretensor.cli.commands.index.inspect",
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

