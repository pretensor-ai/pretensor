"""Tests for individual wizard steps."""

from __future__ import annotations

from pathlib import Path

import pytest

from pretensor.cli.init.inference import InferredDsn, InferredRepo
from pretensor.cli.init.mcp_clients import McpClient
from pretensor.cli.init.steps import (
    choose_clients,
    collect_dsn,
    collect_repository,
    repo_skip_message,
)
from pretensor.cli.prompts import ScriptedPrompter
from pretensor.introspection.models.config import DatabaseType
from pretensor.introspection.models.dsn import connection_config_from_url


def test_collect_dsn_accepts_inferred_value() -> None:
    inferred = InferredDsn(dsn="postgresql://u:p@h:5432/db", env_var="DATABASE_URL")
    result = collect_dsn(ScriptedPrompter([True]), inferred)
    assert result == inferred


def test_collect_dsn_falls_back_to_paste_when_declined() -> None:
    inferred = InferredDsn(dsn="postgresql://u:p@h:5432/db", env_var="DATABASE_URL")
    prompter = ScriptedPrompter([False, 0, "postgresql://x:y@z:5432/other"])
    result = collect_dsn(prompter, inferred)
    assert result.dsn == "postgresql://x:y@z:5432/other"
    assert result.env_var is None


def test_collect_dsn_builds_postgres_from_parts() -> None:
    # 1 = "Enter the details one by one"; 0 = postgres in the dialect question.
    prompter = ScriptedPrompter([1, 0, "h", "5432", "u", "db", "p"])
    result = collect_dsn(prompter, None)
    assert result.dsn == "postgresql://u:p@h:5432/db"


def test_collect_dsn_percent_encodes_password_with_special_chars() -> None:
    prompter = ScriptedPrompter([1, 0, "h", "5432", "u", "db", "p@ss:w/rd"])
    result = collect_dsn(prompter, None)
    assert result.dsn == "postgresql://u:p%40ss%3Aw%2Frd@h:5432/db"


def test_collect_dsn_builds_mysql_from_parts() -> None:
    # 1 = mysql in the dialect question.
    prompter = ScriptedPrompter([1, 1, "h", "3306", "u", "db", "p"])
    result = collect_dsn(prompter, None)
    assert result.dsn == "mysql://u:p@h:3306/db"


def test_collect_dsn_builds_snowflake_from_parts_with_special_char_password() -> None:
    # 2 = snowflake. Answers, in order: account, user, password, database,
    # warehouse, role. Password contains reserved URL characters to verify
    # percent-encoding round-trips for this dialect too.
    prompter = ScriptedPrompter([1, 2, "acct", "u", "p@ss:w/rd", "db", "wh", "role"])
    result = collect_dsn(prompter, None)
    assert result.dsn == "snowflake://u:p%40ss%3Aw%2Frd@acct/db?warehouse=wh&role=role"


def test_collect_dsn_builds_snowflake_omits_empty_warehouse_and_role() -> None:
    prompter = ScriptedPrompter([1, 2, "acct", "u", "p", "db", "", ""])
    result = collect_dsn(prompter, None)
    assert result.dsn == "snowflake://u:p@acct/db"


def test_collect_dsn_builds_bigquery_from_parts() -> None:
    # 3 = bigquery. Answers, in order: project, dataset, location.
    prompter = ScriptedPrompter([1, 3, "proj", "dataset", "loc"])
    result = collect_dsn(prompter, None)
    assert result.dsn == "bigquery://proj/dataset?location=loc"


def test_collect_dsn_bigquery_reasks_once_on_blank_dataset() -> None:
    """A blank dataset answer is re-asked once, not accepted as omitted."""
    prompter = ScriptedPrompter([1, 3, "proj", "", "ds", ""])
    result = collect_dsn(prompter, None)
    assert result.dsn == "bigquery://proj/ds"


def test_collect_dsn_bigquery_two_blank_datasets_fails_downstream() -> None:
    """A second blank answer is not re-asked again.

    It falls through to the existing "must include project and dataset"
    failure at connection time, exactly as an unset dataset always has.
    """
    prompter = ScriptedPrompter([1, 3, "proj", "", "", ""])
    result = collect_dsn(prompter, None)
    assert result.dsn == "bigquery://proj/"
    with pytest.raises(ValueError, match="must include project and dataset"):
        connection_config_from_url(result.dsn, "bq")


# ── Round-trip: assembled DSNs parse back into the expected config ─────────


def test_collect_dsn_mysql_round_trips_through_parser() -> None:
    prompter = ScriptedPrompter([1, 1, "h", "3306", "u", "db", "p@ss:w/rd"])
    result = collect_dsn(prompter, None)
    cfg = connection_config_from_url(result.dsn, "m1")
    assert cfg.type == DatabaseType.MYSQL
    assert cfg.host == "h"
    assert cfg.port == 3306
    assert cfg.user == "u"
    assert cfg.database == "db"
    # The special-char password must decode back to the raw value, not stay
    # percent-encoded.
    assert cfg.password == "p@ss:w/rd"


def test_collect_dsn_snowflake_round_trips_through_parser() -> None:
    prompter = ScriptedPrompter([1, 2, "acct", "u", "p@ss:w/rd", "db", "wh", "role"])
    result = collect_dsn(prompter, None)
    cfg = connection_config_from_url(result.dsn, "sf1")
    assert cfg.type == DatabaseType.SNOWFLAKE
    assert cfg.host == "acct"
    assert cfg.user == "u"
    assert cfg.password == "p@ss:w/rd"
    assert cfg.database == "db"
    assert cfg.metadata_extra["warehouse"] == "wh"
    assert cfg.metadata_extra["role"] == "role"


def test_collect_dsn_bigquery_round_trips_through_parser() -> None:
    prompter = ScriptedPrompter([1, 3, "proj", "dataset", "loc"])
    result = collect_dsn(prompter, None)
    cfg = connection_config_from_url(result.dsn, "bq1")
    assert cfg.type == DatabaseType.BIGQUERY
    assert cfg.database == "proj/dataset"
    assert cfg.metadata_extra["bq_project"] == "proj"
    assert cfg.metadata_extra["bq_location"] == "loc"


def test_collect_dsn_dialect_flag_skips_dialect_question() -> None:
    """A pre-selected dialect (from --dialect) must not prompt for one."""
    prompter = ScriptedPrompter([1, "h", "3306", "u", "db", "p"])
    result = collect_dsn(prompter, None, dialect="mysql")
    assert result.dsn == "mysql://u:p@h:3306/db"


def test_collect_repository_accepts_inferred(tmp_path: Path) -> None:
    inferred = InferredRepo(path=tmp_path, file_count=12)
    assert collect_repository(ScriptedPrompter([True]), inferred, tmp_path) == tmp_path


def test_collect_repository_returns_none_when_declined(tmp_path: Path) -> None:
    inferred = InferredRepo(path=tmp_path, file_count=12)
    assert (
        collect_repository(ScriptedPrompter([False, False]), inferred, tmp_path) is None
    )


def test_repo_skip_message_names_supported_languages(tmp_path: Path) -> None:
    message = repo_skip_message(tmp_path)
    assert ".py" in message
    assert "—" not in message


def test_choose_clients_selects_confirmed_only() -> None:
    clients = (
        McpClient(key="cursor", label="Cursor", config_path=Path("/tmp/a.json")),
        McpClient(key="claude-code", label="Claude Code", config_path=None),
    )
    chosen = choose_clients(ScriptedPrompter([True, False]), clients)
    assert [c.key for c in chosen] == ["cursor"]


def test_collect_repository_custom_path_declines_inferred(tmp_path: Path) -> None:
    """User can decline inferred repo and enter a custom path via prompt.

    When inferred is not None and user declines it, collect_repository asks
    "Link a different code repository?" and then "Repository path". The
    returned path must be expanded and resolved.
    """
    inferred = InferredRepo(path=tmp_path / "inferred", file_count=12)
    custom_path = tmp_path / "custom" / "repo"

    # Answers in order: decline inferred (False), accept "Link different?" (True),
    # enter custom path string.
    prompter = ScriptedPrompter([False, True, str(custom_path)])
    result = collect_repository(prompter, inferred, tmp_path)

    assert result == custom_path.expanduser().resolve()


def test_collect_repository_custom_path_with_no_inferred(tmp_path: Path) -> None:
    """When no inferred repo, user can enter a custom path directly.

    With inferred=None, collect_repository skips the first confirm and asks
    "Link a different code repository?" immediately. User enters a custom path.
    The returned path must be expanded and resolved.
    """
    custom_path = tmp_path / "myrepo"

    # Answers in order: accept "Link a different code repository?" (True),
    # enter custom path string.
    prompter = ScriptedPrompter([True, str(custom_path)])
    result = collect_repository(prompter, None, tmp_path)

    assert result == custom_path.expanduser().resolve()
