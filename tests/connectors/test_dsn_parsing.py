"""Tests for multi-dialect DSN parsing."""

from __future__ import annotations

import pytest

from pretensor.cli.config_file import SourceConfig
from pretensor.cli.paths import default_connection_name
from pretensor.introspection.models.config import DatabaseType
from pretensor.introspection.models.dsn import (
    connection_config_from_source,
    connection_config_from_url,
    dsn_from_source,
    infer_database_type_from_dsn,
    registry_dialect_for,
    validate_source_env_vars,
)


def test_infer_postgres() -> None:
    assert infer_database_type_from_dsn("postgresql://u@h/db") == DatabaseType.POSTGRES
    assert infer_database_type_from_dsn("postgres://u@h/db") == DatabaseType.POSTGRES


def test_infer_snowflake() -> None:
    assert (
        infer_database_type_from_dsn("snowflake://u:p@acct/db/schema")
        == DatabaseType.SNOWFLAKE
    )


def test_infer_bigquery() -> None:
    assert (
        infer_database_type_from_dsn("bigquery://my-project/my_dataset")
        == DatabaseType.BIGQUERY
    )


def test_postgres_config_from_url() -> None:
    cfg = connection_config_from_url(
        "postgresql://alice:sec@db.example:5432/mydb", "c1"
    )
    assert cfg.type == DatabaseType.POSTGRES
    assert cfg.host == "db.example"
    assert cfg.port == 5432
    assert cfg.database == "mydb"
    assert cfg.user == "alice"
    assert cfg.password == "sec"


def test_snowflake_config_from_url() -> None:
    dsn = (
        "snowflake://bob:pw@xy12345.us-east-1.aws/MYDB/PUBLIC?warehouse=WH&role=ANALYST"
    )
    cfg = connection_config_from_url(dsn, "sf1")
    assert cfg.type == DatabaseType.SNOWFLAKE
    assert cfg.host == "xy12345.us-east-1.aws"
    assert cfg.database == "MYDB"
    assert cfg.user == "bob"
    assert cfg.password == "pw"
    assert cfg.metadata_extra.get("snowflake_schema") == "PUBLIC"
    assert cfg.metadata_extra.get("warehouse") == "WH"
    assert cfg.metadata_extra.get("role") == "ANALYST"
    assert "PUBLIC" in cfg.schema_filter.include


def test_unknown_dialect_override() -> None:
    with pytest.raises(ValueError, match="Unknown dialect"):
        connection_config_from_url(
            "postgresql://u@h/db", "x", dialect_override="not-a-dialect"
        )


def test_dialect_override_postgresql_alias() -> None:
    cfg = connection_config_from_url(
        "postgresql://u@h/db", "x", dialect_override="postgres"
    )
    assert cfg.type == DatabaseType.POSTGRES


def test_registry_dialect_mapping() -> None:
    assert registry_dialect_for(DatabaseType.POSTGRES) == "postgres"
    assert registry_dialect_for(DatabaseType.SNOWFLAKE) == "snowflake"
    assert registry_dialect_for(DatabaseType.BIGQUERY) == "bigquery"


def test_bigquery_config_from_url() -> None:
    cfg = connection_config_from_url(
        "bigquery://my-project/analytics?location=EU", "bq1"
    )
    assert cfg.type == DatabaseType.BIGQUERY
    assert cfg.host == "my-project"
    assert cfg.database == "my-project/analytics"
    assert cfg.metadata_extra.get("bq_project") == "my-project"
    assert cfg.metadata_extra.get("bq_location") == "EU"
    assert cfg.schema_filter.include == ["analytics"]


def test_bigquery_dialect_override() -> None:
    cfg = connection_config_from_url(
        "mysql://my-project/analytics", "x", dialect_override="bigquery"
    )
    assert cfg.type == DatabaseType.BIGQUERY
    assert cfg.database == "my-project/analytics"


# ---------------------------------------------------------------------------
# validate_source_env_vars
# ---------------------------------------------------------------------------


def test_validate_source_env_vars_no_refs() -> None:
    """Plain strings without ${…} return an empty list."""
    src = SourceConfig(dialect="postgres", host="localhost", user="alice", password="s3cret")
    assert validate_source_env_vars(src) == []


def test_validate_source_env_vars_all_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PG_USER", "alice")
    monkeypatch.setenv("PG_PASS", "s3cret")
    src = SourceConfig(dialect="postgres", host="localhost", user="${PG_USER}", password="${PG_PASS}")
    assert validate_source_env_vars(src) == []


def test_validate_source_env_vars_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SF_PASSWORD", raising=False)
    src = SourceConfig(dialect="snowflake", account="xy123", password="${SF_PASSWORD}")
    assert validate_source_env_vars(src) == ["SF_PASSWORD"]


def test_validate_source_env_vars_mixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only missing vars are returned; set ones are excluded."""
    monkeypatch.setenv("PG_USER", "alice")
    monkeypatch.delenv("PG_PASS", raising=False)
    monkeypatch.delenv("PG_HOST", raising=False)
    src = SourceConfig(
        dialect="postgres", host="${PG_HOST}", user="${PG_USER}", password="${PG_PASS}",
    )
    result = validate_source_env_vars(src)
    assert "PG_HOST" in result
    assert "PG_PASS" in result
    assert "PG_USER" not in result


def test_validate_source_env_vars_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same var referenced in multiple fields appears only once."""
    monkeypatch.delenv("SHARED_SECRET", raising=False)
    src = SourceConfig(
        dialect="postgres", host="localhost",
        user="${SHARED_SECRET}", password="${SHARED_SECRET}",
    )
    assert validate_source_env_vars(src) == ["SHARED_SECRET"]

def test_default_connection_name_uses_dataset_for_bigquery() -> None:
    assert default_connection_name("bigquery://my-project/analytics") == "analytics"


# ── connection_config_from_source tests ─────────────────────────────────


def test_source_postgres_config() -> None:
    src = SourceConfig(
        dialect="postgres", host="db.example", port=5432, user="alice",
        password="sec", database="mydb",
    )
    cfg = connection_config_from_source("pg1", src)
    assert cfg.type == DatabaseType.POSTGRES
    assert cfg.name == "pg1"
    assert cfg.host == "db.example"
    assert cfg.port == 5432
    assert cfg.database == "mydb"
    assert cfg.user == "alice"
    assert cfg.password == "sec"


def test_source_snowflake_config() -> None:
    src = SourceConfig(
        dialect="snowflake", account="xy12345.us-east-1.aws",
        user="bob", password="pw", database="MYDB",
        schema="PUBLIC", warehouse="WH", role="ANALYST",
    )
    cfg = connection_config_from_source("sf1", src)
    assert cfg.type == DatabaseType.SNOWFLAKE
    assert cfg.name == "sf1"
    assert cfg.host == "xy12345.us-east-1.aws"
    assert cfg.database == "MYDB"
    assert cfg.user == "bob"
    assert cfg.metadata_extra.get("warehouse") == "WH"
    assert cfg.metadata_extra.get("role") == "ANALYST"
    assert cfg.metadata_extra.get("snowflake_schema") == "PUBLIC"
    assert "PUBLIC" in cfg.schema_filter.include


def test_source_bigquery_config() -> None:
    src = SourceConfig(
        dialect="bigquery", project="my-project", dataset="analytics",
        location="EU",
    )
    cfg = connection_config_from_source("bq1", src)
    assert cfg.type == DatabaseType.BIGQUERY
    assert cfg.host == "my-project"
    assert cfg.database == "my-project/analytics"
    assert cfg.metadata_extra.get("bq_project") == "my-project"
    assert cfg.metadata_extra.get("bq_location") == "EU"
    assert cfg.schema_filter.include == ["analytics"]


def test_source_postgres_missing_host() -> None:
    src = SourceConfig(dialect="postgres", database="mydb")
    with pytest.raises(ValueError, match="requires `host`"):
        connection_config_from_source("bad", src)


def test_source_snowflake_missing_account() -> None:
    src = SourceConfig(dialect="snowflake", database="MYDB")
    with pytest.raises(ValueError, match="requires `account`"):
        connection_config_from_source("bad", src)


def test_source_bigquery_missing_project() -> None:
    src = SourceConfig(dialect="bigquery", dataset="ds")
    with pytest.raises(ValueError, match="requires both"):
        connection_config_from_source("bad", src)


def test_source_unknown_dialect() -> None:
    src = SourceConfig(dialect="mysql")
    with pytest.raises(ValueError, match="Unknown dialect"):
        connection_config_from_source("bad", src)


# ── dsn_from_source tests ──────────────────────────────────────────────


def test_dsn_from_source_postgres() -> None:
    src = SourceConfig(
        dialect="postgres", host="db.example", port=5432,
        user="alice", password="s3c!", database="mydb",
    )
    dsn = dsn_from_source("pg1", src)
    assert dsn.startswith("postgresql://")
    assert "db.example" in dsn
    assert "5432" in dsn
    assert "mydb" in dsn
    # Round-trip: parse the DSN and verify
    cfg = connection_config_from_url(dsn, "pg1")
    assert cfg.host == "db.example"
    assert cfg.port == 5432
    assert cfg.database == "mydb"
    assert cfg.user == "alice"


def test_dsn_from_source_snowflake() -> None:
    src = SourceConfig(
        dialect="snowflake", account="xy12345.us-east-1.aws",
        user="bob", password="pw", database="MYDB",
        schema="PUBLIC", warehouse="WH", role="ANALYST",
    )
    dsn = dsn_from_source("sf1", src)
    assert dsn.startswith("snowflake://")
    assert "xy12345.us-east-1.aws" in dsn
    assert "warehouse=WH" in dsn
    assert "role=ANALYST" in dsn
    # Round-trip
    cfg = connection_config_from_url(dsn, "sf1")
    assert cfg.type == DatabaseType.SNOWFLAKE
    assert cfg.host == "xy12345.us-east-1.aws"
    assert cfg.database == "MYDB"
    assert cfg.metadata_extra.get("warehouse") == "WH"


def test_dsn_from_source_bigquery() -> None:
    src = SourceConfig(
        dialect="bigquery", project="my-project",
        dataset="analytics", location="EU",
    )
    dsn = dsn_from_source("bq1", src)
    assert dsn.startswith("bigquery://")
    assert "my-project" in dsn
    assert "analytics" in dsn
    assert "location=EU" in dsn
    # Round-trip
    cfg = connection_config_from_url(dsn, "bq1")
    assert cfg.type == DatabaseType.BIGQUERY
    assert cfg.database == "my-project/analytics"


def test_port_coercion_after_env_var_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``port: ${PORT}`` resolves to string; model_validator must coerce to int."""
    from pretensor.introspection.models.config import ConnectionConfig, DatabaseType

    monkeypatch.setenv("PORT", "5432")
    cfg = ConnectionConfig(
        name="test",
        type=DatabaseType.POSTGRES,
        host="localhost",
        port="${PORT}",  # type: ignore[arg-type]
        database="mydb",
    )
    assert cfg.port == 5432
    assert isinstance(cfg.port, int)
