"""Tests for the pretensor.errors exception hierarchy."""

from __future__ import annotations

from pretensor.errors import ConnectorError, PretensorError


def test_pretensor_error_subclasses_exception() -> None:
    assert issubclass(PretensorError, Exception)


def test_connector_error_subclasses_pretensor_error() -> None:
    assert issubclass(ConnectorError, PretensorError)


def test_postgres_connector_error_is_shared_class() -> None:
    from pretensor.connectors.postgres import ConnectorError as pg_ce

    assert pg_ce is ConnectorError


def test_mysql_connector_error_is_shared_class() -> None:
    from pretensor.connectors.mysql import ConnectorError as my_ce

    assert my_ce is ConnectorError


def test_postgres_and_mysql_connector_error_are_same_object() -> None:
    from pretensor.connectors.mysql import ConnectorError as my_ce
    from pretensor.connectors.postgres import ConnectorError as pg_ce

    assert pg_ce is my_ce


def test_database_not_found_error_is_pretensor_error_and_key_error() -> None:
    from pretensor.core.registry import DatabaseNotFoundError

    exc = DatabaseNotFoundError("missing_db")
    assert isinstance(exc, PretensorError)
    assert isinstance(exc, KeyError)


def test_metric_compile_error_is_pretensor_error_and_value_error() -> None:
    from pretensor.semantic.compiler import MetricCompileError

    exc = MetricCompileError("bad metric")
    assert isinstance(exc, PretensorError)
    assert isinstance(exc, ValueError)


def test_dsn_decrypt_error_is_pretensor_error_and_runtime_error() -> None:
    from pretensor.core.dsn_crypto import DSNDecryptError

    exc = DSNDecryptError("bad key")
    assert isinstance(exc, PretensorError)
    assert isinstance(exc, RuntimeError)


def test_cli_config_error_is_pretensor_error_and_value_error() -> None:
    from pretensor.cli.config_file import CliConfigError

    exc = CliConfigError("bad config")
    assert isinstance(exc, PretensorError)
    assert isinstance(exc, ValueError)


def test_dbt_manifest_error_is_pretensor_error() -> None:
    from pretensor.enrichment.dbt.manifest import DbtManifestError

    exc = DbtManifestError("no manifest")
    assert isinstance(exc, PretensorError)


def test_llm_budget_exceeded_error_is_pretensor_error() -> None:
    from pretensor.intelligence.llm_runtime import LlmBudgetExceededError

    exc = LlmBudgetExceededError("over budget")
    assert isinstance(exc, PretensorError)


def test_cyclic_dependency_error_is_pretensor_error_and_value_error() -> None:
    from pretensor.intelligence.steps import CyclicDependencyError

    exc = CyclicDependencyError("cycle detected")
    assert isinstance(exc, PretensorError)
    assert isinstance(exc, ValueError)


def test_bigquery_connector_error_is_connector_error() -> None:
    from pretensor.connectors.bigquery import BigQueryConnectorError

    assert issubclass(BigQueryConnectorError, ConnectorError)
    assert issubclass(BigQueryConnectorError, PretensorError)


def test_snowflake_connector_error_is_connector_error() -> None:
    from pretensor.connectors.snowflake import SnowflakeConnectorError

    assert issubclass(SnowflakeConnectorError, ConnectorError)
    assert issubclass(SnowflakeConnectorError, PretensorError)


def test_connector_error_catchable_via_pretensor_error() -> None:
    try:
        raise ConnectorError("db down")
    except PretensorError:
        pass
    else:
        raise AssertionError("ConnectorError not caught as PretensorError")


def test_database_not_found_error_catchable_as_key_error() -> None:
    from pretensor.core.registry import DatabaseNotFoundError

    try:
        raise DatabaseNotFoundError("x")
    except KeyError:
        pass
    else:
        raise AssertionError("DatabaseNotFoundError not caught as KeyError")


def test_metric_compile_error_catchable_as_value_error() -> None:
    from pretensor.semantic.compiler import MetricCompileError

    try:
        raise MetricCompileError("bad")
    except ValueError:
        pass
    else:
        raise AssertionError("MetricCompileError not caught as ValueError")
