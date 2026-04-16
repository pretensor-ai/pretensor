"""Registry mapping connection types to connector implementations."""

from __future__ import annotations

from importlib import import_module

from pretensor.connectors.base import BaseConnector
from pretensor.connectors.postgres import PostgresConnector
from pretensor.introspection.models.config import ConnectionConfig, DatabaseType

_SNOWFLAKE_EXTRA_HINT = (
    "Install the Snowflake extra: pip install 'pretensor[snowflake]' "
    "(or pip install snowflake-sqlalchemy)."
)

def _snowflake_connector_class() -> type[BaseConnector]:
    try:
        # Optional dep: string path keeps pyright happy when the extra is not installed.
        import_module("snowflake.sqlalchemy")
    except ImportError as exc:
        msg = f"Snowflake connector requires snowflake-sqlalchemy. {_SNOWFLAKE_EXTRA_HINT}"
        raise ImportError(msg) from exc
    from pretensor.connectors.snowflake import SnowflakeConnector

    return SnowflakeConnector


def _bigquery_connector_class() -> type[BaseConnector]:
    try:
        import_module("google.cloud.bigquery")
    except ImportError as exc:
        from pretensor.connectors.bigquery import _BIGQUERY_EXTRA_HINT
        msg = f"BigQuery connector requires google-cloud-bigquery. {_BIGQUERY_EXTRA_HINT}"
        raise ImportError(msg) from exc
    from pretensor.connectors.bigquery import BigQueryConnector

    return BigQueryConnector


def get_connector(config: ConnectionConfig) -> BaseConnector:
    """Return a connector instance for the given configuration."""
    if config.type == DatabaseType.POSTGRES:
        return PostgresConnector(config)
    if config.type == DatabaseType.SNOWFLAKE:
        cls = _snowflake_connector_class()
        return cls(config)
    if config.type == DatabaseType.BIGQUERY:
        cls = _bigquery_connector_class()
        return cls(config)
    msg = f"No connector registered for database type: {config.type}"
    raise ValueError(msg)
