"""Registry mapping connection types to connector implementations."""

from __future__ import annotations

from importlib import import_module

from pretensor.connectors.base import BaseConnector
from pretensor.introspection.models.config import ConnectionConfig, DatabaseType

_POSTGRES_EXTRA_HINT = (
    "Install the Postgres extra: pip install 'pretensor[postgres]' "
    "(or pip install psycopg2-binary)."
)

_SNOWFLAKE_EXTRA_HINT = (
    "Install the Snowflake extra: pip install 'pretensor[snowflake]' "
    "(or pip install snowflake-sqlalchemy)."
)

_MYSQL_EXTRA_HINT = (
    "Install the MySQL extra: pip install 'pretensor[mysql]' (or pip install PyMySQL)."
)


def _postgres_connector_class() -> type[BaseConnector]:
    try:
        # psycopg2 is an optional dependency under the ``postgres`` extra. Import
        # it lazily here (mirroring the other connectors) so a bare install can
        # import pretensor and only fails — with a clear hint — when a Postgres
        # connection is actually attempted.
        import_module("psycopg2")
    except ImportError as exc:
        msg = f"Postgres connector requires psycopg2. {_POSTGRES_EXTRA_HINT}"
        raise ImportError(msg) from exc
    from pretensor.connectors.postgres import PostgresConnector

    return PostgresConnector


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

        msg = (
            f"BigQuery connector requires google-cloud-bigquery. {_BIGQUERY_EXTRA_HINT}"
        )
        raise ImportError(msg) from exc
    from pretensor.connectors.bigquery import BigQueryConnector

    return BigQueryConnector


def _mysql_connector_class() -> type[BaseConnector]:
    try:
        import_module("pymysql")
    except ImportError as exc:
        msg = f"MySQL connector requires PyMySQL. {_MYSQL_EXTRA_HINT}"
        raise ImportError(msg) from exc
    from pretensor.connectors.mysql import MySQLConnector

    return MySQLConnector


def get_connector(config: ConnectionConfig) -> BaseConnector:
    """Return a connector instance for the given configuration."""
    if config.type == DatabaseType.POSTGRES:
        cls = _postgres_connector_class()
        return cls(config)
    if config.type == DatabaseType.SNOWFLAKE:
        cls = _snowflake_connector_class()
        return cls(config)
    if config.type == DatabaseType.BIGQUERY:
        cls = _bigquery_connector_class()
        return cls(config)
    if config.type == DatabaseType.MYSQL:
        cls = _mysql_connector_class()
        return cls(config)
    msg = f"No connector registered for database type: {config.type}"
    raise ValueError(msg)
