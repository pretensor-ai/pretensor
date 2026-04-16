"""Parse database connection strings into :class:`ConnectionConfig`.

Supports PostgreSQL (``postgres://``, ``postgresql://``, ``postgresql+driver://``),
Snowflake (``snowflake://``), and BigQuery (``bigquery://project/dataset``).
Optional ``--dialect`` overrides scheme detection.

Also provides :func:`connection_config_from_source` and :func:`dsn_from_source`
for building configs from declarative ``sources:`` YAML blocks.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote, unquote, urlparse

from .config import ENV_VAR_PATTERN, ConnectionConfig, DatabaseType, SchemaFilter

if TYPE_CHECKING:
    from pretensor.cli.config_file import SourceConfig

RegistryDialect = Literal["postgres", "mysql", "snowflake", "bigquery"]


def registry_dialect_for(db_type: DatabaseType) -> RegistryDialect:
    """Map :class:`DatabaseType` to the value stored in ``RegistryEntry.dialect``."""
    if db_type == DatabaseType.SNOWFLAKE:
        return "snowflake"
    if db_type == DatabaseType.BIGQUERY:
        return "bigquery"
    return "postgres"


__all__ = [
    "RegistryDialect",
    "connection_config_from_postgres_dsn",
    "connection_config_from_registry_dsn",
    "connection_config_from_source",
    "connection_config_from_url",
    "dsn_from_source",
    "infer_database_type_from_dsn",
    "registry_dialect_for",
    "validate_source_env_vars",
]

_DIALECT_ALIASES: dict[str, DatabaseType] = {
    "postgres": DatabaseType.POSTGRES,
    "postgresql": DatabaseType.POSTGRES,
    "snowflake": DatabaseType.SNOWFLAKE,
    "bigquery": DatabaseType.BIGQUERY,
}


def infer_database_type_from_dsn(dsn: str) -> DatabaseType:
    """Infer :class:`DatabaseType` from the URL scheme (before ``+driver``)."""
    raw = dsn.strip()
    if "://" not in raw:
        msg = "DSN must be a URL with a scheme, e.g. postgresql://user@host/db"
        raise ValueError(msg)
    scheme_part, _, _ = raw.partition("://")
    base_scheme = scheme_part.lower().split("+", 1)[0]
    if base_scheme in ("postgres", "postgresql"):
        return DatabaseType.POSTGRES
    if base_scheme == "snowflake":
        return DatabaseType.SNOWFLAKE
    if base_scheme == "bigquery":
        return DatabaseType.BIGQUERY
    msg = (
        f"Unsupported DSN scheme {scheme_part!r}; use postgresql://, snowflake://, "
        "or bigquery://"
    )
    raise ValueError(msg)


def connection_config_from_url(
    dsn: str,
    connection_name: str,
    *,
    dialect_override: str | None = None,
) -> ConnectionConfig:
    """Build a :class:`ConnectionConfig` from a database URL.

    Args:
        dsn: Connection URL for PostgreSQL or Snowflake.
        connection_name: Logical name for snapshots and the graph registry.
        dialect_override: When set, force connector type (``postgres``,
            ``postgresql``, ``snowflake``, or ``bigquery``) regardless of URL scheme.

    Returns:
        A mutable :class:`ConnectionConfig` for connector dispatch.

    Raises:
        ValueError: If the URL or dialect is not supported.
    """
    raw = dsn.strip()
    if "://" not in raw:
        msg = "DSN must be a URL with a scheme, e.g. postgresql://user@host/db"
        raise ValueError(msg)

    conn_type: DatabaseType
    if dialect_override is not None:
        key = dialect_override.strip().lower()
        try:
            conn_type = _DIALECT_ALIASES[key]
        except KeyError as exc:
            allowed = ", ".join(sorted(_DIALECT_ALIASES))
            msg = f"Unknown dialect {dialect_override!r}; expected one of: {allowed}"
            raise ValueError(msg) from exc
    else:
        conn_type = infer_database_type_from_dsn(raw)

    if conn_type == DatabaseType.POSTGRES:
        return _config_from_postgres_url(raw, connection_name)
    if conn_type == DatabaseType.SNOWFLAKE:
        return _config_from_snowflake_url(raw, connection_name)
    if conn_type == DatabaseType.BIGQUERY:
        scheme_part, _, remainder = raw.partition("://")
        base_scheme = scheme_part.lower().split("+", 1)[0]
        parse_url = (
            raw if base_scheme == "bigquery" else f"bigquery://{remainder}"
        )
        return _config_from_bigquery_url(parse_url, connection_name)
    msg = f"No URL parser for database type: {conn_type}"
    raise ValueError(msg)


def connection_config_from_postgres_dsn(
    dsn: str, connection_name: str
) -> ConnectionConfig:
    """Build a :class:`ConnectionConfig` from a PostgreSQL DSN URL.

    Deprecated path name; prefer :func:`connection_config_from_url`.
    """
    return connection_config_from_url(dsn, connection_name)


def connection_config_from_registry_dsn(
    dsn: str,
    connection_name: str,
    dialect: RegistryDialect,
) -> ConnectionConfig:
    """Parse a registry-stored DSN using the recorded connector dialect."""
    if dialect == "snowflake":
        return connection_config_from_url(
            dsn, connection_name, dialect_override="snowflake"
        )
    if dialect == "bigquery":
        return connection_config_from_url(
            dsn, connection_name, dialect_override="bigquery"
        )
    return connection_config_from_url(dsn, connection_name)


def _config_from_postgres_url(raw: str, connection_name: str) -> ConnectionConfig:
    scheme_part, _, remainder = raw.partition("://")
    base_scheme = scheme_part.lower().split("+", 1)[0]
    if base_scheme not in ("postgres", "postgresql"):
        msg = f"Expected a postgres or postgresql DSN, got scheme {scheme_part!r}"
        raise ValueError(msg)

    parsed = urlparse(f"postgres://{remainder}")
    if parsed.hostname is None or parsed.hostname == "":
        msg = "PostgreSQL DSN must include a host"
        raise ValueError(msg)

    database = parsed.path.lstrip("/") or None
    user = unquote(parsed.username) if parsed.username else ""
    password = unquote(parsed.password) if parsed.password else None

    return ConnectionConfig(
        name=connection_name,
        type=DatabaseType.POSTGRES,
        host=parsed.hostname,
        port=parsed.port,
        database=database,
        user=user or None,
        password=password,
    )


def _config_from_snowflake_url(raw: str, connection_name: str) -> ConnectionConfig:
    """Parse ``snowflake://user:pass@account/db/schema?warehouse=...`` into config."""
    scheme_part, _, remainder = raw.partition("://")
    base_scheme = scheme_part.lower().split("+", 1)[0]
    if base_scheme != "snowflake":
        msg = f"Expected a snowflake DSN, got scheme {scheme_part!r}"
        raise ValueError(msg)

    parsed = urlparse(f"snowflake://{remainder}")
    account = parsed.hostname
    if account is None or account == "":
        msg = "Snowflake DSN must include account host (e.g. xy12345.us-east-1.aws)"
        raise ValueError(msg)

    path = parsed.path.strip("/")
    path_parts = [p for p in path.split("/") if p]
    database = path_parts[0] if len(path_parts) > 0 else None
    snowflake_schema = path_parts[1] if len(path_parts) > 1 else None

    user = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None

    warehouse: str | None = None
    role: str | None = None
    if parsed.query:
        from urllib.parse import parse_qs

        qs = parse_qs(parsed.query, keep_blank_values=True)
        wh = qs.get("warehouse", [None])[0]
        if wh:
            warehouse = unquote(wh)
        rl = qs.get("role", [None])[0]
        if rl:
            role = unquote(rl)

    return ConnectionConfig(
        name=connection_name,
        type=DatabaseType.SNOWFLAKE,
        host=account,
        database=database,
        user=user,
        password=password,
        schema_filter=_snowflake_schema_filter(snowflake_schema),
        metadata_extra={
            "snowflake_schema": snowflake_schema,
            "warehouse": warehouse,
            "role": role,
        },
    )


def _snowflake_schema_filter(snowflake_schema: str | None) -> SchemaFilter:
    if snowflake_schema:
        return SchemaFilter(include=[snowflake_schema.upper()])
    return SchemaFilter()


def validate_source_env_vars(source: SourceConfig) -> list[str]:
    """Return names of unset environment variables referenced in *source* fields.

    Iterates all fields on the dataclass dynamically so new fields are
    automatically covered without maintaining a hardcoded list.
    Returns an empty list when every reference resolves.
    """
    import dataclasses

    missing: list[str] = []
    for f in dataclasses.fields(source):
        value = getattr(source, f.name, None)
        if isinstance(value, str):
            for match in ENV_VAR_PATTERN.finditer(value):
                var = match.group(1)
                if os.environ.get(var) is None and var not in missing:
                    missing.append(var)
    return missing


def connection_config_from_source(
    name: str,
    source: SourceConfig,
) -> ConnectionConfig:
    """Build a :class:`ConnectionConfig` from a declarative source block.

    Args:
        name: Logical connection name (the YAML key under ``sources:``).
        source: Parsed :class:`SourceConfig` from the config file.

    Returns:
        A :class:`ConnectionConfig` ready for connector dispatch.

    Raises:
        ValueError: If the dialect or required fields are invalid.
    """
    key = source.dialect.strip().lower()
    try:
        conn_type = _DIALECT_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DIALECT_ALIASES))
        msg = f"Unknown dialect {source.dialect!r} in source `{name}`; expected one of: {allowed}"
        raise ValueError(msg) from exc

    if conn_type == DatabaseType.POSTGRES:
        if not source.host:
            raise ValueError(f"Source `{name}` (postgres) requires `host`")
        return ConnectionConfig(
            name=name,
            type=DatabaseType.POSTGRES,
            host=source.host,
            port=source.port,
            database=source.database,
            user=source.user,
            password=source.password,
        )

    if conn_type == DatabaseType.SNOWFLAKE:
        if not source.account:
            raise ValueError(f"Source `{name}` (snowflake) requires `account`")
        sf_schema = source.schema
        return ConnectionConfig(
            name=name,
            type=DatabaseType.SNOWFLAKE,
            host=source.account,
            database=source.database,
            user=source.user,
            password=source.password,
            schema_filter=_snowflake_schema_filter(sf_schema),
            metadata_extra={
                "snowflake_schema": sf_schema,
                "warehouse": source.warehouse,
                "role": source.role,
            },
        )

    if conn_type == DatabaseType.BIGQUERY:
        project = source.project
        dataset = source.dataset
        if not project or not dataset:
            raise ValueError(
                f"Source `{name}` (bigquery) requires both `project` and `dataset`"
            )
        return ConnectionConfig(
            name=name,
            type=DatabaseType.BIGQUERY,
            host=project,
            database=f"{project}/{dataset}",
            metadata_extra={"bq_project": project, "bq_location": source.location},
            schema_filter=SchemaFilter(include=[dataset]),
        )

    msg = f"No source builder for dialect: {source.dialect}"
    raise ValueError(msg)


def dsn_from_source(name: str, source: SourceConfig) -> str:
    """Build a DSN URL string from a declarative source block.

    Used for registry storage so that ``reindex`` can reconstruct connections
    from stored DSNs without the config file present.
    """
    key = source.dialect.strip().lower()
    try:
        conn_type = _DIALECT_ALIASES[key]
    except KeyError as exc:
        msg = f"Unknown dialect {source.dialect!r} in source `{name}`"
        raise ValueError(msg) from exc

    def _enc(s: str | None) -> str:
        return quote(s, safe="") if s else ""

    if conn_type == DatabaseType.POSTGRES:
        user = _enc(source.user) or "postgres"
        pw = f":{_enc(source.password)}" if source.password else ""
        host = source.host or "localhost"
        port = f":{source.port}" if source.port else ""
        db = f"/{source.database}" if source.database else ""
        return f"postgresql://{user}{pw}@{host}{port}{db}"

    if conn_type == DatabaseType.SNOWFLAKE:
        user = _enc(source.user) or ""
        pw = f":{_enc(source.password)}" if source.password else ""
        account = source.account or ""
        db = f"/{source.database}" if source.database else ""
        schema = f"/{source.schema}" if source.schema else ""
        params: list[str] = []
        if source.warehouse:
            params.append(f"warehouse={_enc(source.warehouse)}")
        if source.role:
            params.append(f"role={_enc(source.role)}")
        qs = f"?{'&'.join(params)}" if params else ""
        cred = f"{user}{pw}@" if user else ""
        return f"snowflake://{cred}{account}{db}{schema}{qs}"

    if conn_type == DatabaseType.BIGQUERY:
        project = source.project or ""
        dataset = source.dataset or ""
        loc = f"?location={_enc(source.location)}" if source.location else ""
        return f"bigquery://{project}/{dataset}{loc}"

    msg = f"No DSN builder for dialect: {source.dialect}"
    raise ValueError(msg)


def _config_from_bigquery_url(raw: str, connection_name: str) -> ConnectionConfig:
    """Parse ``bigquery://project/dataset?location=...`` into config.

    The URL path is ``/dataset`` or ``/project/dataset``; when the host is
    non-empty it is treated as the GCP project id and the first path segment
    is the dataset. When the host is empty, the first path segment is project
    and the second is dataset.
    """
    scheme_part, _, remainder = raw.partition("://")
    base_scheme = scheme_part.lower().split("+", 1)[0]
    if base_scheme != "bigquery":
        msg = f"Expected a bigquery DSN, got scheme {scheme_part!r}"
        raise ValueError(msg)

    parsed = urlparse(f"bigquery://{remainder}")
    path = parsed.path.strip("/")
    path_parts = [p for p in path.split("/") if p]

    project: str | None = None
    dataset: str | None = None
    host = (parsed.hostname or "").strip()
    if host:
        project = host
        dataset = path_parts[0] if path_parts else None
    else:
        if len(path_parts) >= 2:
            project, dataset = path_parts[0], path_parts[1]
        elif len(path_parts) == 1:
            msg = (
                "BigQuery DSN must include project and dataset "
                "(e.g. bigquery://my-project/my-dataset)"
            )
            raise ValueError(msg)

    if not project or not dataset:
        msg = (
            "BigQuery DSN must include project and dataset "
            "(e.g. bigquery://my-project/my-dataset)"
        )
        raise ValueError(msg)

    location: str | None = None
    if parsed.query:
        from urllib.parse import parse_qs

        qs = parse_qs(parsed.query, keep_blank_values=True)
        loc = qs.get("location", [None])[0]
        if loc:
            location = unquote(loc)

    extra: dict[str, Any] = {"bq_project": project, "bq_location": location}

    return ConnectionConfig(
        name=connection_name,
        type=DatabaseType.BIGQUERY,
        host=project,
        database=f"{project}/{dataset}",
        metadata_extra=extra,
        schema_filter=SchemaFilter(include=[dataset]),
    )
