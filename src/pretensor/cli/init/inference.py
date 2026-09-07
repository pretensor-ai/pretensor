"""Pure detection helpers for the setup wizard.

Nothing here prompts or writes. The wizard's express path is only as good as
this module, so it is kept free of I/O beyond filesystem reads and is tested
without a terminal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from pretensor.enrichment.analyze.walker import walk_repo

__all__ = [
    "DIALECTS",
    "InferredDsn",
    "InferredRepo",
    "assemble_bigquery_dsn",
    "assemble_mysql_dsn",
    "assemble_postgres_dsn",
    "assemble_snowflake_dsn",
    "count_supported_files",
    "infer_dsn_from_env",
    "infer_repo",
]

DIALECTS = ("postgres", "mysql", "snowflake", "bigquery")

_DSN_ENV_VARS = ("DATABASE_URL", "POSTGRES_URL")
_PG_REQUIRED = ("PGHOST", "PGUSER", "PGDATABASE")


@dataclass(frozen=True, slots=True)
class InferredDsn:
    """A DSN discovered in the environment.

    ``env_var`` is the variable it came from, or None when the DSN was
    assembled from several variables and so has no single reference to store.
    """

    dsn: str
    env_var: str | None


@dataclass(frozen=True, slots=True)
class InferredRepo:
    """A code repository worth offering to analyze."""

    path: Path
    file_count: int


def assemble_postgres_dsn(
    *, user: str, password: str, host: str, port: str, database: str
) -> str:
    """Build a ``postgresql://`` DSN from individually collected parts.

    ``user`` and ``password`` are percent-encoded so credentials containing
    reserved URL characters (``@``, ``:``, ``/``, etc.) round-trip correctly.
    """
    encoded_user = quote(user, safe="")
    encoded_password = quote(password, safe="") if password else ""
    credentials = f"{encoded_user}:{encoded_password}" if password else encoded_user
    return f"postgresql://{credentials}@{host}:{port}/{database}"


def assemble_mysql_dsn(
    *, user: str, password: str, host: str, port: str, database: str
) -> str:
    """Build a ``mysql://`` DSN from individually collected parts.

    Mirrors :func:`assemble_postgres_dsn`; the shape is what
    ``_config_from_mysql_url`` in ``introspection.models.dsn`` parses.
    """
    encoded_user = quote(user, safe="")
    encoded_password = quote(password, safe="") if password else ""
    credentials = f"{encoded_user}:{encoded_password}" if password else encoded_user
    return f"mysql://{credentials}@{host}:{port}/{database}"


def assemble_snowflake_dsn(
    *,
    account: str,
    user: str,
    password: str,
    database: str,
    warehouse: str,
    role: str,
) -> str:
    """Build a ``snowflake://`` DSN from individually collected parts.

    ``warehouse`` and ``role`` are optional; an empty string omits the query
    parameter entirely. The shape mirrors what ``_config_from_snowflake_url``
    parses: ``snowflake://user:pass@account/database?warehouse=...&role=...``
    (the same shape ``dsn_from_source`` builds for a declarative source).
    """
    encoded_user = quote(user, safe="") if user else ""
    encoded_password = quote(password, safe="") if password else ""
    credentials = f"{encoded_user}:{encoded_password}@" if encoded_user else ""
    db_part = f"/{database}" if database else ""
    params: list[str] = []
    if warehouse:
        params.append(f"warehouse={quote(warehouse, safe='')}")
    if role:
        params.append(f"role={quote(role, safe='')}")
    query = f"?{'&'.join(params)}" if params else ""
    return f"snowflake://{credentials}{account}{db_part}{query}"


def assemble_bigquery_dsn(*, project: str, dataset: str, location: str) -> str:
    """Build a ``bigquery://`` DSN from individually collected parts.

    ``dataset`` and ``location`` are optional; the shape mirrors what
    ``_config_from_bigquery_url`` parses:
    ``bigquery://project/dataset?location=...``. A DSN built without a
    dataset fails ``connection_config_from_url`` with a clear "must include
    project and dataset" error rather than silently being accepted.
    """
    encoded_location = quote(location, safe="") if location else ""
    loc_part = f"?location={encoded_location}" if location else ""
    return f"bigquery://{project}/{dataset}{loc_part}"


def infer_dsn_from_env(env: Mapping[str, str]) -> InferredDsn | None:
    """Return a DSN discovered in *env*, or None when nothing usable is set."""
    for name in _DSN_ENV_VARS:
        value = env.get(name, "").strip()
        if value:
            return InferredDsn(dsn=value, env_var=name)
    if all(env.get(key, "").strip() for key in _PG_REQUIRED):
        host = env["PGHOST"].strip()
        user = env["PGUSER"].strip()
        database = env["PGDATABASE"].strip()
        port = env.get("PGPORT", "").strip() or "5432"
        password = env.get("PGPASSWORD", "").strip()
        return InferredDsn(
            dsn=assemble_postgres_dsn(
                user=user, password=password, host=host, port=port, database=database
            ),
            env_var=None,
        )
    return None


def count_supported_files(repo_path: Path) -> int:
    """Count files under *repo_path* that the analyze walker would scan."""
    return sum(1 for _ in walk_repo(repo_path))


def infer_repo(cwd: Path) -> InferredRepo | None:
    """Return *cwd* as a candidate repository when it is worth analyzing."""
    if not (cwd / ".git").exists():
        return None
    count = count_supported_files(cwd)
    if count == 0:
        return None
    return InferredRepo(path=cwd, file_count=count)
