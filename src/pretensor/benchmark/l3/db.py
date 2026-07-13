"""Read-only SQL execution helpers for the L3 benchmark runners.

The runners need to execute both the gold SQL and the agent-emitted SQL
against a real PostgreSQL instance, then compare the resulting rows. This
module is the only place that opens DB connections in the L3 layer, so all
the safety guardrails live here:

* **Database URL is resolved from a per-dataset env var.** No CLI flag —
  ``--dataset`` already pins which DB to talk to, and growing the CLI
  surface fragments the contract.
* **Every query runs in a transaction that is rolled back at the end.**
  Defense-in-depth against a misbehaving agent that emits ``DROP TABLE``;
  the read-only intent of the run is preserved even if the syntactic
  guard below misses a corner case.
* **Statement-level timeout** (30s by default) prevents pathological
  agent SQL (cross joins, accidental cartesian products) from hanging
  the runner.
* **Non-SELECT guard** rejects any agent SQL whose first non-comment,
  non-whitespace token is not ``SELECT``, ``WITH``, or ``(``. Together
  with the rolled-back transaction this is two independent layers of
  protection — a slip in either is contained by the other.
"""

from __future__ import annotations

import os
import re
from typing import Any

import sqlalchemy
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from pretensor.benchmark.runner import Dataset
from pretensor.errors import PretensorError

__all__ = [
    "DEFAULT_STATEMENT_TIMEOUT_MS",
    "QueryExecutionError",
    "RefusedNonSelectError",
    "execute_query",
    "is_select_only",
    "resolve_database_url",
]


DEFAULT_STATEMENT_TIMEOUT_MS = 30_000


_DATASET_TO_ENV_VAR: dict[Dataset, str] = {
    Dataset.PAGILA: "PAGILA_DATABASE_URL",
    Dataset.TPCH: "TPCH_DATABASE_URL",
    Dataset.ADVENTUREWORKS: "ADVENTUREWORKS_DATABASE_URL",
}
"""L3 baseline supports the three OSS-bundled datasets that ship with DDL.

The synthetic fixtures (``analytics_dwh``, ``saas_multitenant``,
``adversarial``) have no DDL dump and no live DB they correspond to,
so they are deliberately absent here — calling ``resolve_database_url``
on them raises a clear error.
"""


class QueryExecutionError(PretensorError, RuntimeError):
    """Raised when a SQL statement fails to execute (parse error, missing
    object, runtime fault, statement timeout, etc.)."""


class RefusedNonSelectError(QueryExecutionError):
    """Raised when an agent SQL statement is rejected by the SELECT-only guard.

    A subclass of ``QueryExecutionError`` so callers that want to record
    "could not execute" can use a single ``except`` and still inspect the
    type to distinguish "refused before execution" from "DB error".
    """


def resolve_database_url(dataset: Dataset, env: dict[str, str] | None = None) -> str:
    """Read the per-dataset DSN env var; raise if unset or dataset unsupported.

    ``env`` is for tests (pass an explicit dict to bypass ``os.environ``);
    production callers leave it ``None`` and the function reads the
    process environment.
    """
    if env is None:
        env = dict(os.environ)
    try:
        var_name = _DATASET_TO_ENV_VAR[dataset]
    except KeyError as exc:
        raise LookupError(
            f"L3 baseline runner does not support dataset {dataset.value!r}: "
            f"only {[d.value for d in _DATASET_TO_ENV_VAR]} have DDL bundles."
        ) from exc
    value = env.get(var_name)
    if not value:
        raise LookupError(
            f"Set {var_name} to the PostgreSQL DSN of a {dataset.value} "
            f"database (e.g. {var_name}=postgresql://user@host:5432/{dataset.value})."
        )
    return value


# Strip line comments (``--`` to end of line) and block comments (``/* ... */``);
# whitespace; then peek at the leading token. CTEs (``WITH ...``) and
# parenthesised SELECTs (``(SELECT ...) UNION ...``) are allowed alongside
# bare ``SELECT``.
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_ALLOWED_LEAD_TOKENS = ("SELECT", "WITH", "TABLE", "VALUES", "(")


def is_select_only(sql: str) -> bool:
    """Return ``True`` iff ``sql`` begins with a read-only top-level keyword.

    ``TABLE`` and ``VALUES`` are PostgreSQL read forms; ``WITH`` covers
    CTEs whose final statement is a ``SELECT`` (we don't deep-parse the
    CTE — combined with the rolled-back transaction in
    :func:`execute_query`, that's enough). A bare ``(`` allows
    ``(SELECT ...) UNION ...`` style queries.

    A statement that *contains* DDL/DML keywords later (in a string
    literal, in a column comment, etc.) is not blocked here — that's a
    job for the database's own parser. The only goal of this guard is
    "first thing the LLM emits must be a read".
    """
    stripped = _BLOCK_COMMENT_RE.sub(" ", sql)
    stripped = _LINE_COMMENT_RE.sub(" ", stripped).strip()
    if not stripped:
        return False
    upper = stripped.upper()
    for tok in _ALLOWED_LEAD_TOKENS:
        if tok == "(" and upper.startswith("("):
            return True
        if upper.startswith(tok) and (
            len(upper) == len(tok) or not upper[len(tok)].isalnum()
        ):
            return True
    return False


def execute_query(
    dsn: str,
    sql: str,
    *,
    enforce_select_only: bool = False,
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
) -> tuple[list[tuple[Any, ...]], list[str]]:
    """Run ``sql`` against ``dsn`` inside a rolled-back transaction.

    Returns ``(rows, column_names)``. ``enforce_select_only=True`` runs
    the syntactic guard first and raises :class:`RefusedNonSelectError`
    when it fails — used for agent-emitted SQL. Gold SQL skips the guard
    (it's authored by us and may use any read form, but we still wrap
    in a rolled-back transaction).

    Any DB error (parse failure, missing object, statement timeout, etc.)
    is wrapped in :class:`QueryExecutionError` so callers can record it
    without having to know the SQLAlchemy / driver-specific exception
    hierarchy.
    """
    if enforce_select_only and not is_select_only(sql):
        raise RefusedNonSelectError(
            "agent SQL refused: first non-comment token is not SELECT/WITH/(/VALUES."
        )
    # A new Engine per call is intentional: L3 only runs ~10–25 questions
    # per dataset (so 2N ≤ ~50 engines), each query owns its own pool /
    # transaction lifecycle, and the dispose() in the finally block keeps
    # connection state from leaking across questions if anything goes wrong.
    engine: Engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                conn.execute(
                    sqlalchemy.text(
                        f"SET LOCAL statement_timeout = {int(statement_timeout_ms)}"
                    )
                )
                cursor = conn.execute(sqlalchemy.text(sql))
                rows = [tuple(row) for row in cursor.fetchall()]
                column_names = list(cursor.keys())
                return rows, column_names
            except SQLAlchemyError as exc:
                raise QueryExecutionError(str(exc)) from exc
            finally:
                trans.rollback()
    finally:
        engine.dispose()
