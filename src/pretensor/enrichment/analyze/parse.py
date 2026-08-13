"""Multi-dialect sqlglot retry wrapper for SQL table reference extraction."""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from pretensor.connectors.lineage_sqlglot import dml_write_targets, table_refs_from_sql

__all__ = [
    "ParsedSql",
    "normalize_placeholders",
    "normalize_sql",
    "parse_sql",
    "quiet_sqlglot",
]

logger = logging.getLogger(__name__)

_DIALECTS = ["", "postgres", "mysql", "bigquery", "snowflake", "trino", "spark"]

# DBAPI-style bind placeholders that break sqlglot's parser when left in place.
# Replaced with NULL before parsing so parameterized statements (the norm in
# scanned application code) keep their full structure — most importantly the
# INSERT/UPDATE node, which carries the write-target classification.
_PLACEHOLDER_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"%\([A-Za-z_][A-Za-z0-9_]*\)s"),  # psycopg named: %(name)s
    re.compile(r"%s"),  # psycopg positional: %s
    re.compile(r"(?<![:\w]):[A-Za-z_][A-Za-z0-9_]*"),  # named binds :name, not ::casts
    re.compile(r"(?<=[\s(=,<>])\?(?=[\s),;]|$)"),  # qmark style: ?
)


# MySQL/SQLite REPLACE INTO is a delete-then-insert write to the target table,
# but sqlglot cannot parse it in any dialect (it falls back to an opaque
# Command with no table nodes). Rewriting the head to INSERT INTO preserves
# the write-target classification; the fingerprint is computed from the
# original text so identity is unaffected.
_REPLACE_INTO_RE = re.compile(
    r"^(\s*)REPLACE\s+(?:LOW_PRIORITY\s+|DELAYED\s+)?(?:INTO\s+)?",
    re.IGNORECASE,
)


def normalize_placeholders(sql: str) -> str:
    """Replace DBAPI bind placeholders (``%s``, ``%(name)s``, ``:name``, ``?``)
    with ``NULL`` so sqlglot can parse parameterized statements. ``::`` casts and
    string contents that don't look like bind markers are left alone."""
    for pattern in _PLACEHOLDER_RES:
        sql = pattern.sub("NULL", sql)
    return sql


def normalize_sql(sql: str) -> str:
    """Full pre-parse normalization: placeholder substitution plus the
    ``REPLACE INTO`` → ``INSERT INTO`` rewrite."""
    return _REPLACE_INTO_RE.sub(r"\1INSERT INTO ", normalize_placeholders(sql))


# Characters that never appear in a correctly parsed identifier part. A dialect
# that "succeeds" by tokenizing another dialect's quoting into the name itself
# (e.g. the generic dialect turning `my-project.sales.orders` into a literal
# backtick) must not win the retry — its refs are garbage and the right
# dialect would never get tried.
_INVALID_IDENT_CHARS = ("`", '"', "'", ";", "\n")


def _refs_look_valid(refs: list[tuple[str, str]]) -> bool:
    for schema, table in refs:
        if not table:
            return False
        for part in (schema, table):
            if any(ch in part for ch in _INVALID_IDENT_CHARS):
                return False
        if "." in table:
            return False
    return True


@contextmanager
def quiet_sqlglot() -> Iterator[None]:
    """Silence sqlglot's own error logging during trial parses.

    The multi-dialect retry intentionally feeds sqlglot statements it may not
    parse; without this, every failed attempt prints an ERROR line to the
    user's console.
    """
    sqlglot_logger = logging.getLogger("sqlglot")
    previous = sqlglot_logger.level
    sqlglot_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        sqlglot_logger.setLevel(previous)


@dataclass
class ParsedSql:
    """Extracted table references and fingerprint from a SQL string."""

    table_refs: list[tuple[str, str]] = field(default_factory=list)
    write_targets: list[tuple[str, str]] = field(default_factory=list)
    dialect_used: str = ""
    fingerprint: str = ""


def parse_sql(sql: str, *, default_schema: str = "public") -> ParsedSql:
    """Extract table references via multi-dialect retry.

    Tries each dialect in ``_DIALECTS`` until one returns non-empty,
    plausible ``table_refs`` — refs whose names contain quoting characters or
    dots are treated as a failed parse so the retry can reach the dialect that
    understands the statement's quoting (BigQuery/MySQL backticks).
    Always returns a ``ParsedSql`` (``table_refs`` may be empty if all dialects fail).
    The ``fingerprint`` is ``sha256(stripped+lowercased sql)[:16]`` and is always
    computed from the original text, before normalization.
    """
    fingerprint = hashlib.sha256(sql.strip().lower().encode()).hexdigest()[:16]
    normalized = normalize_sql(sql)

    with quiet_sqlglot():
        for dialect in _DIALECTS:
            refs = table_refs_from_sql(
                normalized, dialect=dialect, default_schema=default_schema
            )
            if refs and _refs_look_valid(refs):
                targets = dml_write_targets(
                    normalized, dialect=dialect, default_schema=default_schema
                )
                return ParsedSql(
                    table_refs=refs,
                    write_targets=targets,
                    dialect_used=dialect,
                    fingerprint=fingerprint,
                )

    return ParsedSql(fingerprint=fingerprint)
