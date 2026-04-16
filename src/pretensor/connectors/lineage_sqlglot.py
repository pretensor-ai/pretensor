"""Shared sqlglot helpers for structural table references (no literal capture)."""

from __future__ import annotations

import logging
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel

logger = logging.getLogger(__name__)


def table_refs_from_sql(
    sql: str,
    *,
    dialect: str,
    default_schema: str,
) -> list[tuple[str, str]]:
    """Return ``(schema, table)`` pairs from SELECT/DML; empty on parse failure."""
    if not sql or not sql.strip():
        return []
    try:
        parsed = sqlglot.parse_one(
            sql, dialect=dialect, error_level=ErrorLevel.WARN
        )
    except Exception as exc:
        logger.warning("sqlglot parse failed (%s): %s", dialect, exc)
        return []
    return list(_tables_from_expression(parsed, default_schema=default_schema))


def _schema_for_table(t: exp.Table, default_schema: str) -> str:
    db = t.db
    if db:
        return str(db)
    cat = t.catalog
    if cat:
        return str(cat)
    return default_schema


def _tables_from_expression(expr: Any, *, default_schema: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for t in expr.find_all(exp.Table):
        name = t.name
        if not name:
            continue
        out.append((_schema_for_table(t, default_schema), str(name)))
    return out


def dml_write_targets(
    sql: str,
    *,
    dialect: str,
    default_schema: str,
) -> list[tuple[str, str]]:
    """Tables targeted by INSERT/UPDATE/DELETE/MERGE (best-effort)."""
    if not sql or not sql.strip():
        return []
    try:
        parsed = sqlglot.parse_one(
            sql, dialect=dialect, error_level=ErrorLevel.WARN
        )
    except Exception as exc:
        logger.warning("sqlglot parse failed (%s): %s", dialect, exc)
        return []
    return _write_targets_from_expr(parsed, default_schema=default_schema)


def _write_targets_from_expr(expr: Any, *, default_schema: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ins in expr.find_all(exp.Insert):
        tbl = ins.this
        if isinstance(tbl, exp.Table):
            out.append((_schema_for_table(tbl, default_schema), str(tbl.name)))
    for upd in expr.find_all(exp.Update):
        tbl = upd.this
        if isinstance(tbl, exp.Table):
            out.append((_schema_for_table(tbl, default_schema), str(tbl.name)))
    for mer in expr.find_all(exp.Merge):
        tgt = mer.this
        if isinstance(tgt, exp.Table):
            out.append((_schema_for_table(tgt, default_schema), str(tgt.name)))
    for delete in expr.find_all(exp.Delete):
        tbl = delete.this
        if isinstance(tbl, exp.Table):
            out.append((_schema_for_table(tbl, default_schema), str(tbl.name)))
    return out
