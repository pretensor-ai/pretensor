"""Adapter treating bare ``.sql`` files as high-confidence SQL candidates."""

from __future__ import annotations

import logging
from pathlib import Path

import sqlglot
from sqlglot.errors import ErrorLevel

from pretensor.enrichment.analyze.extract_python import SqlCandidate
from pretensor.enrichment.analyze.parse import (
    ParsedSql,
    normalize_sql,
    parse_sql,
    quiet_sqlglot,
)

__all__ = ["SqlFileParseError", "extract_sql_file_candidates", "parse_sql_file"]

logger = logging.getLogger(__name__)


class SqlFileParseError(ValueError):
    """No statement in a ``.sql`` file could be parsed as SQL."""


def extract_sql_file_candidates(file_path: Path) -> list[SqlCandidate]:
    """Return one whole-file ``SqlCandidate`` for a bare ``.sql`` file.

    A ``.sql`` file is SQL by construction, so the candidate is emitted at
    ``high`` confidence with ``kind="sql_file"`` and the file basename as its
    symbol. The keyword classifier is skipped downstream — files may legally
    open with comments that would fail the SQL-keyword prefix check.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("extract_sql_file: cannot read %s: %s", file_path, exc)
        return []
    if not text.strip():
        return []
    return [
        SqlCandidate(
            text=text,
            confidence_bucket="high",
            line_start=1,
            line_end=len(text.splitlines()),
            file_path=file_path,
            symbol=file_path.name,
            kind="sql_file",
        )
    ]


def parse_sql_file(text: str, *, default_schema: str = "public") -> ParsedSql:
    """Parse a whole ``.sql`` file's contents, unioning refs across statements.

    ``parse_sql`` already accepts multi-statement input — sqlglot parses
    semicolon-separated statements into one block and ``table_refs_from_sql`` /
    ``dml_write_targets`` walk every statement — so the union of table refs
    falls out of a single call. Raises ``SqlFileParseError`` when the file
    yields no table refs *and* does not parse as SQL at all, so the pipeline
    can count it as a failed candidate instead of silently skipping it.
    """
    parsed = parse_sql(text, default_schema=default_schema)
    if parsed.table_refs or parsed.write_targets:
        return parsed
    if not _parses_as_sql(text):
        raise SqlFileParseError("no statement in file parses as SQL")
    # Valid SQL that references no tables (e.g. ``SELECT 1``): zero edges,
    # but not a parse failure.
    return parsed


def _parses_as_sql(text: str) -> bool:
    """Trial-parse (post-normalization), mirroring ``classify_sql``'s check."""
    with quiet_sqlglot():
        try:
            sqlglot.parse_one(normalize_sql(text), error_level=ErrorLevel.RAISE)
        except Exception:
            return False
    return True
