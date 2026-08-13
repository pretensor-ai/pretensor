"""SQL fingerprint classifier: keyword regex + trial sqlglot parse."""

from __future__ import annotations

import re

import sqlglot
from sqlglot.errors import ErrorLevel

from pretensor.enrichment.analyze.parse import normalize_sql, quiet_sqlglot

__all__ = ["CONFIDENCE_FLOAT", "classify_sql"]

_SQL_START_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|MERGE|WITH|REPLACE)\b", re.IGNORECASE
)

CONFIDENCE_FLOAT: dict[str, float] = {
    "high": 0.9,
    "medium": 0.7,
    "low": 0.5,
}


def classify_sql(text: str) -> tuple[str, float]:
    """Return ``(confidence_bucket, confidence_float)`` or ``('', 0.0)`` if not SQL.

    Steps:
    1. If ``text`` does not start with a SQL keyword, return ``('', 0.0)``.
    2. If sqlglot can parse it without raising, return ``('high', 0.9)``.
    3. Keyword match only → ``('medium', 0.7)``.

    DBAPI bind placeholders (``%s``, ``%(name)s``, ``:name``, ``?``) are
    normalized away before the trial parse, so parameterized statements — the
    norm in scanned application code — classify on their structure. MySQL
    ``REPLACE INTO`` is rewritten to ``INSERT INTO`` first (sqlglot cannot
    parse it in any dialect).
    """
    if not text or not text.strip():
        return "", 0.0
    if not _SQL_START_RE.match(text):
        return "", 0.0
    try:
        with quiet_sqlglot():
            sqlglot.parse_one(normalize_sql(text), error_level=ErrorLevel.RAISE)
        bucket = "high"
    except Exception:
        bucket = "medium"
    return bucket, CONFIDENCE_FLOAT[bucket]
