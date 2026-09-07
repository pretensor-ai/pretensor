"""Orchestrate the analyze enrichment pipeline (walker → extractor → writer)."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from pretensor.core.store import KuzuStore
from pretensor.enrichment.analyze.classify import CONFIDENCE_FLOAT, classify_sql
from pretensor.enrichment.analyze.extract_python import (
    SqlCandidate,
    extract_sql_candidates,
)
from pretensor.enrichment.analyze.extract_sql_file import (
    extract_sql_file_candidates,
    parse_sql_file,
)
from pretensor.enrichment.analyze.parse import parse_sql
from pretensor.enrichment.analyze.summary import AnalyzeSummary
from pretensor.enrichment.analyze.walker import walk_repo
from pretensor.enrichment.analyze.writers import (
    ParsedCandidate,
    write_consumers_for_run,
)
from pretensor.observability import log_timed_operation

__all__ = ["EmptyGraphError", "resolve_default_schema", "run_analyze_enrichment"]

logger = logging.getLogger(__name__)

_FSTRING_CONFIDENCE_BUCKET = "low"

_EXTRACTORS: dict[str, Callable[[Path], list[SqlCandidate]]] = {
    "python": extract_sql_candidates,
    "sql": extract_sql_file_candidates,
}

_FALLBACK_DEFAULT_SCHEMA = "public"

# Conventional default schemas per dialect, used when the graph snapshot spans
# multiple schemas and the dialect has a fixed convention. MySQL and BigQuery
# are handled separately: their "default schema" is the connection's database
# (MySQL schemas ARE databases; BigQuery datasets play the schema role).
_DIALECT_DEFAULT_SCHEMAS = {
    "postgres": "public",
    "snowflake": "PUBLIC",
}


class EmptyGraphError(ValueError):
    """The graph has no ``SchemaTable`` rows for the requested connection."""


def resolve_default_schema(
    store: KuzuStore,
    connection_name: str,
    *,
    dialect: str | None = None,
    database: str | None = None,
) -> str:
    """Derive the schema unqualified table refs resolve against.

    Resolution order:

    1. Graph snapshot: when the connection's indexed ``SchemaTable`` rows span
       exactly one schema, use it — every unqualified ref can only live there.
    2. Connection metadata (registry ``dialect`` / ``database``):
       ``postgres`` → ``"public"``, ``snowflake`` → ``"PUBLIC"``,
       ``mysql`` → the database name, ``bigquery`` → the dataset portion of
       the database (``"project/dataset"``).
    3. ``"public"`` as the final fallback.
    """
    rows = store.query_all_rows(
        "MATCH (t:SchemaTable {connection_name: $cn}) RETURN DISTINCT t.schema_name",
        {"cn": connection_name},
    )
    schemas = {str(r[0]) for r in rows if r and r[0]}
    if len(schemas) == 1:
        return next(iter(schemas))

    db = (database or "").strip()
    if dialect == "mysql" and db:
        return db
    if dialect == "bigquery" and db:
        return db.split("/", 1)[1] if "/" in db else db
    return _DIALECT_DEFAULT_SCHEMAS.get(dialect or "", _FALLBACK_DEFAULT_SCHEMA)


def run_analyze_enrichment(
    repo_path: Path,
    store: KuzuStore,
    connection_name: str,
    *,
    service_name: str | None = None,
    includes: Sequence[str] = (),
    excludes: Sequence[str] = (),
    max_file_bytes: int = 1_000_000,
    min_confidence: float = 0.6,
    default_schema: str | None = None,
    dialect: str | None = None,
    database: str | None = None,
    scan_run_id: str | None = None,
    dry_run: bool = False,
) -> AnalyzeSummary:
    """Walk ``repo_path``, extract SQL from Python and ``.sql`` files, and write
    graph nodes/edges.

    Hard-errors when the graph has no ``SchemaTable`` rows for ``connection_name``
    (the user must run ``pretensor index --connection <name>`` first).

    Args:
        repo_path: Root of the repository to scan.
        store: Open Kuzu graph store.
        connection_name: Pretensor connection name; only tables indexed under this
            connection are considered valid CONSUMES targets.
        service_name: Label for the scanned service (defaults to ``repo_path.name``).
        includes: Gitignore-style glob patterns limiting which files are scanned.
        excludes: Gitignore-style glob patterns excluding files from scanning.
        max_file_bytes: Skip files larger than this (default 1 MB).
        min_confidence: Drop candidates below this confidence level (default 0.6,
            which excludes ``low``-confidence f-string prefix candidates).
        default_schema: Schema assumed for unqualified table references in the
            scanned SQL. When ``None`` (the default), it is derived via
            :func:`resolve_default_schema` from the graph snapshot and the
            connection metadata, falling back to ``"public"``.
        dialect: Connection dialect from the registry entry (``"postgres"``,
            ``"mysql"``, ``"snowflake"``, ``"bigquery"``); only consulted when
            ``default_schema`` is ``None``.
        database: Logical database name from the registry entry; only consulted
            when ``default_schema`` is ``None``.
        scan_run_id: Idempotency key for this run; stale consumers from other run IDs
            are swept at the end. Defaults to a random 8-hex-char string.

    Returns:
        ``AnalyzeSummary`` with counts and timing.
    """
    _guard_non_empty_graph(store, connection_name)

    effective_schema = (
        default_schema
        if default_schema is not None
        else resolve_default_schema(
            store, connection_name, dialect=dialect, database=database
        )
    )
    effective_service = service_name or repo_path.name
    effective_run_id = scan_run_id or uuid.uuid4().hex[:8]

    started = time.perf_counter()
    files_scanned = 0
    sql_candidates_found = 0
    sql_candidates_failed = 0
    parsed_candidates: list[ParsedCandidate] = []

    with log_timed_operation(
        logger,
        event="analyze.walk",
        connection_name=connection_name,
        repo_path=str(repo_path),
    ):
        file_list = list(
            walk_repo(
                repo_path,
                includes=includes,
                excludes=excludes,
                max_file_bytes=max_file_bytes,
            )
        )

    with log_timed_operation(
        logger,
        event="analyze.extract",
        connection_name=connection_name,
        file_count=len(file_list),
    ):
        for file_path, language in file_list:
            files_scanned += 1
            extractor = _EXTRACTORS.get(language)
            if extractor is None:
                continue

            raw_candidates = extractor(file_path)
            for cand in raw_candidates:
                sql_candidates_found += 1

                if cand.kind == "sql_file":
                    # A bare .sql file is SQL by construction: high confidence,
                    # no keyword classifier (files may open with comments).
                    bucket = "high"
                    conf = CONFIDENCE_FLOAT[bucket]
                else:
                    bucket, conf = classify_sql(cand.text)
                    if not bucket:
                        continue

                    if cand.kind == "fstring":
                        bucket = _FSTRING_CONFIDENCE_BUCKET
                        conf = CONFIDENCE_FLOAT[bucket]
                cand.confidence_bucket = bucket

                if conf < min_confidence:
                    continue

                try:
                    if cand.kind == "sql_file":
                        parsed = parse_sql_file(
                            cand.text, default_schema=effective_schema
                        )
                    else:
                        parsed = parse_sql(cand.text, default_schema=effective_schema)
                except Exception as exc:
                    logger.debug(
                        "analyze: parse error for candidate in %s: %s", file_path, exc
                    )
                    sql_candidates_failed += 1
                    continue

                if not parsed.table_refs and not parsed.write_targets:
                    continue

                try:
                    rel_path = str(file_path.relative_to(repo_path))
                except ValueError:
                    rel_path = file_path.name

                parsed_candidates.append(
                    ParsedCandidate(
                        candidate=cand,
                        parsed=parsed,
                        rel_path=rel_path,
                        language=language,
                    )
                )

    with log_timed_operation(
        logger,
        event="analyze.write",
        connection_name=connection_name,
        candidate_count=len(parsed_candidates),
    ):
        write_summary = write_consumers_for_run(
            store,
            connection_name,
            parsed_candidates,
            effective_run_id,
            service_name=effective_service,
            min_confidence=min_confidence,
            dry_run=dry_run,
        )

    duration_ms = (time.perf_counter() - started) * 1000

    return AnalyzeSummary(
        files_scanned=files_scanned,
        sql_candidates_found=sql_candidates_found,
        sql_candidates_parsed=len(parsed_candidates),
        sql_candidates_failed=sql_candidates_failed,
        consumers_written=write_summary.consumers_written,
        edges_written=write_summary.edges_written,
        cross_connection_dropped=write_summary.cross_connection_dropped,
        scan_run_id=effective_run_id,
        duration_ms=duration_ms,
        default_schema=effective_schema,
        rows=write_summary.rows,
    )


def _guard_non_empty_graph(store: KuzuStore, connection_name: str) -> None:
    """Raise ``EmptyGraphError`` when no ``SchemaTable`` rows exist for the connection."""
    rows = store.query_all_rows(
        "MATCH (t:SchemaTable {connection_name: $cn}) RETURN count(*) LIMIT 1",
        {"cn": connection_name},
    )
    count = int(rows[0][0]) if rows and rows[0][0] is not None else 0
    if count == 0:
        raise EmptyGraphError(
            f"No SchemaTable nodes found for connection '{connection_name}'. "
            f"Index the database first: pretensor index --connection {connection_name}"
        )
