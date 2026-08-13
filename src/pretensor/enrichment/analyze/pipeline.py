"""Orchestrate the analyze enrichment pipeline (walker → extractor → writer)."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from pretensor.core.store import KuzuStore
from pretensor.enrichment.analyze.classify import CONFIDENCE_FLOAT, classify_sql
from pretensor.enrichment.analyze.extract_python import extract_sql_candidates
from pretensor.enrichment.analyze.parse import parse_sql
from pretensor.enrichment.analyze.summary import AnalyzeSummary
from pretensor.enrichment.analyze.walker import walk_repo
from pretensor.enrichment.analyze.writers import (
    ParsedCandidate,
    write_consumers_for_run,
)
from pretensor.observability import log_timed_operation

__all__ = ["EmptyGraphError", "run_analyze_enrichment"]

logger = logging.getLogger(__name__)

_FSTRING_CONFIDENCE_BUCKET = "low"


class EmptyGraphError(ValueError):
    """The graph has no ``SchemaTable`` rows for the requested connection."""


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
    default_schema: str = "public",
    scan_run_id: str | None = None,
    dry_run: bool = False,
) -> AnalyzeSummary:
    """Walk ``repo_path``, extract SQL from Python files, and write graph nodes/edges.

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
            scanned SQL (default ``"public"``; pass the connection's actual
            default schema for non-Postgres dialects).
        scan_run_id: Idempotency key for this run; stale consumers from other run IDs
            are swept at the end. Defaults to a random 8-hex-char string.

    Returns:
        ``AnalyzeSummary`` with counts and timing.
    """
    _guard_non_empty_graph(store, connection_name)

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
            if language != "python":
                continue

            raw_candidates = extract_sql_candidates(file_path)
            for cand in raw_candidates:
                sql_candidates_found += 1

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
                    parsed = parse_sql(cand.text, default_schema=default_schema)
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
