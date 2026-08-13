"""Frozen summary dataclasses for the analyze enrichment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AnalyzeSummary", "ConsumerRow"]


@dataclass(frozen=True, slots=True)
class ConsumerRow:
    """One planned/written CONSUMES edge, for CLI rendering (no raw SQL)."""

    service_name: str
    table: str  # "schema.table" as resolved
    op: str  # "read" | "write"
    file_path: str  # repo-relative
    line_start: int
    line_end: int
    confidence: float


@dataclass(frozen=True, slots=True)
class AnalyzeSummary:
    """Aggregate counts returned by ``run_analyze_enrichment``."""

    files_scanned: int
    sql_candidates_found: int
    sql_candidates_parsed: int
    sql_candidates_failed: int
    consumers_written: int
    edges_written: int
    cross_connection_dropped: int
    scan_run_id: str
    duration_ms: float
    # Planned (dry-run) or written CONSUMES rows, for CLI rendering. Defaulted so
    # existing keyword construction and the frozen field set stay compatible.
    rows: tuple[ConsumerRow, ...] = field(default=())
