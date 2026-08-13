"""Write ExternalConsumer nodes and CONSUMES edges into the Kuzu graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pretensor.core.ids import (
    consumes_edge_id,
    external_consumer_node_id,
    table_node_id,
)
from pretensor.core.store import KuzuStore
from pretensor.enrichment.analyze.classify import CONFIDENCE_FLOAT
from pretensor.enrichment.analyze.extract_python import SqlCandidate
from pretensor.enrichment.analyze.parse import ParsedSql
from pretensor.enrichment.analyze.summary import ConsumerRow
from pretensor.graph_models.consumer import ConsumesEdge, ExternalConsumerNode

__all__ = ["ParsedCandidate", "WriteSummary", "write_consumers_for_run"]

logger = logging.getLogger(__name__)

_CONSUMES_SOURCE = "analyze"


@dataclass
class ParsedCandidate:
    """A SQL candidate paired with its parsed table refs and per-file provenance."""

    candidate: SqlCandidate
    parsed: ParsedSql
    rel_path: str  # repo-relative path (never absolute)
    language: str


@dataclass
class WriteSummary:
    """Counts and planned/written rows from ``write_consumers_for_run``."""

    consumers_written: int
    edges_written: int
    cross_connection_dropped: int
    rows: tuple[ConsumerRow, ...] = ()


def write_consumers_for_run(
    store: KuzuStore,
    connection_name: str,
    candidates: list[ParsedCandidate],
    scan_run_id: str,
    *,
    service_name: str,
    min_confidence: float = 0.6,
    dry_run: bool = False,
) -> WriteSummary:
    """Upsert ``ExternalConsumer`` nodes and ``CONSUMES`` edges for one scan run.

    For each candidate:
    - Splits parsed refs into reads (``table_refs`` minus write targets) and
      writes (``write_targets``), so each CONSUMES edge carries a truthful ``op``.
    - Resolves each ``(schema, table)`` ref to an existing ``SchemaTable`` node id;
      unresolved (cross-connection or unknown) refs are dropped with a counter.
    - Upserts one ``ExternalConsumer`` node (no raw SQL stored, only the fingerprint).
    - Upserts one ``CONSUMES`` edge per resolved ``(table, op)``.
    - Marks ``has_external_consumers = true`` on every touched ``SchemaTable``.

    After all candidates, sweeps ``ExternalConsumer`` rows (and stale
    ``CONSUMES`` edges) for this ``service_name``/``connection_name`` carrying a
    different ``scan_run_id``, and clears ``has_external_consumers`` on tables
    that lost their last consumer.
    """
    known_tables = _known_table_ids(store, connection_name)
    folded_known = _casefold_index(known_tables)
    written_consumers: set[str] = set()
    written_edges: set[str] = set()
    cross_connection_dropped = 0
    touched_tables: set[str] = set()
    rows: list[ConsumerRow] = []

    for pc in candidates:
        cand = pc.candidate
        parsed = pc.parsed

        confidence = CONFIDENCE_FLOAT.get(cand.confidence_bucket, 0.0)
        if confidence < min_confidence:
            continue

        write_set = _dedup_refs(parsed.write_targets)
        read_set = [r for r in _dedup_refs(parsed.table_refs) if r not in write_set]

        resolved: list[tuple[str, str, str]] = []  # (table_node_id, op, "schema.table")
        for refs, op in ((read_set, "read"), (write_set, "write")):
            for schema, table in refs:
                nid = table_node_id(connection_name, schema, table)
                if nid not in known_tables:
                    nid = folded_known.get(nid.casefold(), "")
                if not nid:
                    logger.debug(
                        "analyze writers: unresolved ref %s.%s for connection %s (dropped)",
                        schema,
                        table,
                        connection_name,
                    )
                    cross_connection_dropped += 1
                    continue
                _, canon_schema, canon_table = nid.split("::", 2)
                resolved.append((nid, op, f"{canon_schema}.{canon_table}"))

        if not resolved:
            continue

        consumer_nid = external_consumer_node_id(
            connection_name,
            service_name,
            pc.rel_path,
            cand.line_start,
            parsed.fingerprint,
        )
        if not dry_run:
            store.upsert_external_consumer(
                ExternalConsumerNode(
                    node_id=consumer_nid,
                    connection_name=connection_name,
                    service_name=service_name,
                    file_path=pc.rel_path,
                    language=pc.language,
                    symbol=cand.symbol,
                    kind=cand.kind,
                    line_start=cand.line_start,
                    line_end=cand.line_end,
                    sql_fingerprint=parsed.fingerprint,
                    confidence=confidence,
                    dialect_used=parsed.dialect_used,
                    scan_run_id=scan_run_id,
                )
            )
        written_consumers.add(consumer_nid)

        for table_nid, op, table_label in resolved:
            eid = consumes_edge_id(consumer_nid, table_nid, op)
            if eid in written_edges:
                continue
            if not dry_run:
                store.upsert_consumes_edge(
                    ConsumesEdge(
                        edge_id=eid,
                        source_node_id=consumer_nid,
                        target_node_id=table_nid,
                        op=op,
                        source=_CONSUMES_SOURCE,
                        confidence=confidence,
                        scan_run_id=scan_run_id,
                    )
                )
            touched_tables.add(table_nid)
            written_edges.add(eid)
            rows.append(
                ConsumerRow(
                    service_name=service_name,
                    table=table_label,
                    op=op,
                    file_path=pc.rel_path,
                    line_start=cand.line_start,
                    line_end=cand.line_end,
                    confidence=confidence,
                )
            )

    if not dry_run:
        store.mark_tables_have_external_consumers(sorted(touched_tables))
        store.sweep_stale_consumers(service_name, connection_name, scan_run_id)

    return WriteSummary(
        consumers_written=len(written_consumers),
        edges_written=len(written_edges),
        cross_connection_dropped=cross_connection_dropped,
        rows=tuple(rows),
    )


def _known_table_ids(store: KuzuStore, connection_name: str) -> set[str]:
    """All ``SchemaTable`` node ids indexed under ``connection_name``, in one query."""
    rows = store.query_all_rows(
        "MATCH (t:SchemaTable {connection_name: $cn}) RETURN t.node_id",
        {"cn": connection_name},
    )
    return {str(r[0]) for r in rows}


def _casefold_index(known: set[str]) -> dict[str, str]:
    """Casefolded node id → node id, for unique case-insensitive fallback.

    Snowflake stores unquoted identifiers uppercase while application SQL is
    conventionally lowercase, so an exact-match-only lookup drops every ref.
    A ref that misses exactly resolves case-insensitively when precisely one
    stored table matches; ambiguous folds (case-variant twins) stay dropped.
    """
    index: dict[str, str | None] = {}
    for nid in known:
        key = nid.casefold()
        index[key] = None if key in index else nid
    return {k: v for k, v in index.items() if v is not None}


def _dedup_refs(refs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deduplicate (schema, table) pairs while preserving order."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out
