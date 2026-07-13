"""Shared ranking helpers for the ``query`` (hybrid) and ``semantic_search`` tools.

Module-private to the ``pretensor.mcp.tools`` package: centralizes the
``KuzuStore.iter_table_embeddings`` + ``cosine_similarity`` scan
so the hybrid BM25+cosine fusion path in ``query`` and the
cosine-only ``semantic_search`` tool share one implementation.

Also hosts the Reciprocal Rank Fusion helper and its ``RRF_K`` constant.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from pretensor.core.registry import RegistryEntry
from pretensor.core.store import TableEmbeddingRow
from pretensor.intelligence.embeddings import cosine_similarity
from pretensor.visibility.filter import VisibilityFilter

from ..service_registry import open_store_for_entry, release_store

__all__ = [
    "RRF_K",
    "CosineHit",
    "cosine_rank_tables",
    "fusion_key",
    "rrf_fuse",
    "score_embedding_rows",
]

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant (Cormack, Clarke & Büttcher, 2009).
# 60 is the value from the canonical paper; kept as a named constant so a
# future tweak (e.g. after an L2 benchmark regression) is a one-line change.
RRF_K: int = 60


@dataclass(frozen=True, slots=True)
class CosineHit:
    """One cosine-ranked ``SchemaTable`` hit with its fusion key pre-computed."""

    key: str
    node_id: str
    connection_name: str
    database: str
    schema_name: str
    table_name: str
    description: str
    score: float
    cluster_id: str | None


def fusion_key(
    *,
    connection_name: str,
    database: str,
    schema_name: str,
    table_name: str,
) -> str:
    """Stable key shared by BM25 and cosine rankings for RRF fusion.

    The pipe delimiter cannot collide with legal identifier characters, so
    ``"a|b|c.d"`` maps 1:1 to ``(a, b, c, d)``.
    """
    return f"{connection_name}|{database}|{schema_name}.{table_name}"


def score_embedding_rows(
    rows: Iterable[TableEmbeddingRow],
    qvec: list[float],
    *,
    visibility_filter: VisibilityFilter | None,
) -> tuple[list[CosineHit], bool]:
    """Score a stream of ``TableEmbeddingRow`` against ``qvec``.

    Shared inner loop used by :func:`cosine_rank_tables` (which chains rows
    from freshly-opened stores) and by ``context.similar_tables_for_table``
    (which feeds rows from an already-open ``KuzuStore``). Rows without an
    embedding are skipped; rows filtered by ``visibility_filter`` are
    dropped **after** flipping ``any_candidate_had_vector`` so callers can
    distinguish "no tables carry vectors" from "hidden by visibility".

    Returns ``(scored, any_candidate_had_vector)`` **unsorted and
    un-truncated** — callers sort + slice to taste.
    """
    # ``KuzuStore.iter_table_embeddings(cluster=None)`` joins through
    # ``IN_CLUSTER`` via OPTIONAL MATCH, which yields one row per cluster
    # a table belongs to.  A table in N clusters would therefore show up
    # N times here, double-counting in downstream RRF fusion and clobbering
    # the cosine_by_key dict in ``_maybe_fuse``.  Dedup by node_id, keeping
    # the first occurrence — the embedding (and therefore the cosine
    # score) is identical across the duplicate rows; only ``cluster_id``
    # varies and we already preserve the first cluster encountered.
    scored: list[CosineHit] = []
    seen_nids: set[str] = set()
    any_candidate_had_vector = False

    for row in rows:
        if row.embedding is None:
            continue
        if row.node_id in seen_nids:
            continue
        # Mark the node as seen regardless of whether we end up emitting
        # a hit for it. A subsequent (table, cluster) row for the same
        # ``node_id`` would otherwise bypass the dedup — Kuzu's OPTIONAL
        # MATCH yields one row per cluster the table is in, and we want
        # exactly one decision per table here.
        seen_nids.add(row.node_id)
        any_candidate_had_vector = True
        if visibility_filter is not None and not visibility_filter.is_table_visible(
            row.connection_name, row.schema_name, row.table_name
        ):
            continue
        try:
            score = cosine_similarity(qvec, row.embedding)
        except ValueError as exc:
            logger.warning(
                "score_embedding_rows: dim mismatch for %s: %s",
                row.node_id,
                exc,
            )
            continue
        scored.append(
            CosineHit(
                key=fusion_key(
                    connection_name=row.connection_name,
                    database=row.database,
                    schema_name=row.schema_name,
                    table_name=row.table_name,
                ),
                node_id=row.node_id,
                connection_name=row.connection_name,
                database=row.database,
                schema_name=row.schema_name,
                table_name=row.table_name,
                description=row.description,
                score=score,
                cluster_id=row.cluster_id,
            )
        )

    return scored, any_candidate_had_vector


def any_entry_has_vectors(entries: Iterable[RegistryEntry]) -> bool:
    """Cheap LIMIT-1 probe: does any entry's store carry a table vector?

    Callers run this BEFORE embedding the query text so a store with no
    vectors never pays the embedding-client cost — on a cold process the
    first ``embed()`` call triggers the ONNX model download + session
    init, which must not happen when the result would be discarded by the
    "no tables carry vectors" fallback anyway.
    """
    for entry in entries:
        try:
            store = open_store_for_entry(entry)
        except RuntimeError as exc:
            logger.warning(
                "any_entry_has_vectors: skipping %s (%s)",
                entry.connection_name,
                exc,
            )
            continue
        try:
            if store.has_any_table_embeddings():
                return True
        finally:
            release_store(store)
    return False


def cosine_rank_tables(
    entries: Iterable[RegistryEntry],
    qvec: list[float],
    *,
    database: str | None,
    cluster: str | None,
    visibility_filter: VisibilityFilter | None,
    k: int,
) -> tuple[list[CosineHit], bool]:
    """Top-``k`` ``SchemaTable`` rows by cosine similarity to ``qvec``.

    Iterates ``KuzuStore.iter_table_embeddings`` for each registry entry,
    applies the ``visibility_filter``, scores rows whose ``embedding`` is
    present, sorts descending by score, and truncates to ``k``.

    Returns ``(hits, any_candidate_had_vector)``. The boolean preserves the
    "no tables carry vectors" fallback signal without a second store scan —
    callers in ``query`` and ``semantic_search`` use it to route to the
    documented fallback envelope instead of returning an empty result set
    that looks indistinguishable from "scored but nothing matched".

    Entry selection (including the "unknown database" error path) is left to
    callers since ``semantic_search`` and ``query`` want different behavior
    there (error envelope vs. silent empty fusion).
    """
    scored: list[CosineHit] = []
    any_candidate_had_vector = False

    for entry in entries:
        try:
            store = open_store_for_entry(entry)
        except RuntimeError as exc:
            logger.warning(
                "cosine_rank_tables: skipping %s (%s)", entry.connection_name, exc
            )
            continue
        try:
            entry_scored, entry_had_vector = score_embedding_rows(
                store.iter_table_embeddings(database=database, cluster=cluster),
                qvec,
                visibility_filter=visibility_filter,
            )
        finally:
            release_store(store)
        scored.extend(entry_scored)
        any_candidate_had_vector = any_candidate_had_vector or entry_had_vector

    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:k], any_candidate_had_vector


def rrf_fuse(
    *,
    bm25_keys: list[str],
    cosine_keys: list[str],
    k_constant: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of two ranked key lists.

    Input lists are ordered best-first (rank 1 = index 0). Returns a list of
    ``(key, fused_score)`` ordered by fused score descending. A key appearing
    in only one list contributes only that list's term — missing means rank
    infinity, i.e. zero contribution.
    """
    scores: dict[str, float] = {}
    for rank, key in enumerate(bm25_keys, start=1):
        scores[key] = scores.get(key, 0.0) + 1.0 / (k_constant + rank)
    for rank, key in enumerate(cosine_keys, start=1):
        scores[key] = scores.get(key, 0.0) + 1.0 / (k_constant + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
