"""Embedding-based ``RelationshipScorer``.

Walks every pair of tables that carry a non-null ``SchemaTable.embedding``
in the store, computes cosine similarity, and emits ``INFERRED_JOIN``
candidates with ``source='embedding'`` and ``status='suggested'`` for
pairs whose cosine ≥ ``threshold`` and whose implied join columns pass
the heuristic's type-family compatibility check.

Determinism contract:

* Opt-in via ``EmbeddingsConfig.join_threshold`` — ``None`` skips the
  scorer entirely; the scorer registry behaves byte-identically to the
  pre-embedding pipeline output.
* Additive only: candidates pass through the same combiner / shadow-alias
  filter / explicit-FK dedup as heuristic candidates.  An embedding-only
  proposal that conflicts with an explicit FK is dropped at the
  ``discovery.discover`` filter step (line 82); the scorer also skips
  ``JoinKey``s already in the explicit-FK set up front.
* Type-family veto: every emitted candidate has ``types_compatible``
  green for its ``(source_column, target_column)`` pair, reusing the
  heuristic's check.

Index-time staging: the scorer reads embeddings from the store.  The
index/reindex paths call ``compute_table_embeddings`` right after schema
rows are written and *before* relationship discovery, so the scorer sees
this run's vectors even on the very first index.  Callers that drive
``run_intelligence_layer`` directly without precomputing still get the
in-pipeline ``embedding_index`` step, in which case discovery (which runs
earlier) only sees vectors from the previous cycle.
"""

from __future__ import annotations

import logging

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.embeddings import (
    cosine_similarity,
    embeddings_disabled_via_env,
)
from pretensor.intelligence.heuristic import types_compatible
from pretensor.intelligence.scoring import JoinKey, RelationshipScorer, ScorerRegistry

__all__ = ["EmbeddingRelationshipScorer", "extend_with_embedding_scorer"]

logger = logging.getLogger(__name__)


class EmbeddingRelationshipScorer(RelationshipScorer):
    """Cosine-similarity-driven candidate generator for ``INFERRED_JOIN``."""

    def __init__(
        self,
        *,
        store: KuzuStore,
        database_key: str,
        threshold: float,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            msg = f"threshold must be in [0.0, 1.0], got {threshold}"
            raise ValueError(msg)
        self._store = store
        self._database_key = database_key
        self._threshold = threshold

    def name(self) -> str:
        return "embedding"

    def score(
        self,
        snapshot: SchemaSnapshot,
        explicit_fk_keys: set[JoinKey],
    ) -> list[RelationshipCandidate]:
        embeddings = self._load_embeddings()
        if not embeddings:
            logger.debug(
                "EmbeddingRelationshipScorer: no embedded tables for %s; skipping",
                self._database_key,
            )
            return []

        table_by_nid = self._index_snapshot_tables(snapshot)
        if not table_by_nid:
            return []

        candidates: list[RelationshipCandidate] = []
        embedded_ids = sorted(nid for nid in embeddings if nid in table_by_nid)
        for i in range(len(embedded_ids)):
            a_id = embedded_ids[i]
            for j in range(i + 1, len(embedded_ids)):
                b_id = embedded_ids[j]
                cos = cosine_similarity(embeddings[a_id], embeddings[b_id])
                if cos < self._threshold:
                    continue
                a_tbl = table_by_nid[a_id]
                b_tbl = table_by_nid[b_id]

                # Both directions: a → b (a is source, b is target) and b → a.
                # The combiner downstream picks the highest-confidence winner
                # per directed JoinKey, so emitting both is safe.
                for src_tbl, dst_tbl, src_id, dst_id in (
                    (a_tbl, b_tbl, a_id, b_id),
                    (b_tbl, a_tbl, b_id, a_id),
                ):
                    pair = _pick_join_columns(src_tbl, dst_tbl)
                    if pair is None:
                        continue
                    src_col, dst_col = pair
                    join_key: JoinKey = (src_id, dst_id, src_col, dst_col)
                    if join_key in explicit_fk_keys:
                        continue
                    candidate_id = (
                        f"embedding::{src_id}::{src_col}::{dst_id}::{dst_col}"
                    )
                    candidates.append(
                        RelationshipCandidate(
                            candidate_id=candidate_id,
                            source_node_id=src_id,
                            target_node_id=dst_id,
                            source_column=src_col,
                            target_column=dst_col,
                            source="embedding",
                            confidence=float(cos),
                            status="suggested",
                            reasoning=(
                                f"cosine={cos:.3f} between {src_tbl.schema_name}."
                                f"{src_tbl.name} and {dst_tbl.schema_name}.{dst_tbl.name}"
                            ),
                        )
                    )
        logger.debug(
            "EmbeddingRelationshipScorer: emitted %d candidates for %s "
            "(threshold=%.3f, embedded_tables=%d)",
            len(candidates),
            self._database_key,
            self._threshold,
            len(embedded_ids),
        )
        return candidates

    def _load_embeddings(self) -> dict[str, list[float]]:
        """One-shot fetch of every embedded ``SchemaTable`` in the database."""
        out: dict[str, list[float]] = {}
        for row in self._store.iter_table_embeddings(database=self._database_key):
            if row.embedding is None:
                continue
            out[row.node_id] = [float(x) for x in row.embedding]
        return out

    def _index_snapshot_tables(self, snapshot: SchemaSnapshot) -> dict[str, Table]:
        """Map ``SchemaTable.node_id`` → ``Table`` for column lookups."""
        out: dict[str, Table] = {}
        for tbl in snapshot.tables:
            nid = table_node_id(snapshot.connection_name, tbl.schema_name, tbl.name)
            out[nid] = tbl
        return out


def _pick_join_columns(src: Table, dst: Table) -> tuple[str, str] | None:
    """Pick a likely (src_col, dst_col) join pair for an embedding candidate.

    Strategy:
    1.  Prefer the canonical FK shape: ``src.<dst_table>_id`` → ``dst.id``.
    2.  Fall back to any type-family-compatible ``(FK-like, PK-like)`` pair
        where FK-like means a column ending in ``_id`` and PK-like means
        ``id`` or ``<dst_table>_id``.
    3.  If no compatible pair exists, return ``None`` (the candidate is
        dropped).

    Type compatibility is enforced via the heuristic scorer's
    ``types_compatible`` so high-cosine pairs whose join columns disagree
    on type family (e.g. ``customer_id INT`` ↔ ``customer_id VARCHAR``)
    never reach the graph.
    """
    src_fk_cols: list[Column] = [
        c for c in src.columns if c.name.lower().endswith("_id")
    ]
    dst_lower = dst.name.lower()
    dst_pk_cols: list[Column] = [
        c
        for c in dst.columns
        if c.name.lower() == "id" or c.name.lower() == f"{dst_lower}_id"
    ]
    if not src_fk_cols or not dst_pk_cols:
        return None

    # Preferred shape: src.<dst_table>_id → dst.id.
    pref_src = next(
        (c for c in src_fk_cols if c.name.lower() == f"{dst_lower}_id"), None
    )
    pref_dst = next((c for c in dst_pk_cols if c.name.lower() == "id"), None)
    if (
        pref_src is not None
        and pref_dst is not None
        and types_compatible(pref_src, pref_dst)
    ):
        return pref_src.name, pref_dst.name

    # Fallback: any type-compat pair.
    for sc in src_fk_cols:
        for dc in dst_pk_cols:
            if types_compatible(sc, dc):
                return sc.name, dc.name
    return None


def extend_with_embedding_scorer(
    base: ScorerRegistry,
    *,
    store: KuzuStore,
    database_key: str,
    threshold: float | None,
) -> ScorerRegistry:
    """Return a new ``ScorerRegistry`` containing ``base``'s scorers in order
    plus an ``EmbeddingRelationshipScorer`` when ``threshold is not None``.

    When ``threshold is None``, returns ``base`` unchanged (null-path
    invariant).  ``PRETENSOR_EMBEDDINGS_DISABLED=1`` does the same even
    when a threshold is set, so the kill switch forces the null path here
    too.  Otherwise builds a fresh registry, copies every base scorer in
    registration order, and appends the embedding scorer last so the
    combiner sees heuristic / statistical candidates before the embedding
    pass.
    """
    if threshold is None or embeddings_disabled_via_env():
        return base
    out = ScorerRegistry()
    for scorer in base:
        out.register(scorer)
    out.register(
        EmbeddingRelationshipScorer(
            store=store,
            database_key=database_key,
            threshold=threshold,
        )
    )
    return out
