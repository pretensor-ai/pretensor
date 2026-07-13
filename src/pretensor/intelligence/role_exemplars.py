"""Curated exemplar tables per ``TableRole`` for embedding-based role voting.

The classifier in :mod:`pretensor.entities.classifier` uses several signals
(name patterns, column shape, structural degree).  This module adds one
more: a nearest-centroid vote against a small per-role exemplar set.

For each role, we list a few real-ish ``(qualified_name, columns)`` pairs
that are *prototypical* of that role.  At runtime, an
:class:`EmbeddingClient` embeds the exemplar text via
:func:`format_entity_text` and we mean-pool the vectors per role to get
one centroid per role.  A new table's role vote is then
``cosine(table_text_embedding, centroid)``, normalized so the best
matching role gets a score in roughly ``[0.0, 1.0]``.

The exemplar set ships in source so the wheel is self-contained — no
fixture files at runtime.  Embeddings are computed lazily (first call)
and cached on the embedding client via ``_centroid_cache`` so repeated
classification runs reuse the same centroids without re-embedding.

Determinism contract:

* ``EmbeddingsConfig.role_weight=0.0`` (default) skips this signal entirely.
* When > 0, the vote is *added* to existing heuristic scores; it never
  replaces them.  ``role_weight`` should stay well below the typical
  heuristic score magnitudes (~0.5–2.5) so the heuristic remains primary.
* If the embedding client is unavailable (extra not installed, or
  ``embed`` raises), the function returns an all-zeros dict and the
  classifier falls back to heuristic-only scoring.
"""

from __future__ import annotations

import logging
import weakref

from pretensor.intelligence.embeddings import (
    EmbeddingClient,
    cosine_similarity,
    format_entity_text,
)

__all__ = [
    "ROLE_EXEMPLARS",
    "compute_role_centroids",
    "embedding_role_vote",
]

logger = logging.getLogger(__name__)


# Per-role exemplars: (qualified_name, column_names).  A handful per role is
# sufficient — mean-pooling smooths out idiosyncrasies of any single example.
# Names are illustrative; the embeddings encode the schema "shape" via
# format_entity_text, not the brand-name.
ROLE_EXEMPLARS: dict[str, list[tuple[str, list[str]]]] = {
    "fact": [
        ("public.orders", ["id", "customer_id", "order_date", "total_amount"]),
        ("public.transactions", ["id", "account_id", "amount", "ts"]),
        ("sales.invoice_lines", ["id", "invoice_id", "product_id", "qty", "price"]),
        ("public.events", ["id", "user_id", "event_type", "occurred_at"]),
        ("public.payments", ["id", "order_id", "amount", "paid_at", "method"]),
    ],
    "dimension": [
        ("public.customers", ["id", "first_name", "last_name", "email"]),
        ("public.products", ["id", "name", "category", "price"]),
        ("public.users", ["id", "username", "email", "created_at"]),
        ("public.locations", ["id", "country", "region", "city"]),
        ("public.suppliers", ["id", "name", "address", "phone"]),
    ],
    "bridge": [
        ("public.user_roles", ["user_id", "role_id"]),
        ("public.product_categories", ["product_id", "category_id"]),
        ("public.order_tags", ["order_id", "tag_id"]),
        ("public.film_actor", ["film_id", "actor_id"]),
    ],
    "junction": [
        ("public.user_groups", ["user_id", "group_id"]),
        ("public.role_permissions", ["role_id", "permission_id"]),
    ],
    "staging": [
        ("staging.raw_orders", ["id", "raw_payload", "ingested_at"]),
        ("staging.tmp_users", ["id", "data", "loaded_at"]),
        ("public.import_buffer", ["id", "source", "raw_json", "imported_at"]),
    ],
    "audit": [
        ("public.audit_log", ["id", "actor", "action", "target", "ts"]),
        ("public.access_log", ["id", "user_id", "endpoint", "ip", "ts"]),
        ("public.history", ["id", "entity_id", "old_value", "new_value", "ts"]),
    ],
    "snapshot_scd": [
        ("public.customer_history", ["id", "customer_id", "valid_from", "valid_to"]),
        ("public.product_snapshot", ["id", "product_id", "snapshot_date", "price"]),
        ("public.account_scd2", ["id", "account_id", "is_current", "effective_at"]),
    ],
    "aggregate": [
        ("analytics.daily_revenue", ["day", "total", "order_count"]),
        ("analytics.monthly_active_users", ["month", "mau", "growth_rate"]),
        ("public.summary_stats", ["metric", "value", "computed_at"]),
    ],
    "system": [
        ("public.schema_migrations", ["version", "applied_at"]),
        ("public.ar_internal_metadata", ["key", "value", "updated_at"]),
    ],
    "entity_candidate": [
        ("public.thing", ["id", "name"]),
        ("public.unknown_table", ["id", "data"]),
    ],
    # 'unknown' has no exemplars by design — it's the fallback when nothing
    # else fires; voting against it would defeat the purpose.
}


# Cache: maps client instance → {role → centroid}.  Keyed on the actual
# object via WeakKeyDictionary so a freed client's entry vanishes
# automatically — guards against CPython recycling a deleted client's
# ``id()`` for a fresh, unrelated allocation, which would let a stale
# centroid leak into the new client's lookups in long-running processes.
_CENTROID_CACHE: "weakref.WeakKeyDictionary[EmbeddingClient, dict[str, list[float]]]" = weakref.WeakKeyDictionary()
# Bounded fallback for client types that aren't weakref-able (e.g. mocks
# built from raw ``object()``).
#
# CAVEAT: this fallback path keys on ``id(client)`` directly and therefore
# does NOT have the recycling protection of the WeakKeyDictionary above.
# If a non-weakref-able client A is freed and its id is recycled by a
# fresh, unrelated client B before A's fallback entry is evicted, B's
# lookup would return A's stale centroid. The hazard is bounded in
# practice because (a) only test mocks land here — production
# ``LocalEmbeddingClient`` is weakref-able, (b) the FIFO ceiling
# below caps the window in which a stale entry can survive, and (c) tests
# clear the cache between runs via ``_clear_centroid_cache_for_tests``.
# Treat the fallback as best-effort for tests, not a load-bearing
# correctness guarantee.
_CENTROID_CACHE_FALLBACK: dict[int, dict[str, list[float]]] = {}
# Maximum entries retained in the non-weakref-able fallback cache. The
# typical caller is one ``LocalEmbeddingClient`` per process; the only
# things landing here are short-lived test mocks. A small ceiling
# (insertion-order eviction, FIFO) is plenty to keep the dict bounded
# without losing the cache's value on the realistic call patterns.
_CENTROID_CACHE_FALLBACK_MAX = 32


def _cache_get(client: EmbeddingClient) -> dict[str, list[float]] | None:
    try:
        return _CENTROID_CACHE.get(client)
    except TypeError:
        return _CENTROID_CACHE_FALLBACK.get(id(client))


def _cache_set(client: EmbeddingClient, value: dict[str, list[float]]) -> None:
    try:
        _CENTROID_CACHE[client] = value
    except TypeError:
        # FIFO evict to keep the dict bounded.
        while len(_CENTROID_CACHE_FALLBACK) >= _CENTROID_CACHE_FALLBACK_MAX:
            oldest_key = next(iter(_CENTROID_CACHE_FALLBACK))
            del _CENTROID_CACHE_FALLBACK[oldest_key]
        _CENTROID_CACHE_FALLBACK[id(client)] = value


def compute_role_centroids(
    client: EmbeddingClient,
) -> dict[str, list[float]]:
    """Return one centroid vector per role with at least one exemplar.

    Embeds every exemplar via :func:`format_entity_text` and mean-pools
    per role.  Cached on the client (weak-ref keyed) so the second call
    is free without leaking memory if the client is later garbage-collected.
    Returns an empty dict (no centroids) when the client raises any
    exception while embedding — caller falls back to heuristic-only.
    """
    cached = _cache_get(client)
    if cached is not None:
        return cached

    centroids: dict[str, list[float]] = {}
    try:
        # Flatten the exemplar set into one batch for one embed() call.
        flat_texts: list[str] = []
        flat_owners: list[str] = []
        for role, exemplars in ROLE_EXEMPLARS.items():
            for qname, cols in exemplars:
                flat_texts.append(format_entity_text(qname, cols))
                flat_owners.append(role)
        if not flat_texts:
            _cache_set(client, centroids)
            return centroids

        vectors = client.embed(flat_texts)
        if not vectors or len(vectors) != len(flat_texts):
            # NullEmbeddingClient returns []; treat as no centroids.
            _cache_set(client, centroids)
            return centroids

        per_role: dict[str, list[list[float]]] = {}
        for role, vec in zip(flat_owners, vectors, strict=True):
            per_role.setdefault(role, []).append(vec)

        for role, vecs in per_role.items():
            if not vecs:
                continue
            dim = len(vecs[0])
            mean = [0.0] * dim
            count = 0
            for v in vecs:
                if len(v) != dim:
                    continue
                for i in range(dim):
                    mean[i] += float(v[i])
                count += 1
            if count == 0:
                continue
            centroids[role] = [x / count for x in mean]
    except ImportError as exc:
        logger.warning(
            "role_exemplars: [embeddings] extra not installed (%s); "
            "skipping role centroids",
            exc,
        )
        centroids = {}
    except Exception as exc:  # noqa: BLE001 — additive signal, never raise
        logger.warning(
            "role_exemplars: failed to compute centroids (%s); "
            "falling back to heuristic-only role vote",
            exc,
        )
        centroids = {}

    _cache_set(client, centroids)
    return centroids


def embedding_role_vote(
    table_embedding: list[float] | None,
    centroids: dict[str, list[float]],
) -> dict[str, float]:
    """Cosine vote for the table's embedding against per-role centroids.

    ``table_embedding`` is typically pre-computed at index time via
    ``EmbeddingIndexStep`` and read back from the store; pass ``None``
    when the table doesn't carry one and this function returns an
    all-zeros vote.

    Returns ``{role: cosine}`` for every role with a centroid; roles
    without a centroid are not present in the output.  Cosine values are
    in ``[-1, 1]``; the classifier scales by ``role_weight`` before
    blending.
    """
    if table_embedding is None or not centroids:
        return {role: 0.0 for role in centroids}

    out: dict[str, float] = {}
    for role, centroid in centroids.items():
        try:
            out[role] = cosine_similarity(table_embedding, centroid)
        except ValueError:
            out[role] = 0.0
    return out


def _clear_centroid_cache_for_tests() -> None:
    """Test helper: drop the module-level centroid cache."""
    _CENTROID_CACHE.clear()
    _CENTROID_CACHE_FALLBACK.clear()
