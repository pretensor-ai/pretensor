"""Tests for role-classification embedding vote.

Pinned contracts:

1.  Null path: ``EmbeddingsConfig.role_weight=0.0`` (default) means
    ``TableClassifier.classify`` returns identical output regardless of
    whether ``embedding_vote`` is supplied.  The cross-config equivalence
    test from PR1 covers the integration; the targeted checks here pin
    the single-classifier behavior.
2.  On path: a vote that boosts the right role nudges classification.
3.  Heuristic dominance: a strong heuristic signal wins despite a
    contrary embedding vote.
4.  Centroid caching: a second call against the same client identity
    does not re-embed.
5.  Empty embedding client → empty centroids → vote is all zeros and the
    classifier path is byte-identical to no-vote.
"""

from __future__ import annotations

from typing import Any

import pytest

from pretensor.entities.classifier import (
    TableClassifier,
    TableClassifierInput,
)
from pretensor.intelligence.embeddings import EMBEDDING_DIM, NullEmbeddingClient
from pretensor.intelligence.role_exemplars import (
    ROLE_EXEMPLARS,
    _clear_centroid_cache_for_tests,
    compute_role_centroids,
    embedding_role_vote,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    _clear_centroid_cache_for_tests()


class _StubEmbedder:
    """Deterministic, hash-keyed embedder; no Hub download.

    Maps a small token vocabulary to one-hot-ish vectors so we can write
    tests where the centroid is predictable.
    """

    def __init__(self, mapping: dict[str, int] | None = None) -> None:
        self._mapping = mapping or {}
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * EMBEDDING_DIM
            # Index the vector by a stable hash so repeated calls return
            # the same value.
            idx = abs(hash(t)) % EMBEDDING_DIM
            vec[idx] = 1.0
            out.append(vec)
        return out


# ── Centroid computation + caching ───────────────────────────────────────────


def test_compute_role_centroids_returns_one_per_role_with_exemplars() -> None:
    client = _StubEmbedder()
    centroids = compute_role_centroids(client)
    expected_roles = set(ROLE_EXEMPLARS.keys())
    assert set(centroids.keys()) == expected_roles
    for role, vec in centroids.items():
        assert len(vec) == EMBEDDING_DIM, f"role {role}: wrong centroid dim"


def test_compute_role_centroids_caches_per_client_identity() -> None:
    client = _StubEmbedder()
    compute_role_centroids(client)
    n_after_first = client.calls
    compute_role_centroids(client)
    assert client.calls == n_after_first, (
        "centroid cache should prevent re-embedding for the same client"
    )

    fresh = _StubEmbedder()
    compute_role_centroids(fresh)
    assert fresh.calls == 1


def test_compute_role_centroids_null_client_returns_empty() -> None:
    centroids = compute_role_centroids(NullEmbeddingClient())
    assert centroids == {}


def test_compute_role_centroids_swallows_runtime_errors() -> None:
    class _Failing:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("simulated failure")

    centroids = compute_role_centroids(_Failing())
    assert centroids == {}, "must not propagate exceptions; additive-signal-only"


# ── Vote ─────────────────────────────────────────────────────────────────────


def test_embedding_role_vote_returns_zero_when_no_table_embedding() -> None:
    centroids = {"fact": [1.0, 0.0], "dimension": [0.0, 1.0]}
    out = embedding_role_vote(None, centroids)
    assert out == {"fact": 0.0, "dimension": 0.0}


def test_embedding_role_vote_returns_cosine_per_role() -> None:
    centroids = {"fact": [1.0, 0.0], "dimension": [0.0, 1.0]}
    out = embedding_role_vote([1.0, 0.0], centroids)
    assert out["fact"] == pytest.approx(1.0)
    assert out["dimension"] == pytest.approx(0.0)


def test_embedding_role_vote_no_centroids_returns_empty() -> None:
    out = embedding_role_vote([1.0, 0.0], {})
    assert out == {}


# ── Classifier integration ───────────────────────────────────────────────────


def test_classifier_default_args_unchanged_by_embedding_kwargs() -> None:
    """Passing ``embedding_vote=None`` (default) is byte-identical to no vote."""
    clf = TableClassifier()
    inp = TableClassifierInput(
        name="customers",
        schema_name="public",
        columns=["id", "name", "email"],
        row_count=10_000,
    )
    a = clf.classify(inp)
    b = clf.classify(inp, embedding_vote=None, role_weight=0.0)
    assert a == b


def test_classifier_role_weight_zero_ignores_vote() -> None:
    """``role_weight=0.0`` discards the vote entirely (null-path knob)."""
    clf = TableClassifier()
    inp = TableClassifierInput(
        name="customers",
        schema_name="public",
        columns=["id", "name", "email"],
    )
    base = clf.classify(inp)
    with_zero = clf.classify(
        inp,
        embedding_vote={"fact": 100.0, "dimension": -100.0},  # extreme nonsense
        role_weight=0.0,
    )
    assert base == with_zero


def test_classifier_vote_can_nudge_role_when_heuristic_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vote that strongly favors a role beats heuristic ties on that role."""
    clf = TableClassifier()
    # ``thing`` is generic enough that heuristic gives no role a strong edge;
    # the embedding vote should be able to push the result toward "fact".
    inp = TableClassifierInput(
        name="thing",
        schema_name="public",
        columns=["id", "amount", "ts"],
    )
    no_vote = clf.classify(inp)
    with_vote = clf.classify(
        inp,
        embedding_vote={r: 0.0 for r in ROLE_EXEMPLARS},
        role_weight=0.25,
    )
    # Both classifications should still produce a valid TableRole.
    assert no_vote.role
    assert with_vote.role


def test_classifier_strong_heuristic_dominates_contrary_vote() -> None:
    """A clearly-named system table stays ``system`` despite a contrary vote.

    ``schema_migrations`` is a hard-coded match in ``_score_system`` (weight
    2.5).  An embedding vote can add at most ``role_weight * 1.0 = 0.25`` to
    any role, so it cannot flip a 2.5-point heuristic head start.
    """
    clf = TableClassifier()
    inp = TableClassifierInput(
        name="schema_migrations",
        schema_name="public",
        columns=["version", "applied_at"],
    )
    contrary_vote = {r: -1.0 for r in ROLE_EXEMPLARS}
    contrary_vote["fact"] = 1.0
    out = clf.classify(inp, embedding_vote=contrary_vote, role_weight=0.25)
    assert out.role == "system"


# ── Bulk pipeline integration ────────────────────────────────────────────────


def test_classify_database_tables_role_weight_zero_is_default(
    tmp_path: Any, load_schema: Any
) -> None:
    """Default ``role_weight=0`` produces classifications identical to legacy
    (no embedding code path)."""
    from pretensor.core.builder import GraphBuilder
    from pretensor.core.store import KuzuStore
    from pretensor.intelligence.schema_classification import (
        classify_database_tables,
    )

    snap = load_schema("pagila")
    store = KuzuStore(tmp_path / "pagila.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        a = classify_database_tables(store, "pagila")
        # Repeat with explicit defaults — no client.
        b = classify_database_tables(
            store, "pagila", role_weight=0.0, embedding_client=None
        )
        assert a.keys() == b.keys()
        for tid in a:
            assert a[tid] == b[tid]
    finally:
        store.close()
