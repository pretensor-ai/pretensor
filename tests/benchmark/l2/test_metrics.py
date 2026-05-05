"""Tests for the pure L2 metric functions."""

from __future__ import annotations

import pytest

from pretensor.benchmark.l2.metrics import (
    compile_metric_correctness,
    query_recall_at_k,
    semantic_search_recall_at_k,
    top_k_with_ties,
    traverse_correctness,
)

# ---------------------------------------------------------------------------
# top_k_with_ties
# ---------------------------------------------------------------------------


def test_top_k_with_ties_under_k_returns_all() -> None:
    ranked = [(2.0, "a"), (1.0, "b")]
    assert top_k_with_ties(ranked, 5) == [(2.0, "a"), (1.0, "b")]


def test_top_k_with_ties_strict_top_k_when_no_tie() -> None:
    ranked = [(3.0, "a"), (2.0, "b"), (1.0, "c")]
    assert top_k_with_ties(ranked, 2) == [(3.0, "a"), (2.0, "b")]


def test_top_k_with_ties_includes_all_tied_at_cutoff() -> None:
    # AC #5: top-K must include every item tied at the K-th score.
    ranked = [(3.0, "a"), (2.0, "b"), (2.0, "c"), (2.0, "d"), (1.0, "e")]
    out = top_k_with_ties(ranked, 2)
    # K=2 cuts at score 2.0 → b, c, d all qualify.
    assert {name for _, name in out} == {"a", "b", "c", "d"}


def test_top_k_with_ties_sorts_unsorted_input() -> None:
    ranked = [(1.0, "z"), (3.0, "a"), (2.0, "m")]
    out = top_k_with_ties(ranked, 2)
    assert out == [(3.0, "a"), (2.0, "m")]


def test_top_k_with_ties_zero_k_or_empty() -> None:
    assert top_k_with_ties([(1.0, "a")], 0) == []
    assert top_k_with_ties([], 5) == []


def test_top_k_with_ties_breaks_ties_alphabetically() -> None:
    ranked = [(2.0, "z"), (2.0, "a"), (1.0, "b")]
    # Equal scores → name asc keeps the slice deterministic.
    assert top_k_with_ties(ranked, 1) == [(2.0, "a"), (2.0, "z")]


def test_top_k_with_ties_collapses_bm25_micro_noise() -> None:
    # Upstream BM25 emits scores like -1e-06 vs -0.0 for hits that are
    # functionally tied; the docstring promises rounding to 4 decimals
    # collapses both into the same bucket so the cutoff doesn't flip
    # across re-runs. Two inputs that differ only in micro-noise must
    # produce the same {name} set under top-K-with-ties.
    bucket_a = [(2.0, "x"), (-1e-06, "a"), (-1e-06, "b"), (0.0, "c")]
    bucket_b = [(2.0, "x"), (-0.0, "a"), (0.0, "b"), (-1e-06, "c")]
    names_a = {n for _, n in top_k_with_ties(bucket_a, 2)}
    names_b = {n for _, n in top_k_with_ties(bucket_b, 2)}
    assert names_a == names_b == {"x", "a", "b", "c"}


# ---------------------------------------------------------------------------
# query_recall_at_k
# ---------------------------------------------------------------------------


def test_query_recall_at_k_perfect() -> None:
    obs = [
        ({"film", "actor"}, [(2.0, "film"), (1.5, "actor"), (1.0, "rental")]),
    ]
    assert query_recall_at_k(obs, k=5) == pytest.approx(1.0)


def test_query_recall_at_k_partial() -> None:
    obs = [
        # gold has 2; top_3 retrieves 1 of them → recall 0.5
        ({"film", "actor"}, [(2.0, "film"), (1.5, "rental"), (1.0, "store")]),
    ]
    assert query_recall_at_k(obs, k=3) == pytest.approx(0.5)


def test_query_recall_at_k_skips_observations_without_gold() -> None:
    obs = [
        (set(), [(2.0, "film")]),
        ({"film"}, [(2.0, "film")]),
    ]
    # Only the second observation contributes; recall = 1.0.
    assert query_recall_at_k(obs, k=5) == pytest.approx(1.0)


def test_query_recall_at_k_returns_zero_when_no_observations() -> None:
    assert query_recall_at_k([], k=5) == 0.0


def test_query_recall_at_k_means_across_observations() -> None:
    obs = [
        ({"a"}, [(1.0, "a")]),  # 1.0
        ({"b"}, [(1.0, "x"), (0.5, "y")]),  # 0.0
    ]
    # Mean = (1.0 + 0.0) / 2 = 0.5
    assert query_recall_at_k(obs, k=5) == pytest.approx(0.5)


def test_query_recall_at_k_tie_inclusion_can_save_borderline() -> None:
    # k=2 strict would pick only [a, b], missing the gold "c". With ties at
    # cutoff (b, c, d all share score 1.0) the retrieved set covers c.
    obs = [
        ({"c"}, [(2.0, "a"), (1.0, "b"), (1.0, "c"), (1.0, "d")]),
    ]
    assert query_recall_at_k(obs, k=2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# semantic_search_recall_at_k
# ---------------------------------------------------------------------------


def test_semantic_search_recall_at_k_shares_query_semantics() -> None:
    obs = [({"film"}, [(2.0, "film"), (1.0, "rental")])]
    assert semantic_search_recall_at_k(obs, k=5) == query_recall_at_k(obs, k=5)


# ---------------------------------------------------------------------------
# traverse_correctness
# ---------------------------------------------------------------------------


def test_traverse_correctness_exact_match_single_path() -> None:
    gold = [("public.actor", "public.film_actor"), ("public.film_actor", "public.film")]
    returned_paths = [gold]
    assert traverse_correctness([(gold, returned_paths)]) == pytest.approx(1.0)


def test_traverse_correctness_no_match() -> None:
    gold = [("public.a", "public.b")]
    returned_paths = [[("public.a", "public.c")]]
    assert traverse_correctness([(gold, returned_paths)]) == pytest.approx(0.0)


def test_traverse_correctness_pre367_any_returned_matches_counts() -> None:
    # The traverse tool may emit multiple equally-ranked paths when tied.
    # The metric counts the item correct as long as ANY returned path == gold.
    gold = [("public.a", "public.b"), ("public.b", "public.c")]
    returned_paths = [
        [("public.a", "public.x"), ("public.x", "public.c")],  # wrong
        gold,  # match
        [("public.a", "public.y"), ("public.y", "public.c")],  # wrong
    ]
    assert traverse_correctness([(gold, returned_paths)]) == pytest.approx(1.0)


def test_traverse_correctness_fraction_across_items() -> None:
    obs = [
        # match
        (
            [("a", "b")],
            [[("a", "b")]],
        ),
        # miss
        (
            [("c", "d")],
            [[("c", "x")]],
        ),
    ]
    assert traverse_correctness(obs) == pytest.approx(0.5)


def test_traverse_correctness_empty_returns_zero() -> None:
    assert traverse_correctness([]) == 0.0


def test_traverse_correctness_empty_returned_paths_is_miss() -> None:
    obs = [([("a", "b")], [])]
    assert traverse_correctness(obs) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compile_metric_correctness
# ---------------------------------------------------------------------------


def test_compile_metric_correctness_all_valid() -> None:
    assert compile_metric_correctness([True, True, True]) == pytest.approx(1.0)


def test_compile_metric_correctness_all_invalid() -> None:
    assert compile_metric_correctness([False, False]) == pytest.approx(0.0)


def test_compile_metric_correctness_partial() -> None:
    assert compile_metric_correctness([True, False, True, False]) == pytest.approx(0.5)


def test_compile_metric_correctness_empty_returns_zero() -> None:
    assert compile_metric_correctness([]) == 0.0
