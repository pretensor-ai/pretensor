"""Unit tests for the four pure L1 metric functions."""

from __future__ import annotations

import math

import pytest

from pretensor.benchmark.l1.metrics import (
    cluster_stability_jaccard,
    inferred_join_pr,
    role_f1,
)

# --- inferred_join_pr ----------------------------------------------------


def test_inferred_join_pr_perfect():
    keys = [
        ("public.orders", "customer_id", "public.customer", "id"),
        ("public.payment", "rental_id", "public.rental", "id"),
    ]
    p, r = inferred_join_pr(keys, keys)
    assert p == 1.0
    assert r == 1.0


def test_inferred_join_pr_zero_overlap():
    inferred = [("a", "x", "b", "y")]
    declared = [("c", "u", "d", "v")]
    p, r = inferred_join_pr(inferred, declared)
    assert p == 0.0
    assert r == 0.0


def test_inferred_join_pr_partial_overlap():
    inferred = [
        ("a", "x", "b", "y"),  # match
        ("a", "z", "c", "w"),  # not in declared
    ]
    declared = [
        ("a", "x", "b", "y"),  # match
        ("d", "u", "e", "v"),  # not in inferred
    ]
    p, r = inferred_join_pr(inferred, declared)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)


def test_inferred_join_pr_treats_reverse_orientation_as_match():
    declared = [("a", "x", "b", "y")]
    reversed_pred = [("b", "y", "a", "x")]
    p, r = inferred_join_pr(reversed_pred, declared)
    assert p == 1.0
    assert r == 1.0


def test_inferred_join_pr_empty_ground_truth():
    p, r = inferred_join_pr([("a", "x", "b", "y")], [])
    assert p == 0.0
    assert r == 0.0


def test_inferred_join_pr_empty_prediction():
    p, r = inferred_join_pr([], [("a", "x", "b", "y")])
    assert p == 0.0
    assert r == 0.0


def test_inferred_join_pr_both_empty():
    p, r = inferred_join_pr([], [])
    assert p == 0.0
    assert r == 0.0


# --- cluster_stability_jaccard -------------------------------------------


def test_cluster_jaccard_identical():
    a = [["t1", "t2"], ["t3"]]
    assert cluster_stability_jaccard(a, a) == 1.0


def test_cluster_jaccard_disjoint():
    a = [["t1", "t2"]]
    b = [["t3", "t4"]]
    assert cluster_stability_jaccard(a, b) == 0.0


def test_cluster_jaccard_partial():
    # cluster_a's best match in cluster_b is {t1,t2,t3} ∩ {t1,t2} / {t1,t2,t3} = 2/3
    a = [["t1", "t2", "t3"]]
    b = [["t1", "t2"]]
    assert cluster_stability_jaccard(a, b) == pytest.approx(2 / 3)


def test_cluster_jaccard_both_empty_is_one():
    assert cluster_stability_jaccard([], []) == 1.0


def test_cluster_jaccard_one_side_empty_is_zero():
    assert cluster_stability_jaccard([["t1"]], []) == 0.0
    assert cluster_stability_jaccard([], [["t1"]]) == 0.0


def test_cluster_jaccard_filters_empty_clusters():
    # Empty member lists should be ignored (treat as no cluster).
    a = [["t1"], []]
    b = [["t1"]]
    assert cluster_stability_jaccard(a, b) == 1.0


def test_cluster_jaccard_handles_frozenset_input():
    a = [frozenset({"t1", "t2"})]
    b = [frozenset({"t1", "t2"})]
    assert cluster_stability_jaccard(a, b) == 1.0


# --- role_f1 -------------------------------------------------------------


def test_role_f1_perfect():
    gold = {"t1": "fact", "t2": "dimension"}
    pred = dict(gold)
    assert role_f1(pred, gold) == 1.0


def test_role_f1_zero():
    gold = {"t1": "fact"}
    pred = {"t1": "dimension"}
    assert role_f1(pred, gold) == 0.0


def test_role_f1_mixed():
    gold = {"t1": "fact", "t2": "fact", "t3": "dimension"}
    # Predicts t1 and t3 correctly, mislabels t2 as dimension
    pred = {"t1": "fact", "t2": "dimension", "t3": "dimension"}
    score = role_f1(pred, gold)
    # Class fact: P=1/1=1.0, R=1/2=0.5 → F1 = 2/3
    # Class dimension: P=1/2=0.5, R=1/1=1.0 → F1 = 2/3
    assert score == pytest.approx(2 / 3)


def test_role_f1_both_empty():
    assert role_f1({}, {}) == 1.0


def test_role_f1_predict_empty():
    assert role_f1({}, {"t1": "fact"}) == 0.0


def test_role_f1_gold_empty():
    assert role_f1({"t1": "fact"}, {}) == 0.0


def test_role_f1_extra_predictions_ignored():
    gold = {"t1": "fact"}
    pred = {"t1": "fact", "t2": "junction", "t3": "audit"}
    # Class fact: P=1/1=1.0, R=1/1=1.0 → F1=1.0; only one class in gold.
    assert role_f1(pred, gold) == 1.0


def test_role_f1_returns_float_in_unit_interval():
    gold = {"a": "fact", "b": "fact", "c": "dimension"}
    pred = {"a": "fact", "b": "dimension", "c": "fact"}
    score = role_f1(pred, gold)
    assert 0.0 <= score <= 1.0
    assert math.isfinite(score)
