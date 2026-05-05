"""Pure L1 metric functions.

Each function takes only data — no graph store, no I/O. The runner
collects predictions from the intelligence pipeline, then hands them
to these helpers. Boundary cases (empty ground truth, identical
inputs, etc.) are handled here so the runner never divides by zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = [
    "canonicalise_join_key",
    "cluster_stability_jaccard",
    "inferred_join_pr",
    "role_f1",
]


JoinKey = tuple[str, str, str, str]
"""``(src_table, src_column, dst_table, dst_column)`` — bare names.

Schema is folded into the table name (``schema.table``) so callers can
key on either form consistently.
"""


def inferred_join_pr(
    inferred: Iterable[JoinKey],
    declared_fks: Iterable[JoinKey],
) -> tuple[float, float]:
    """Precision and recall of inferred joins against declared FK ground truth.

    Both inputs are sets of ``(src_table, src_column, dst_table, dst_column)``
    tuples. Keys are canonicalised to unordered pairs before comparison —
    declared FKs are directional, but the heuristic discovery layer may
    emit either orientation for the same physical relationship, and an
    inferred edge ``A.x ↔ B.y`` should match the declared FK ``A.x → B.y``
    regardless of orientation.

    Returns:
        ``(precision, recall)``. Both are ``0.0`` when the corresponding
        denominator is zero (no candidates ⇒ precision 0; no ground truth
        ⇒ recall 0). Identical sets ⇒ ``(1.0, 1.0)``.
    """
    inferred_set = {canonicalise_join_key(k) for k in inferred}
    declared_set = {canonicalise_join_key(k) for k in declared_fks}

    true_positives = inferred_set & declared_set
    precision = len(true_positives) / len(inferred_set) if inferred_set else 0.0
    recall = len(true_positives) / len(declared_set) if declared_set else 0.0
    return precision, recall


def cluster_stability_jaccard(
    run_a: Iterable[Iterable[str]],
    run_b: Iterable[Iterable[str]],
) -> float:
    """Mean pairwise Jaccard between two cluster partitions of the same input.

    Each cluster is a collection of table node IDs. A score of ``1.0``
    means the two partitions are identical (modulo cluster ordering); a
    score of ``0.0`` means no cluster from ``run_a`` matches any cluster
    in ``run_b``.

    For each cluster in ``run_a``, this picks the best-matching cluster
    in ``run_b`` (highest Jaccard) and averages those scores. Empty
    inputs on both sides ⇒ ``1.0`` (vacuously stable). Empty input on
    one side only ⇒ ``0.0``.
    """
    a_sets = [frozenset(c) for c in run_a if c]
    b_sets = [frozenset(c) for c in run_b if c]

    if not a_sets and not b_sets:
        return 1.0
    if not a_sets or not b_sets:
        return 0.0

    total = 0.0
    for cluster_a in a_sets:
        best = 0.0
        for cluster_b in b_sets:
            # Both sides are filtered to non-empty frozensets above, so
            # the union is always non-empty — no zero-divide guard needed.
            jacc = len(cluster_a & cluster_b) / len(cluster_a | cluster_b)
            if jacc > best:
                best = jacc
        total += best
    return total / len(a_sets)


def role_f1(
    predicted: Mapping[str, str],
    gold: Mapping[str, str],
) -> float:
    """Macro-averaged F1 across the gold role labels.

    For each role appearing in ``gold``, computes precision and recall
    against ``predicted`` and averages the per-class F1 scores. Tables
    in ``gold`` that the predictor did not classify count as misses;
    tables in ``predicted`` whose gold role is absent are ignored
    (they're outside the evaluation set).

    Returns ``1.0`` when ``gold`` and ``predicted`` are both empty
    (vacuously correct). Returns ``0.0`` when ``gold`` is non-empty but
    ``predicted`` is empty, and vice versa.
    """
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0

    gold_roles = sorted(set(gold.values()))
    f1_sum = 0.0
    for role in gold_roles:
        gold_for_role = {t for t, r in gold.items() if r == role}
        pred_for_role = {t for t, r in predicted.items() if r == role}
        true_positives = gold_for_role & pred_for_role
        precision = len(true_positives) / len(pred_for_role) if pred_for_role else 0.0
        recall = len(true_positives) / len(gold_for_role) if gold_for_role else 0.0
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_sum += f1
    return f1_sum / len(gold_roles)


def canonicalise_join_key(key: JoinKey) -> tuple[tuple[str, str], tuple[str, str]]:
    """Collapse a directional ``JoinKey`` to an unordered pair.

    Treats ``(A.x → B.y)`` and ``(B.y → A.x)`` as the same edge by
    sorting the two endpoint ``(table, column)`` tuples. Exposed so
    other modules (e.g. the runner's per-item collector) can canonicalise
    keys against the same convention without duplicating the logic.
    """
    src_t, src_c, dst_t, dst_c = key
    a = (src_t, src_c)
    b = (dst_t, dst_c)
    return (a, b) if a <= b else (b, a)
