"""Pure L2 metric functions.

Each function takes already-collected observations and returns a scalar.
The runner owns all I/O — these helpers stay testable without spinning
up Kuzu, the keyword index, or any MCP machinery.

Boundary cases (empty input, missing gold per item) are handled here
so the runner never divides by zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = [
    "JoinPair",
    "RankedHit",
    "compile_metric_correctness",
    "query_recall_at_k",
    "semantic_search_recall_at_k",
    "top_k_with_ties",
    "traverse_correctness",
]


JoinPair = tuple[str, str]
"""``(from_table, to_table)`` — schema-qualified bare names."""

RankedHit = tuple[float, str]
"""``(score, table_name)`` — one search hit; higher score = better."""


_SCORE_ROUNDING_DECIMALS = 4
"""Bucket precision applied to scores before sorting/cutoff.

Upstream BM25 (SQLite FTS5) has been observed to emit micro-noise — e.g.
``-1e-06`` versus ``-0.0`` — for hits that are functionally tied in
ranking. Rounding to 4 decimals collapses that noise into clean zero
buckets while preserving meaningful score differences (anything ≥ 1e-4).
"""


def top_k_with_ties(
    ranked: Sequence[RankedHit],
    k: int,
) -> list[RankedHit]:
    """Return the top-K hits plus any extra items tied at the K-th score.

    Sorts defensively by ``(-score, name)`` before slicing so the caller
    needn't pre-sort. AC #5: a tied top-K includes ALL tied items so a
    near-cutoff hit is not missed when comparing against gold.

    Scores are rounded to :data:`_SCORE_ROUNDING_DECIMALS` decimals before
    comparison so that floating-point noise (e.g. SQLite FTS5's ``-1e-06``
    vs ``-0.0`` for functionally-tied BM25 hits) doesn't make the cutoff
    flip between runs. The returned tuples carry the rounded score; the
    metric only consumes the table-name field, so this loss of precision
    is invisible downstream.
    """
    if k <= 0 or not ranked:
        return []
    rounded: list[RankedHit] = [
        (round(score, _SCORE_ROUNDING_DECIMALS), name) for score, name in ranked
    ]
    ordered = sorted(rounded, key=lambda h: (-h[0], h[1]))
    if len(ordered) <= k:
        return ordered
    cutoff_score = ordered[k - 1][0]
    return [h for h in ordered if h[0] >= cutoff_score]


def query_recall_at_k(
    observations: Iterable[tuple[Iterable[str], Sequence[RankedHit]]],
    k: int = 5,
) -> float:
    """Mean Recall@K across observations.

    Each observation is ``(gold_tables, ranked_results)``. ``ranked_results``
    may arrive in any order — :func:`top_k_with_ties` re-sorts it. Recall
    per observation is ``|gold ∩ retrieved_top_k| / |gold|``. Observations
    with empty ``gold_tables`` are skipped (recall undefined). Returns
    ``0.0`` when no observation has gold data.
    """
    total = 0.0
    n = 0
    for gold, ranked in observations:
        gold_set = set(gold)
        if not gold_set:
            continue
        top_k = top_k_with_ties(list(ranked), k)
        retrieved_set = {name for _, name in top_k}
        recall = len(gold_set & retrieved_set) / len(gold_set)
        total += recall
        n += 1
    return total / n if n > 0 else 0.0


def semantic_search_recall_at_k(
    observations: Iterable[tuple[Iterable[str], Sequence[RankedHit]]],
    k: int = 5,
) -> float:
    """Mean Recall@K against ``semantic_search`` retrievals.

    Same shape and semantics as :func:`query_recall_at_k` — kept distinct
    so the two retrieval systems can diverge later (different ranking
    semantics, different post-processing) without forcing one to fit the
    other's API.
    """
    return query_recall_at_k(observations, k=k)


def traverse_correctness(
    observations: Iterable[tuple[Sequence[JoinPair], Sequence[Sequence[JoinPair]]]],
) -> float:
    """Fraction of items whose gold path matches at least one returned path.

    The ``traverse`` tool emits all top-ranked paths when ties exist;
    this metric counts the item as correct when ANY returned path
    equals the gold sequence (tuple-wise comparison). Returns ``0.0``
    when ``observations`` is empty.
    """
    obs_list = list(observations)
    if not obs_list:
        return 0.0

    correct = 0
    for gold_path, returned_paths in obs_list:
        gold_t = tuple(tuple(p) for p in gold_path)
        for path in returned_paths:
            if tuple(tuple(p) for p in path) == gold_t:
                correct += 1
                break
    return correct / len(obs_list)


def compile_metric_correctness(observations: Iterable[bool]) -> float:
    """Fraction of metric templates that compiled to valid SQL.

    Each observation is the boolean ``valid`` flag returned by
    ``compile_metric_payload``. Returns ``0.0`` when ``observations`` is
    empty.
    """
    obs_list = list(observations)
    if not obs_list:
        return 0.0
    return sum(1 for ok in obs_list if ok) / len(obs_list)
