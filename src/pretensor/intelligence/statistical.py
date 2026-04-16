"""Adjust relationship confidence using sample overlap scores (0.0–1.0)."""

from __future__ import annotations

from pretensor.graph_models.relationship import (
    RelationshipCandidate,
    RelationshipSource,
)

__all__ = ["apply_statistical_adjustment"]

# Weight overlap more than prior confidence when data supports the join.
_OVERLAP_WEIGHT = 0.65
_PRIOR_WEIGHT = 0.35


def apply_statistical_adjustment(
    candidate: RelationshipCandidate,
    overlap_score: float | None,
) -> RelationshipCandidate:
    """Blend ``overlap_score`` into ``confidence`` and mark ``source`` as statistical.

    Args:
        candidate: Incoming hypothesis (heuristic or LLM).
        overlap_score: Jaccard-like overlap in ``[0, 1]``, or ``None`` to skip.

    Returns:
        Updated candidate; unchanged if ``overlap_score`` is ``None``.
    """
    if overlap_score is None:
        return candidate
    o = max(0.0, min(1.0, float(overlap_score)))
    blended = _PRIOR_WEIGHT * candidate.confidence + _OVERLAP_WEIGHT * o
    blended = max(0.0, min(1.0, blended))
    prior = candidate.source
    note = f"statistical overlap={o:.3f} (prior source={prior})"
    reasoning = f"{note}. {candidate.reasoning}".strip()
    return candidate.model_copy(
        update={
            "confidence": blended,
            "source": _statistical_source(candidate.source),
            "reasoning": reasoning,
        }
    )


def _statistical_source(prior: RelationshipSource) -> RelationshipSource:
    """Preserve lineage in reasoning; edge ``source`` is ``statistical`` per contract."""
    _ = prior
    return "statistical"
