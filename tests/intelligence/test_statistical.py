"""Unit tests for intelligence/statistical.py — blending formula and edge cases."""

from __future__ import annotations

from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.statistical import apply_statistical_adjustment

_OVERLAP_WEIGHT = 0.65
_PRIOR_WEIGHT = 0.35


def _cand(confidence: float = 0.6, source: str = "heuristic") -> RelationshipCandidate:
    return RelationshipCandidate(
        candidate_id="test:a->b",
        source_node_id="a",
        target_node_id="b",
        source_column="x",
        target_column="y",
        source=source,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning="original",
    )


def test_blend_formula() -> None:
    prior = 0.6
    overlap = 0.8
    result = apply_statistical_adjustment(_cand(prior), overlap)
    expected = _PRIOR_WEIGHT * prior + _OVERLAP_WEIGHT * overlap
    assert abs(result.confidence - expected) < 1e-9


def test_overlap_none_returns_unchanged() -> None:
    cand = _cand(0.7)
    result = apply_statistical_adjustment(cand, None)
    assert result is cand


def test_overlap_zero_pulls_confidence_down() -> None:
    result = apply_statistical_adjustment(_cand(0.9), 0.0)
    expected = _PRIOR_WEIGHT * 0.9 + _OVERLAP_WEIGHT * 0.0
    assert abs(result.confidence - expected) < 1e-9
    assert result.confidence < 0.9


def test_overlap_one_pulls_confidence_up() -> None:
    result = apply_statistical_adjustment(_cand(0.2), 1.0)
    expected = _PRIOR_WEIGHT * 0.2 + _OVERLAP_WEIGHT * 1.0
    assert abs(result.confidence - expected) < 1e-9
    assert result.confidence > 0.2


def test_blended_clamped_above_one() -> None:
    # Overlap slightly above 1.0 in float arithmetic — must clamp
    result = apply_statistical_adjustment(_cand(1.0), 1.0)
    assert result.confidence <= 1.0


def test_blended_clamped_below_zero() -> None:
    # Overlap < 0 is clipped to 0 first, then blend
    result = apply_statistical_adjustment(_cand(0.0), -0.5)
    assert result.confidence >= 0.0


def test_source_always_becomes_statistical() -> None:
    for src in ("heuristic", "llm_inferred", "explicit_fk"):
        result = apply_statistical_adjustment(_cand(source=src), 0.5)  # type: ignore[arg-type]
        assert result.source == "statistical"


def test_reasoning_includes_overlap_value() -> None:
    result = apply_statistical_adjustment(_cand(0.5), 0.75)
    assert "0.750" in result.reasoning


def test_prior_source_preserved_in_reasoning() -> None:
    result = apply_statistical_adjustment(_cand(source="heuristic"), 0.5)
    assert "heuristic" in result.reasoning
