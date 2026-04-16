"""FP/FN tracking utility for cross-DB entity resolution benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroundTruthPair:
    """A labelled entity pair from a benchmark fixture."""

    entity_a: str
    entity_b: str
    is_match: bool


@dataclass
class ScoreResult:
    """Precision / recall / F1 for one threshold level."""

    precision: float
    recall: float
    f1: float
    false_positives: list[tuple[str, str]] = field(default_factory=list)
    false_negatives: list[tuple[str, str]] = field(default_factory=list)


class ScoreTracker:
    """Evaluate a scorer's precision / recall against labelled ground truth.

    ``scored`` is a list of ``(entity_a, entity_b, combined_score)`` triples
    produced by whatever scorer is under test. The tracker compares those
    predictions against ``pairs`` at the given ``threshold``.
    """

    def evaluate(
        self,
        pairs: list[GroundTruthPair],
        scored: list[tuple[str, str, float]],
        threshold: float = 0.55,
    ) -> ScoreResult:
        # Build predicted set (above threshold)
        predicted: set[tuple[str, str]] = set()
        for a, b, score in scored:
            if score >= threshold:
                predicted.add(_canonical(a, b))

        # Build ground-truth positive and negative sets
        positives: set[tuple[str, str]] = set()
        negatives: set[tuple[str, str]] = set()
        for p in pairs:
            key = _canonical(p.entity_a, p.entity_b)
            if p.is_match:
                positives.add(key)
            else:
                negatives.add(key)

        tp = predicted & positives
        fp_pairs = [_ordered(k) for k in predicted & negatives]
        fn_pairs = [_ordered(k) for k in positives - predicted]

        precision = len(tp) / len(predicted) if predicted else 1.0
        recall = len(tp) / len(positives) if positives else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return ScoreResult(
            precision=precision,
            recall=recall,
            f1=f1,
            false_positives=fp_pairs,
            false_negatives=fn_pairs,
        )

    def threshold_sweep(
        self,
        pairs: list[GroundTruthPair],
        scored: list[tuple[str, str, float]],
        low: float = 0.4,
        high: float = 0.8,
        step: float = 0.05,
    ) -> list[tuple[float, ScoreResult]]:
        """Return (threshold, ScoreResult) for a range of threshold values."""
        results: list[tuple[float, ScoreResult]] = []
        t = low
        while t <= high + 1e-9:
            results.append((round(t, 10), self.evaluate(pairs, scored, threshold=t)))
            t += step
        return results


def _canonical(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


def _ordered(k: tuple[str, str]) -> tuple[str, str]:
    return k
