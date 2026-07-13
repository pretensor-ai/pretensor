"""End-to-end tests for ``run_l2`` against committed fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from pretensor.benchmark.l2 import run_l2
from pretensor.benchmark.results import read_json
from pretensor.benchmark.runner import Dataset
from tests.benchmark._compare import normalize_baseline_bytes

_BASELINES_DIR = Path(__file__).resolve().parents[1] / "baselines"
_DATASETS_WITH_GOLD = [
    Dataset.PAGILA,
    Dataset.TPCH,
    Dataset.ADVENTUREWORKS,
]
_REQUIRED_METRIC_KEYS = {
    "query_recall_at_5",
    "traverse_correctness",
    "compile_metric_correctness",
}


@pytest.fixture
def graph_dir(tmp_path: Path) -> Path:
    """L2 builds a fresh in-process graph; this dir is only for the CLI contract."""
    return tmp_path / ".pretensor"


def test_run_l2_emits_required_metrics(tmp_path: Path, graph_dir: Path) -> None:
    out = tmp_path / "out.json"
    run_l2(Dataset.PAGILA, out, graph_dir, embeddings=False)
    result = read_json(out)
    assert result.level == "l2"
    assert result.dataset == "pagila"
    # The three core metrics MUST always be present (AC #2).
    assert _REQUIRED_METRIC_KEYS <= set(result.metrics.keys())
    for name, metric in result.metrics.items():
        assert metric.direction == "higher_is_better", name


def test_run_l2_is_byte_identical_across_runs(tmp_path: Path, graph_dir: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    run_l2(Dataset.PAGILA, a, graph_dir, embeddings=False)
    run_l2(Dataset.PAGILA, b, graph_dir, embeddings=False)
    assert a.read_bytes() == b.read_bytes()


def test_run_l2_omits_semantic_search_without_embeddings(
    tmp_path: Path, graph_dir: Path
) -> None:
    """``--no-embeddings`` MUST omit the semantic_search metric key entirely."""
    out = tmp_path / "out.json"
    run_l2(Dataset.PAGILA, out, graph_dir, embeddings=False)
    result = read_json(out)
    assert "semantic_search_recall_at_5" not in result.metrics


def test_run_l2_emits_skip_note_when_semantic_search_unavailable(
    tmp_path: Path, graph_dir: Path
) -> None:
    """When the semantic_search tool is missing, the runner notes the skip clearly."""
    out = tmp_path / "out.json"
    run_l2(Dataset.PAGILA, out, graph_dir, embeddings=False)
    result = read_json(out)
    assert any("semantic_search" in note.lower() for note in result.notes), (
        "expected a note explaining the semantic_search skip"
    )


def test_run_l2_embeddings_flag_without_extra_emits_install_hint(
    tmp_path: Path, graph_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force the embeddings sentinel to fail to import even when the
    # `[embeddings]` extra is present in the test environment.
    import pretensor.benchmark.l2.runner as runner_mod

    monkeypatch.setattr(runner_mod, "_EMBEDDINGS_SENTINEL", "definitely_not_a_module")
    out = tmp_path / "out.json"
    run_l2(Dataset.PAGILA, out, graph_dir, embeddings=True)
    result = read_json(out)
    assert result.embeddings_enabled is False
    assert any("embeddings" in note.lower() for note in result.notes)


def test_run_l2_per_item_records_are_sorted_deterministically(
    tmp_path: Path, graph_dir: Path
) -> None:
    out = tmp_path / "out.json"
    run_l2(Dataset.PAGILA, out, graph_dir, embeddings=False)
    result = read_json(out)
    # per_item is sorted by (kind, id) so the JSON envelope stays
    # byte-stable across re-runs even when observation collection order
    # shifts.
    keys = [(item.get("kind", ""), item.get("id", "")) for item in result.per_item]
    assert keys == sorted(keys)


@pytest.mark.parametrize("dataset", _DATASETS_WITH_GOLD, ids=lambda d: d.value)
def test_run_l2_matches_committed_baseline(
    tmp_path: Path, graph_dir: Path, dataset: Dataset
) -> None:
    """Re-running L2 must reproduce the committed baseline byte-for-byte.

    Skips gracefully if the baseline file is absent — useful while
    developing the runner before the first baseline-commit pass.
    """
    baseline = _BASELINES_DIR / f"{dataset.value}-l2.json"
    if not baseline.exists():
        pytest.skip(f"baseline {baseline.name} not committed yet")

    out = tmp_path / f"{dataset.value}.json"
    run_l2(dataset, out, graph_dir, embeddings=False)
    assert normalize_baseline_bytes(out.read_bytes()) == normalize_baseline_bytes(
        baseline.read_bytes()
    ), (
        f"Re-run for {dataset.value} drifted from committed baseline; "
        f"regenerate with `uv run pretensor benchmark l2 --dataset "
        f"{dataset.value} --out {baseline}`."
    )


def test_run_l2_embeddings_fails_loudly_when_no_vectors_computed(
    tmp_path: Path, graph_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed model fetch must abort the embeddings lane, not silently
    benchmark the no-vectors fallback against the embeddings baseline."""
    from pretensor.intelligence.embeddings import (
        LocalEmbeddingClient,
        embeddings_extra_installed,
    )

    if not embeddings_extra_installed():
        pytest.skip(
            "requires the [embeddings] extra: without it the runner "
            "resolves --embeddings to off and the guard never engages"
        )

    def _download_failed(
        self: LocalEmbeddingClient, texts: list[str]
    ) -> list[list[float]]:
        raise RuntimeError("simulated HF Hub 429")

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _download_failed)

    with pytest.raises(RuntimeError, match="no table vectors"):
        run_l2(Dataset.PAGILA, tmp_path / "out.json", graph_dir, embeddings=True)
