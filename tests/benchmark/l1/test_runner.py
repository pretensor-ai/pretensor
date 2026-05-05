"""End-to-end tests for ``run_l1`` against committed fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from pretensor.benchmark.l1 import run_l1
from pretensor.benchmark.results import read_json
from pretensor.benchmark.runner import Dataset
from tests.benchmark._compare import normalize_baseline_bytes

_BASELINES_DIR = Path(__file__).resolve().parents[1] / "baselines"
_ALL_DATASETS = [
    Dataset.PAGILA,
    Dataset.TPCH,
    Dataset.ADVENTUREWORKS,
    Dataset.ADVERSARIAL,
    Dataset.ANALYTICS_DWH,
    Dataset.SAAS_MULTITENANT,
]
_REQUIRED_METRIC_KEYS = {
    "inferred_join_precision",
    "inferred_join_recall",
    "cluster_stability_jaccard",
    "role_f1",
}


@pytest.fixture
def graph_dir(tmp_path: Path) -> Path:
    """L1 doesn't read from this dir but the runner contract requires one."""
    return tmp_path / ".pretensor"


def test_run_l1_emits_required_metrics(tmp_path: Path, graph_dir: Path):
    out = tmp_path / "out.json"
    run_l1(Dataset.PAGILA, out, graph_dir, embeddings=False)
    result = read_json(out)
    assert result.level == "l1"
    assert result.dataset == "pagila"
    assert set(result.metrics.keys()) == _REQUIRED_METRIC_KEYS
    for name, metric in result.metrics.items():
        assert metric.direction == "higher_is_better", name


def test_run_l1_is_byte_identical_across_runs(tmp_path: Path, graph_dir: Path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    run_l1(Dataset.PAGILA, a, graph_dir, embeddings=False)
    run_l1(Dataset.PAGILA, b, graph_dir, embeddings=False)
    assert a.read_bytes() == b.read_bytes()


def test_run_l1_role_f1_only_for_adversarial(tmp_path: Path, graph_dir: Path):
    out = tmp_path / "out.json"
    run_l1(Dataset.ADVERSARIAL, out, graph_dir, embeddings=False)
    adv = read_json(out)
    assert adv.metrics["role_f1"].value is not None

    run_l1(Dataset.PAGILA, out, graph_dir, embeddings=False)
    pag = read_json(out)
    assert pag.metrics["role_f1"].value is None
    assert any("role_f1" in note for note in pag.notes)


def test_run_l1_embeddings_flag_without_extra_emits_install_hint(
    tmp_path: Path, graph_dir: Path, monkeypatch
):
    # Force the embeddings sentinel to fail to import even when the
    # `[embeddings]` extra is present in the test environment.
    import pretensor.benchmark.l1.runner as runner_mod

    monkeypatch.setattr(runner_mod, "_EMBEDDINGS_SENTINEL", "definitely_not_a_module")
    out = tmp_path / "out.json"
    run_l1(Dataset.PAGILA, out, graph_dir, embeddings=True)
    result = read_json(out)
    assert result.embeddings_enabled is False
    assert any("embeddings" in note.lower() for note in result.notes)


@pytest.mark.parametrize("dataset", _ALL_DATASETS, ids=lambda d: d.value)
def test_run_l1_matches_committed_baseline(
    tmp_path: Path, graph_dir: Path, dataset: Dataset
):
    """Re-running L1 must reproduce the committed baseline byte-for-byte.

    Skips gracefully if the baseline file is absent — useful while
    developing the runner before the first baseline-commit pass.
    """
    baseline = _BASELINES_DIR / f"{dataset.value}-l1.json"
    if not baseline.exists():
        pytest.skip(f"baseline {baseline.name} not committed yet")

    out = tmp_path / f"{dataset.value}.json"
    run_l1(dataset, out, graph_dir, embeddings=False)
    assert normalize_baseline_bytes(out.read_bytes()) == normalize_baseline_bytes(
        baseline.read_bytes()
    ), (
        f"Re-run for {dataset.value} drifted from committed baseline; "
        f"regenerate with `uv run pretensor benchmark l1 --dataset "
        f"{dataset.value} --out {baseline}`."
    )
