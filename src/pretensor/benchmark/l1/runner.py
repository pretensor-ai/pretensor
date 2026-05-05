"""``run_l1`` — collect L1 metrics into a deterministic JSON document."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from pretensor.benchmark.fixtures import load_dataset
from pretensor.benchmark.l1.metrics import (
    canonicalise_join_key,
    cluster_stability_jaccard,
    inferred_join_pr,
    role_f1,
)
from pretensor.benchmark.l1.pipeline import (
    build_l1_artifacts,
    collect_declared_fks,
    discover_inferred_joins_blind,
)
from pretensor.benchmark.results import BenchmarkResult, Metric, write_json
from pretensor.connectors.models import SchemaSnapshot

if TYPE_CHECKING:
    from pretensor.benchmark.runner import Dataset

__all__ = ["run_l1"]


_DETERMINISTIC_RAN_AT = "1970-01-01T00:00:00Z"
"""Pinned timestamp so two consecutive runs produce byte-identical JSON.

Matches the precedent in ``tests/benchmark/test_results.py``. Real
wall-clock would break the determinism gate baked into the spec.
"""

_EMBEDDINGS_SENTINEL = "onnxruntime"
"""Importable module that ships with the ``[embeddings]`` extra.

The runner uses this as a probe: if ``import onnxruntime`` fails the
extra is not installed, and we report the install hint rather than
silently emitting heuristic-only numbers as if embeddings were on.
"""

_FIXTURES_ROOT = Path(__file__).resolve().parents[3].parent / "tests" / "fixtures"
"""``<repo>/tests/fixtures`` resolved relative to the source tree.

Used to locate gold-role companion files (``<dataset>_roles.yaml``).
``parents[3]`` walks up ``runner.py → l1 → benchmark → pretensor`` and
``.parent`` strips ``src/``; the result is the repo root.
"""


def run_l1(
    dataset: Dataset,
    out: Path | None,
    graph_dir: Path,  # noqa: ARG001 — L1 builds a fresh in-process graph
    *,
    embeddings: bool,
) -> None:
    """Run the four OSS L1 metrics for ``dataset`` and emit a ``BenchmarkResult``.

    ``graph_dir`` is part of the CLI contract for L2 / L3 but unused here —
    L1 indexes the fixture into a fresh per-call temporary store so the
    metric is reproducible from a clean checkout. It's accepted (and
    ignored) to keep ``run_l1`` interchangeable with the other runners.
    """
    fixture = load_dataset(dataset)
    snapshot_text = fixture.schema_yaml_path.read_text(encoding="utf-8")
    snapshot = SchemaSnapshot.from_yaml(snapshot_text)
    fixture_sha = (
        "sha256:" + hashlib.sha256(fixture.schema_yaml_path.read_bytes()).hexdigest()
    )

    notes: list[str] = []
    embeddings_enabled = _resolve_embeddings_flag(embeddings, notes)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp = Path(tmp_root)
        artifacts_a = build_l1_artifacts(snapshot, work_dir=tmp / "run-a")
        artifacts_b = build_l1_artifacts(snapshot, work_dir=tmp / "run-b")
        inferred = discover_inferred_joins_blind(snapshot, work_dir=tmp / "blind")

    declared_fks = collect_declared_fks(snapshot)
    p, r = inferred_join_pr(inferred, declared_fks)
    jacc = cluster_stability_jaccard(artifacts_a.clusters, artifacts_b.clusters)

    gold_roles = _load_gold_roles(dataset)
    if gold_roles is None:
        rf: float | None = None
        notes.append(
            f"role_f1 unavailable for dataset '{dataset.value}': "
            f"no gold-role annotation file (only adversarial has labels today)."
        )
    else:
        # Normalise predicted keys (``schema.table``) to bare table names so
        # they line up with the gold-role file's bare-name keys.
        predicted_bare: dict[str, str] = {}
        for key, role in artifacts_a.roles.items():
            bare = key.split(".", 1)[1] if "." in key else key
            predicted_bare[bare] = role
        rf = role_f1(predicted_bare, gold_roles)

    metrics = {
        "inferred_join_precision": Metric(value=p, direction="higher_is_better"),
        "inferred_join_recall": Metric(value=r, direction="higher_is_better"),
        "cluster_stability_jaccard": Metric(value=jacc, direction="higher_is_better"),
        "role_f1": Metric(value=rf, direction="higher_is_better"),
    }

    result = BenchmarkResult(
        level="l1",
        dataset=dataset.value,
        pretensor_version=_resolve_version(),
        embeddings_enabled=embeddings_enabled,
        ran_at=_DETERMINISTIC_RAN_AT,
        fixture_sha=fixture_sha,
        metrics=metrics,
        per_item=_collect_per_item(
            inferred=inferred,
            declared_fks=declared_fks,
            predicted_roles=artifacts_a.roles,
            gold_roles=gold_roles,
        ),
        notes=notes,
    )

    if out is None:
        sys.stdout.write(json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n")
    else:
        write_json(result, out)


def _resolve_embeddings_flag(requested: bool, notes: list[str]) -> bool:
    """Resolve the user's ``--embeddings`` request against the install state.

    Returns the ``embeddings_enabled`` value the JSON envelope should
    record. Appends an install-hint to ``notes`` when the user asked
    for embeddings but the optional extra is not present.
    """
    if not requested:
        return False
    try:
        importlib.import_module(_EMBEDDINGS_SENTINEL)
    except ImportError:
        notes.append(
            "--embeddings was requested but the 'embeddings' extra is not "
            "installed; emitted heuristic-only metrics. Install with "
            "`uv sync --extra embeddings` to enable the embedding path."
        )
        return False
    return True


def _resolve_version() -> str:
    try:
        return importlib.metadata.version("pretensor")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+unknown"


def _load_gold_roles(dataset: Dataset) -> dict[str, str] | None:
    path = _FIXTURES_ROOT / "schemas" / f"{dataset.value}_roles.yaml"
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    roles = raw.get("roles") if "roles" in raw else raw
    if not isinstance(roles, dict):
        return None
    return {str(k): str(v) for k, v in roles.items()}


def _collect_per_item(
    *,
    inferred: list[tuple[str, str, str, str]],
    declared_fks: list[tuple[str, str, str, str]],
    predicted_roles: dict[str, str],
    gold_roles: dict[str, str] | None,
) -> list[dict[str, object]]:
    """Emit a deterministic per-assertion record list for the JSON output."""
    items: list[dict[str, object]] = []

    declared_canon = {canonicalise_join_key(k) for k in declared_fks}
    inferred_canon = {canonicalise_join_key(k) for k in inferred}
    edge_ids = sorted(declared_canon | inferred_canon)
    for edge in edge_ids:
        (src_t, src_c), (dst_t, dst_c) = edge
        items.append(
            {
                "id": f"{src_t}.{src_c}↔{dst_t}.{dst_c}",
                "kind": "inferred_join",
                "expected": edge in declared_canon,
                "predicted": edge in inferred_canon,
            }
        )

    if gold_roles is not None:
        for table_name in sorted(gold_roles):
            # The bare table name lookup matches the format produced by
            # ``build_l1_artifacts`` (``schema.table``) — gold uses bare
            # names without schema, so we resolve via suffix match.
            predicted = _resolve_predicted_role(predicted_roles, table_name)
            items.append(
                {
                    "id": f"role:{table_name}",
                    "kind": "role_classification",
                    "expected": gold_roles[table_name],
                    "predicted": predicted,
                }
            )

    return items


def _resolve_predicted_role(
    predicted: dict[str, str],
    bare_table_name: str,
) -> str | None:
    """Look up a role by bare table name across the predictor's keyed-by-schema dict."""
    if bare_table_name in predicted:
        return predicted[bare_table_name]
    for key, role in predicted.items():
        # Predicted keys are ``schema.table``; gold is bare ``table``.
        if key.endswith(f".{bare_table_name}"):
            return role
    return None
