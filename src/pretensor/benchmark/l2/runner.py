"""``run_l2`` — collect L2 metrics into a deterministic JSON document.

The runner orchestrates the four L2 metrics by:

1. Building a fresh in-memory Kuzu graph + ``registry.json`` from the
   fixture's schema YAML (see :mod:`pretensor.benchmark.l2.pipeline`).
2. Calling the MCP tool payload functions in-process for each gold
   observation — no MCP transport, no subprocess, fully deterministic.
3. Handing already-collected observations to the pure metric helpers
   in :mod:`pretensor.benchmark.l2.metrics`.
4. Wrapping the scalar values in :class:`Metric` and emitting a
   :class:`BenchmarkResult` with byte-identical JSON across re-runs.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pretensor.benchmark.fixtures import load_dataset
from pretensor.benchmark.l2.gold import (
    MetricTemplateEntry,
    QueryGoldEntry,
    TraverseGoldEntry,
    bare_table_name,
    load_metric_templates,
    load_query_gold,
    load_traverse_gold,
)
from pretensor.benchmark.l2.metrics import (
    JoinPair,
    RankedHit,
    compile_metric_correctness,
    query_recall_at_k,
    semantic_search_recall_at_k,
    top_k_with_ties,
    traverse_correctness,
)
from pretensor.benchmark.l2.pipeline import build_l2_graph_dir
from pretensor.benchmark.results import BenchmarkResult, Metric, write_json
from pretensor.connectors.models import SchemaSnapshot
from pretensor.mcp.tools.compile_metric import compile_metric_payload
from pretensor.mcp.tools.search import query_payload
from pretensor.mcp.tools.traverse import traverse_payload

if TYPE_CHECKING:
    from pretensor.benchmark.runner import Dataset

__all__ = ["run_l2"]


_DETERMINISTIC_RAN_AT = "1970-01-01T00:00:00Z"
"""Pinned timestamp so two consecutive runs produce byte-identical JSON."""

_RECALL_K = 5
"""Top-K cutoff used for both ``query`` and ``semantic_search`` Recall@K.

The metric key in the JSON output (``query_recall_at_5`` /
``semantic_search_recall_at_5``) is built from this value.
"""

_EMBEDDINGS_SENTINEL = "onnxruntime"
"""Importable module that ships with the ``[embeddings]`` extra.

Same probe L1 uses; mirrors the ``onnxruntime`` runtime check the
embedding model relies on.
"""


def run_l2(
    dataset: Dataset,
    out: Path | None,
    graph_dir: Path,  # noqa: ARG001 — L2 builds a fresh in-process graph
    *,
    embeddings: bool,
) -> None:
    """Run the four L2 metrics for ``dataset`` and emit a ``BenchmarkResult``.

    ``graph_dir`` is part of the CLI contract for parity with L1 / L3 but
    unused here — L2 indexes the fixture into a fresh per-call temporary
    store so the output is reproducible from a clean checkout.
    """
    fixture = load_dataset(dataset)
    snapshot_text = fixture.schema_yaml_path.read_text(encoding="utf-8")
    snapshot = SchemaSnapshot.from_yaml(snapshot_text)
    fixture_sha = (
        "sha256:" + hashlib.sha256(fixture.schema_yaml_path.read_bytes()).hexdigest()
    )

    notes: list[str] = []
    embeddings_enabled = _resolve_embeddings_flag(embeddings, notes)
    semantic_search_payload = _import_semantic_search_payload()

    query_gold = load_query_gold(fixture)
    traverse_gold = load_traverse_gold(fixture)
    metric_templates = load_metric_templates(fixture)

    metrics: dict[str, Metric] = {}
    per_item: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_dir = Path(tmp_root) / "graph"
        gdir = build_l2_graph_dir(snapshot, work_dir=tmp_dir)

        # query Recall@K --------------------------------------------------
        query_observations = _collect_query_observations(
            graph_dir=gdir,
            database=snapshot.connection_name,
            gold=query_gold,
        )
        per_item.extend(
            _recall_per_item(kind="query_recall", observations=query_observations)
        )
        if any(g.tables_touched for g in query_gold):
            metrics[f"query_recall_at_{_RECALL_K}"] = Metric(
                value=query_recall_at_k(
                    [
                        (g.tables_touched, _bare_hits(hits))
                        for g, hits in query_observations
                    ],
                    k=_RECALL_K,
                ),
                direction="higher_is_better",
            )
        else:
            notes.append(
                f"query_recall_at_{_RECALL_K} skipped for dataset "
                f"'{dataset.value}': no gold question→table mappings present "
                "(fixture missing a *_nl2sql_bench.json with parseable SQL)."
            )

        # semantic_search Recall@K ---------------------------------------
        if semantic_search_payload is None:
            notes.append(
                f"semantic_search_recall_at_{_RECALL_K} skipped: the "
                "semantic_search MCP tool is not yet available in this "
                "build. Re-run once the tool ships."
            )
        elif not embeddings_enabled:
            notes.append(
                f"semantic_search_recall_at_{_RECALL_K} skipped: "
                "--embeddings not requested or the [embeddings] extra is "
                "not installed."
            )
        else:
            sem_observations = _collect_semantic_observations(
                semantic_search_payload=semantic_search_payload,
                graph_dir=gdir,
                database=snapshot.connection_name,
                gold=query_gold,
            )
            per_item.extend(
                _recall_per_item(
                    kind="semantic_search_recall", observations=sem_observations
                )
            )
            if any(g.tables_touched for g in query_gold):
                metrics[f"semantic_search_recall_at_{_RECALL_K}"] = Metric(
                    value=semantic_search_recall_at_k(
                        [
                            (g.tables_touched, _bare_hits(hits))
                            for g, hits in sem_observations
                        ],
                        k=_RECALL_K,
                    ),
                    direction="higher_is_better",
                )

        # traverse correctness -------------------------------------------
        traverse_observations = _collect_traverse_observations(
            graph_dir=gdir,
            database=snapshot.connection_name,
            gold=traverse_gold,
        )
        per_item.extend(_traverse_per_item(traverse_observations))
        if traverse_gold:
            metrics["traverse_correctness"] = Metric(
                value=traverse_correctness(
                    [(g.gold_path, paths) for g, paths in traverse_observations]
                ),
                direction="higher_is_better",
            )
        else:
            notes.append(
                f"traverse_correctness skipped for dataset '{dataset.value}': "
                "no gold_path entries authored in the bench JSON."
            )

        # compile_metric correctness -------------------------------------
        cm_observations = _collect_compile_metric_observations(
            graph_dir=gdir,
            templates=metric_templates,
        )
        per_item.extend(_compile_metric_per_item(metric_templates, cm_observations))
        if metric_templates:
            metrics["compile_metric_correctness"] = Metric(
                value=compile_metric_correctness(
                    [valid for valid, _err in cm_observations]
                ),
                direction="higher_is_better",
            )
        else:
            notes.append(
                f"compile_metric_correctness skipped for dataset "
                f"'{dataset.value}': no metric-templates YAML "
                "(scripts/data/<dataset>_metric_templates.yaml absent)."
            )

    per_item.sort(key=lambda r: (r.get("kind", ""), r.get("id", "")))

    result = BenchmarkResult(
        level="l2",
        dataset=dataset.value,
        pretensor_version=_resolve_version(),
        embeddings_enabled=embeddings_enabled,
        ran_at=_DETERMINISTIC_RAN_AT,
        fixture_sha=fixture_sha,
        metrics=metrics,
        per_item=per_item,
        notes=notes,
    )

    if out is None:
        sys.stdout.write(json.dumps(result.to_dict(), sort_keys=True, indent=2) + "\n")
    else:
        write_json(result, out)


# ---------------------------------------------------------------------------
# observation collectors
# ---------------------------------------------------------------------------


def _collect_query_observations(
    *,
    graph_dir: Path,
    database: str,
    gold: list[QueryGoldEntry],
) -> list[tuple[QueryGoldEntry, list[RankedHit]]]:
    """Run ``query_payload`` for each gold question.

    The MCP tool's ``limit`` is set generously above ``_RECALL_K`` so the
    metric's tie-aware top-K can pick up tied items past the strict cut.
    """
    out: list[tuple[QueryGoldEntry, list[RankedHit]]] = []
    fetch_limit = max(_RECALL_K * 4, 20)
    for entry in gold:
        payload = query_payload(
            graph_dir, q=entry.question, db=database, limit=fetch_limit
        )
        ranked = _ranked_hits_from_query(payload)
        out.append((entry, ranked))
    return out


def _collect_semantic_observations(
    *,
    semantic_search_payload: Any,
    graph_dir: Path,
    database: str,
    gold: list[QueryGoldEntry],
) -> list[tuple[QueryGoldEntry, list[RankedHit]]]:
    """Run ``semantic_search`` for each gold question.

    The exact return shape depends on the tool that ships. We accept
    any callable that returns ``{"results": [...]}`` with ``score`` and
    ``name`` fields per hit — same contract as ``query_payload`` — so
    the forward-compatible mapping below works the moment the tool
    lands.
    """
    out: list[tuple[QueryGoldEntry, list[RankedHit]]] = []
    fetch_limit = max(_RECALL_K * 4, 20)
    for entry in gold:
        payload = semantic_search_payload(
            graph_dir, q=entry.question, db=database, limit=fetch_limit
        )
        ranked = _ranked_hits_from_query(payload)
        out.append((entry, ranked))
    return out


def _collect_traverse_observations(
    *,
    graph_dir: Path,
    database: str,
    gold: list[TraverseGoldEntry],
) -> list[tuple[TraverseGoldEntry, list[list[JoinPair]]]]:
    """Run ``traverse_payload`` for each gold ``(from, to)`` pair.

    Returns the full list of returned paths per item — the traverse
    tool emits all top-ranked paths when tied, and the metric counts
    a match if ANY returned path equals the gold sequence.
    """
    out: list[tuple[TraverseGoldEntry, list[list[JoinPair]]]] = []
    for entry in gold:
        payload = traverse_payload(
            graph_dir,
            from_table=entry.from_table,
            to_table=entry.to_table,
            database=database,
        )
        paths = _path_pairs_from_traverse(payload)
        out.append((entry, paths))
    return out


def _collect_compile_metric_observations(
    *,
    graph_dir: Path,
    templates: list[MetricTemplateEntry],
) -> list[tuple[bool, str]]:
    """Run ``compile_metric_payload`` for each template; return ``(valid, error)``."""
    out: list[tuple[bool, str]] = []
    for tpl in templates:
        payload = compile_metric_payload(
            graph_dir,
            semantic_yaml=tpl.semantic_yaml,
            metric=tpl.metric,
            database=tpl.database,
        )
        if "error" in payload:
            out.append((False, str(payload["error"])))
        else:
            out.append((bool(payload.get("valid", False)), ""))
    return out


# ---------------------------------------------------------------------------
# per-item record builders
# ---------------------------------------------------------------------------


def _recall_per_item(
    *,
    kind: str,
    observations: list[tuple[QueryGoldEntry, list[RankedHit]]],
) -> list[dict[str, Any]]:
    """Build per-question Recall@K records for the JSON envelope.

    Used for both ``query_recall`` and ``semantic_search_recall`` —
    the only thing that varies is the ``kind`` discriminator.
    """
    out: list[dict[str, Any]] = []
    for entry, hits in observations:
        retrieved = top_k_with_ties(hits, _RECALL_K)
        retrieved_bare = sorted({bare_table_name(name) for _score, name in retrieved})
        gold_bare = sorted(entry.tables_touched)
        intersection = sorted(set(gold_bare) & set(retrieved_bare))
        out.append(
            {
                "kind": kind,
                "id": entry.id,
                "gold_tables": gold_bare,
                "retrieved_tables": retrieved_bare,
                "matched": intersection,
                "recall": (len(intersection) / len(gold_bare) if gold_bare else None),
            }
        )
    return out


def _traverse_per_item(
    observations: list[tuple[TraverseGoldEntry, list[list[JoinPair]]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry, paths in observations:
        gold_t = tuple(tuple(p) for p in entry.gold_path)
        match = any(tuple(tuple(p) for p in path) == gold_t for path in paths)
        out.append(
            {
                "kind": "traverse",
                "id": entry.id,
                "from_table": entry.from_table,
                "to_table": entry.to_table,
                "gold_path": [list(hop) for hop in entry.gold_path],
                "returned_paths": [[list(hop) for hop in path] for path in paths],
                "correct": match,
            }
        )
    return out


def _compile_metric_per_item(
    templates: list[MetricTemplateEntry],
    observations: list[tuple[bool, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tpl, (valid, error) in zip(templates, observations, strict=True):
        out.append(
            {
                "kind": "compile_metric",
                "id": tpl.metric,
                "valid": valid,
                "error": error or None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# extractors
# ---------------------------------------------------------------------------


def _ranked_hits_from_query(payload: dict[str, Any]) -> list[RankedHit]:
    """Pull ``[(score, name), ...]`` out of a ``query_payload`` response."""
    hits: list[RankedHit] = []
    if not isinstance(payload, dict):
        return hits
    results = payload.get("results")
    if not isinstance(results, list):
        return hits
    for hit in results:
        if not isinstance(hit, dict):
            continue
        name = hit.get("name")
        score = hit.get("score")
        if isinstance(name, str) and isinstance(score, (int, float)):
            hits.append((float(score), name))
    return hits


def _path_pairs_from_traverse(
    payload: dict[str, Any],
) -> list[list[JoinPair]]:
    """Pull ``[[(from_table, to_table), ...], ...]`` from a traverse response."""
    out: list[list[JoinPair]] = []
    if not isinstance(payload, dict) or "error" in payload:
        return out
    paths = payload.get("paths")
    if not isinstance(paths, list):
        return out
    for path in paths:
        if not isinstance(path, dict):
            continue
        steps = path.get("steps")
        if not isinstance(steps, list):
            continue
        hops: list[JoinPair] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            frm = step.get("from_table")
            to = step.get("to_table")
            if isinstance(frm, str) and isinstance(to, str):
                hops.append((frm, to))
        if hops:
            out.append(hops)
    return out


def _bare_hits(hits: list[RankedHit]) -> list[RankedHit]:
    """Strip the schema prefix from each hit's table name.

    The metric's gold side carries bare table names (the gold loader
    strips schema prefixes when populating ``tables_touched``). The
    MCP ``query`` payload returns ``schema.table``; we collapse before
    the intersection so the recall calculation lines up.
    """
    return [(score, bare_table_name(name)) for score, name in hits]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_embeddings_flag(requested: bool, notes: list[str]) -> bool:
    """Mirror L1's resolver: probe ``onnxruntime`` for the [embeddings] extra."""
    if not requested:
        return False
    try:
        importlib.import_module(_EMBEDDINGS_SENTINEL)
    except ImportError:
        notes.append(
            "--embeddings was requested but the 'embeddings' extra is not "
            "installed; the semantic_search metric will be omitted. Install "
            "with `uv sync --extra embeddings` to enable it."
        )
        return False
    return True


def _import_semantic_search_payload() -> Any:
    """Forward-compatible import of the semantic_search MCP tool.

    Returns the ``semantic_search_payload`` callable when the module
    exists, or ``None`` when the module itself has not landed yet.
    This lets L2 keep its public contract ("emit
    ``semantic_search_recall_at_5`` when embeddings are installed")
    satisfiable the moment the tool ships, with no L2 code change.

    Raises:
        AttributeError: when the module loads but does not expose
            ``semantic_search_payload``. We deliberately do NOT swallow
            this — a future rename of the entry point should fail
            loudly here, not silently skip the metric.
    """
    try:
        module = importlib.import_module("pretensor.mcp.tools.semantic_search")
    except ImportError:
        return None
    try:
        return module.semantic_search_payload
    except AttributeError as exc:
        raise AttributeError(
            "pretensor.mcp.tools.semantic_search loaded but does not "
            "expose `semantic_search_payload`; the L2 runner expects "
            "this entry point. If the tool was renamed, update L2 to "
            "match — silently skipping the metric here would mask the "
            "regression."
        ) from exc


def _resolve_version() -> str:
    try:
        return importlib.metadata.version("pretensor")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+unknown"
