"""dbt exposures and ``sources.json`` freshness → ``SchemaTable`` signals.

Writes three independent signals onto matching ``SchemaTable`` rows:

* ``has_external_consumers`` — tables backing dbt exposures (and every
  upstream model/source reachable through ``parent_map``) are marked ``true``.
* ``staleness_status`` / ``staleness_as_of`` — dbt source freshness results
  from ``sources.json`` are persisted verbatim (``pass``/``warn``/``error``)
  with the most recent ``max_loaded_at`` timestamp. This is staleness metadata,
  NOT a rewrite of ``potentially_unused``.
* ``test_count`` — number of dbt tests attached to each model (via
  ``manifest.tests[*].attached_node``).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.manifest import DbtManifest
from pretensor.enrichment.dbt.resolution import resolve_dbt_parent_node_id

__all__ = ["DbtSignalsWriteStats", "write_dbt_signals"]

logger = logging.getLogger(__name__)

_VALID_STATUSES = frozenset({"pass", "warn", "error", "runtime error"})
_STALE_STATUSES = frozenset({"warn", "error", "runtime error"})


@dataclass(frozen=True, slots=True)
class DbtSignalsWriteStats:
    """Counts from ``write_dbt_signals`` for CLI summaries."""

    exposures_marked: int
    freshness_rows_applied: int
    tests_counted: int


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _iter_sources_json_results(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Normalize v3 ``results[]`` and legacy ``sources`` map shapes."""
    results = payload.get("results")
    if isinstance(results, list):
        return [r for r in results if isinstance(r, Mapping)]
    legacy = payload.get("sources")
    if isinstance(legacy, dict):
        out: list[Mapping[str, Any]] = []
        for _key, row in legacy.items():
            if isinstance(row, Mapping):
                out.append(row)
        return out
    return []


def _freshness_status(row: Mapping[str, Any]) -> str | None:
    raw = _as_str(row.get("status")) or _as_str(row.get("state"))
    if raw is None:
        return None
    return raw.strip().lower()


def _load_sources_freshness_payload(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        logger.warning("dbt sources.json not found or not a file: %s", path)
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("cannot read dbt sources.json %s: %s", path, exc)
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("dbt sources.json is not valid JSON %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("dbt sources.json root must be a JSON object: %s", path)
        return None
    return data


def _apply_freshness_from_sources_json(
    manifest: DbtManifest,
    store: KuzuStore,
    connection_name: str,
    path: Path,
) -> int:
    payload = _load_sources_freshness_payload(path)
    if payload is None:
        return 0
    applied = 0
    for row in _iter_sources_json_results(payload):
        uid = _as_str(row.get("unique_id"))
        if uid is None or not uid.startswith("source."):
            continue
        status = _freshness_status(row)
        if status is None or status not in _VALID_STATUSES:
            logger.debug("dbt freshness: skipping unknown status %r for %s", status, uid)
            continue
        table_nid = resolve_dbt_parent_node_id(manifest, connection_name, uid)
        if table_nid is None:
            logger.debug(
                "dbt freshness: could not resolve source %s to SchemaTable", uid
            )
            continue
        max_loaded = _as_str(row.get("max_loaded_at")) or ""
        updated = store.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $nid, connection_name: $cn})
            SET t.staleness_status = $status,
                t.staleness_as_of = $as_of
            RETURN t.node_id
            """,
            {
                "nid": table_nid,
                "cn": connection_name,
                "status": status,
                "as_of": max_loaded,
            },
        )
        if not updated:
            logger.debug(
                "dbt freshness: SchemaTable missing for source %s (node_id=%s)",
                uid,
                table_nid,
            )
            continue
        applied += 1
        if status in _STALE_STATUSES:
            ago = row.get("max_loaded_at_time_ago_in_s")
            logger.warning(
                "dbt source freshness: stale source %s status=%s max_loaded_at=%r "
                "max_loaded_at_time_ago_in_s=%r",
                uid,
                status,
                max_loaded,
                ago,
            )
    return applied


def _collect_exposure_reachable_dbt_nodes(manifest: DbtManifest) -> set[str]:
    """BFS upstream over ``parent_map`` from exposure ``depends_on`` model/source ids."""
    seeds: list[str] = []
    for exposure in manifest.exposures.values():
        for dep in exposure.depends_on_nodes:
            if dep.startswith("model.") or dep.startswith("source."):
                seeds.append(dep)
    if not seeds:
        return set()
    seen: set[str] = set()
    frontier: deque[str] = deque(seeds)
    while frontier:
        dbid = frontier.popleft()
        if dbid in seen:
            continue
        seen.add(dbid)
        if not dbid.startswith("model."):
            continue
        for parent_id in manifest.parent_map.get(dbid, ()):
            if parent_id.startswith("model.") or parent_id.startswith("source."):
                frontier.append(parent_id)
    return seen


def _mark_external_consumers(
    manifest: DbtManifest,
    store: KuzuStore,
    connection_name: str,
    dbt_node_ids: set[str],
) -> int:
    marked = 0
    for dbid in dbt_node_ids:
        if not (dbid.startswith("model.") or dbid.startswith("source.")):
            continue
        table_nid = resolve_dbt_parent_node_id(manifest, connection_name, dbid)
        if table_nid is None:
            logger.debug(
                "dbt signals: could not resolve %s to SchemaTable for exposure propagation",
                dbid,
            )
            continue
        updated = store.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $nid, connection_name: $cn})
            SET t.has_external_consumers = true
            RETURN t.node_id
            """,
            {"nid": table_nid, "cn": connection_name},
        )
        if not updated:
            logger.debug(
                "dbt signals: SchemaTable missing for exposure consumer %s (node_id=%s)",
                dbid,
                table_nid,
            )
            continue
        marked += 1
    return marked


def _apply_test_counts(
    manifest: DbtManifest,
    store: KuzuStore,
    connection_name: str,
) -> int:
    """Count ``DbtTest.attached_node`` per model and write ``SchemaTable.test_count``."""
    if not manifest.tests:
        return 0
    counter: Counter[str] = Counter()
    for test in manifest.tests.values():
        attached = test.attached_node
        if attached is None or not attached.startswith("model."):
            continue
        counter[attached] += 1
    if not counter:
        return 0
    applied = 0
    for model_id, n in counter.items():
        table_nid = resolve_dbt_parent_node_id(manifest, connection_name, model_id)
        if table_nid is None:
            continue
        updated = store.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $nid, connection_name: $cn})
            SET t.test_count = $n
            RETURN t.node_id
            """,
            {"nid": table_nid, "cn": connection_name, "n": int(n)},
        )
        if not updated:
            logger.debug(
                "dbt signals: SchemaTable missing for model %s (test_count=%d)",
                model_id,
                n,
            )
            continue
        applied += 1
    return applied


def write_dbt_signals(
    manifest: DbtManifest,
    store: KuzuStore,
    connection_name: str,
    sources_path: Path | None = None,
) -> DbtSignalsWriteStats:
    """Apply dbt exposure, freshness, and test-count signals to ``SchemaTable`` rows.

    **Exposures:** Tables backing models or sources listed under any exposure, and every
    upstream model/source reachable via ``parent_map``, get ``has_external_consumers=true``
    when a matching ``SchemaTable`` exists.

    **Freshness:** When ``sources_path`` points to dbt's ``sources.json``, every row with
    a recognized status (``pass``, ``warn``, ``error``, ``runtime error``) is persisted as
    ``staleness_status`` plus ``staleness_as_of`` (the ``max_loaded_at`` timestamp verbatim).
    Stale statuses (``warn``/``error``) additionally emit a logger warning. This function
    does not modify ``potentially_unused`` — that field remains governed by access stats.

    **Test counts:** For each ``DbtTest`` with ``attached_node`` pointing at a model,
    the number of tests is written to ``SchemaTable.test_count``.

    Args:
        manifest: Parsed dbt manifest (exposures, sources, models, tests, ``parent_map``).
        store: Open Kuzu store.
        connection_name: Pretensor connection name for ``SchemaTable`` nodes.
        sources_path: Optional path to ``sources.json`` from ``dbt source freshness``.

    Returns:
        Counts of tables marked / freshness rows applied / tests counted for CLI summaries.
    """
    freshness_applied = 0
    if sources_path is not None:
        freshness_applied = _apply_freshness_from_sources_json(
            manifest, store, connection_name, sources_path
        )

    exposures_marked = 0
    if manifest.exposures:
        reachable = _collect_exposure_reachable_dbt_nodes(manifest)
        if reachable:
            exposures_marked = _mark_external_consumers(
                manifest, store, connection_name, reachable
            )

    tests_counted = _apply_test_counts(manifest, store, connection_name)

    return DbtSignalsWriteStats(
        exposures_marked=exposures_marked,
        freshness_rows_applied=freshness_applied,
        tests_counted=tests_counted,
    )
