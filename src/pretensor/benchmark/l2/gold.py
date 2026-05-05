"""Gold-data loaders for the L2 benchmark.

Reads the per-dataset NL→SQL bench JSON and the metric-templates YAML
file into typed dataclasses the runner consumes. Centralised here so
the runner stays thin and the loaders are independently testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml

from pretensor.benchmark.fixtures import Fixture
from pretensor.benchmark.l2.metrics import JoinPair
from pretensor.connectors.lineage_sqlglot import table_refs_from_sql

__all__ = [
    "MetricTemplateEntry",
    "QueryGoldEntry",
    "TraverseGoldEntry",
    "bare_table_name",
    "default_schema_for",
    "load_metric_templates",
    "load_query_gold",
    "load_traverse_gold",
]


# Default schema applied to unqualified table references in expected_sql.
# Every gold SQL we ship today either uses public.* (Pagila, TPC-H) or is
# fully schema-qualified (AdventureWorks: humanresources.*, sales.*, etc.),
# so the default is only an absolute-last-resort fallback. Datasets we
# don't list here use ``"public"`` for the same reason.
_DEFAULT_SCHEMAS: dict[str, str] = {
    "pagila": "public",
    "tpch": "public",
}


@dataclass(frozen=True, slots=True)
class QueryGoldEntry:
    """One question→tables observation for the ``query`` Recall@K metric."""

    id: str
    question: str
    expected_sql: str
    tables_touched: frozenset[str]


@dataclass(frozen=True, slots=True)
class TraverseGoldEntry:
    """One ``(from_table, to_table, gold_path)`` entry for traverse correctness.

    ``gold_path`` is the canonical sequence of ``(from_table, to_table)``
    hops (schema-qualified bare names). The first hop's ``from_table`` is
    the traverse start; the last hop's ``to_table`` is the destination.
    """

    id: str
    from_table: str
    to_table: str
    gold_path: tuple[JoinPair, ...]


@dataclass(frozen=True, slots=True)
class MetricTemplateEntry:
    """One semantic-layer metric template for ``compile_metric``."""

    metric: str
    semantic_yaml: str
    database: str
    # Optional sidecar for diagnostics — never used by the metric value.
    notes: str = field(default="")


def default_schema_for(dataset_name: str) -> str:
    """Return the default schema to use when SQL refs are unqualified."""
    return _DEFAULT_SCHEMAS.get(dataset_name, "public")


def load_query_gold(fixture: Fixture) -> list[QueryGoldEntry]:
    """Load the bench JSON and resolve ``tables_touched`` for each entry.

    When an entry already has an explicit ``tables_touched`` array we use
    it verbatim. When absent, we parse ``expected_sql`` via sqlglot and
    extract the bare table names. Entries whose ``expected_sql`` fails to
    parse are returned with an empty ``tables_touched`` set so the
    metric loop skips them — the runner notes the parse failure.
    """
    if fixture.questions_path is None:
        return []
    raw = json.loads(fixture.questions_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []

    default_schema = default_schema_for(fixture.name.value)
    entries: list[QueryGoldEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("id", "")).strip()
        question = str(item.get("question", ""))
        expected_sql = str(item.get("expected_sql", ""))
        if not identifier:
            continue

        tables = item.get("tables_touched")
        if isinstance(tables, list) and all(isinstance(t, str) for t in tables):
            tables_set = frozenset(bare_table_name(t) for t in tables)
        else:
            tables_set = frozenset(
                t
                for _, t in table_refs_from_sql(
                    expected_sql,
                    dialect="postgres",
                    default_schema=default_schema,
                )
            )

        entries.append(
            QueryGoldEntry(
                id=identifier,
                question=question,
                expected_sql=expected_sql,
                tables_touched=tables_set,
            )
        )
    return entries


def load_traverse_gold(fixture: Fixture) -> list[TraverseGoldEntry]:
    """Pull ``gold_path`` entries out of the bench JSON.

    Only entries that explicitly carry a ``gold_path`` array are returned
    — the metric ignores entries authored only for ``query`` Recall@K.
    """
    if fixture.questions_path is None:
        return []
    raw = json.loads(fixture.questions_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []

    out: list[TraverseGoldEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        gold_path_raw = item.get("gold_path")
        if not isinstance(gold_path_raw, list) or not gold_path_raw:
            continue

        hops: list[JoinPair] = []
        for hop in gold_path_raw:
            if not isinstance(hop, dict):
                continue
            frm = hop.get("from_table")
            to = hop.get("to_table")
            if isinstance(frm, str) and isinstance(to, str):
                hops.append((frm, to))
        if not hops:
            continue

        identifier = str(item.get("id", "")).strip()
        out.append(
            TraverseGoldEntry(
                id=identifier or f"{hops[0][0]}->{hops[-1][1]}",
                from_table=hops[0][0],
                to_table=hops[-1][1],
                gold_path=tuple(hops),
            )
        )
    return out


def load_metric_templates(fixture: Fixture) -> list[MetricTemplateEntry]:
    """Parse the per-dataset metric-templates YAML.

    The YAML file lives at ``scripts/data/<dataset>_metric_templates.yaml``
    and shadows :class:`Fixture.metric_templates_path`. Returns an empty
    list when the file is absent — the metric reports
    :class:`compile_metric_correctness` as ``None`` upstream.

    Format:

    .. code-block:: yaml

        connection_name: pagila
        templates:
          - metric: total_rentals
            notes: Optional human-readable rationale
            semantic_yaml: |
              connection_name: pagila
              domains:
                - name: rentals
                  ...

    The ``connection_name`` at the top level is forwarded as the
    ``database`` argument to ``compile_metric_payload``.
    """
    path = fixture.metric_templates_path
    if path is None or not path.exists():
        return []

    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return []

    database = str(raw.get("connection_name", fixture.name.value))
    templates_raw = raw.get("templates", [])
    if not isinstance(templates_raw, list):
        return []

    out: list[MetricTemplateEntry] = []
    for item in templates_raw:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric", "")).strip()
        semantic_yaml = str(item.get("semantic_yaml", ""))
        if not metric or not semantic_yaml.strip():
            continue
        out.append(
            MetricTemplateEntry(
                metric=metric,
                semantic_yaml=semantic_yaml,
                database=database,
                notes=str(item.get("notes", "")),
            )
        )
    return out


def bare_table_name(name: str) -> str:
    """Strip the schema prefix from a possibly-qualified table name."""
    return name.split(".", 1)[1] if "." in name else name
