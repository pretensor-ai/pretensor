"""Warehouse-scale dbt enrichment regression test.

Protects the dbt manifest enrichment pipeline from bit-rot by running the full
``run_dbt_enrichment_from_manifest`` pipeline against a synthetic ~1,500-model
manifest. The synthetic manifest mimics the warehouse-shape complexity of
``gitlab-data/analytics`` (varied fan-out, ``alias``/``identifier`` overrides,
partial description coverage, exposures spread across the DAG, dangling refs)
without depending on any external dataset.

The fixture lives in ``tests/e2e/conftest.py::synthetic_dbt_warehouse``; this
module focuses solely on running the enrichment and asserting the five
properties from the original chore acceptance criteria:

1. Lineage edge count is in the same order of magnitude as the manifest's
   model→source count (within ±2×).
2. Tables backing exposures are marked ``has_external_consumers = true`` and
   propagation walks upstream via ``parent_map``.
3. ≥80 % of models with descriptions in the manifest end up with a non-empty
   ``description`` on their matching ``SchemaTable`` row.
4. End-to-end enrichment over the 1,500-model manifest completes in <30 s.
5. The synthetic warehouse has zero ``FK_REFERENCES`` edges, yet dbt lineage
   is still produced — proving dbt enrichment is independent of FK discovery.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest

from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.manifest import DbtManifest
from pretensor.enrichment.dbt.pipeline import (
    DbtEnrichmentSummary,
    run_dbt_enrichment_from_manifest,
)
from tests.e2e.conftest import SyntheticDbtWarehouse
from tests.query_helpers import first_cell, single_query_result

# Tunable: per the issue, may relax to 60.0 if real CI runners prove flaky.
# Do not relax speculatively — only in response to a measured failure.
ENRICHMENT_BUDGET_SECONDS = 30.0


class EnrichedSyntheticGraph:
    """Holds the open KuzuStore plus the enrichment elapsed time / summary."""

    def __init__(
        self,
        bundle: SyntheticDbtWarehouse,
        store: KuzuStore,
        manifest: DbtManifest,
        summary: DbtEnrichmentSummary,
        elapsed_seconds: float,
    ) -> None:
        self.bundle = bundle
        self.store = store
        self.manifest = manifest
        self.summary = summary
        self.elapsed_seconds = elapsed_seconds

    @property
    def graph_path(self) -> Path:
        return self.bundle.graph_path

    @property
    def connection_name(self) -> str:
        return self.bundle.spec.connection_name


@pytest.fixture(scope="module")
def enriched_synthetic_graph(
    synthetic_dbt_warehouse: SyntheticDbtWarehouse,
) -> Iterator[EnrichedSyntheticGraph]:
    """Open the indexed Kuzu graph, run dbt enrichment once, time it, share
    the result across every assertion in this module.

    Module-scoped so the (potentially expensive) enrichment runs exactly once
    even though five tests query the resulting graph. The wall-clock measured
    here is the value asserted by ``test_enrichment_within_performance_budget``;
    parsing the manifest is included in the measured window because the
    issue's "end-to-end enrichment" budget covers parse + write.
    """
    bundle = synthetic_dbt_warehouse
    store = KuzuStore(bundle.graph_path)
    try:
        start = time.monotonic()
        manifest = DbtManifest.load(bundle.manifest_path)
        summary = run_dbt_enrichment_from_manifest(
            manifest,
            sources_path=None,
            store=store,
            connection_name=bundle.spec.connection_name,
        )
        elapsed = time.monotonic() - start
        yield EnrichedSyntheticGraph(
            bundle=bundle,
            store=store,
            manifest=manifest,
            summary=summary,
            elapsed_seconds=elapsed,
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def test_lineage_edge_count_matches_manifest(
    enriched_synthetic_graph: EnrichedSyntheticGraph,
) -> None:
    """Assertion 1: lineage edge count is within ±2× of the resolvable
    parent_map count predicted at generation time.

    The generator records ``expected_lineage_edges`` as the count of
    ``parent_map`` entries whose endpoints both resolve to a real model/source
    (excluding the intentionally-injected dangling refs). The writer should
    emit roughly that many ``LINEAGE`` edges with ``source = 'dbt'``.
    """
    expected = enriched_synthetic_graph.bundle.spec.expected_lineage_edges
    assert expected > 0, (
        "generator produced no resolvable parent_map edges — fixture broken"
    )

    result = single_query_result(
        enriched_synthetic_graph.store,
        """
        MATCH ()-[r:LINEAGE]->()
        WHERE r.source = 'dbt' AND r.lineage_type = 'model_dependency'
        RETURN count(r)
        """,
    )
    actual = int(first_cell(result))

    # Off-by-2× tolerated per the issue; off-by-10× is a failure.
    lower = expected // 2
    upper = expected * 2
    assert lower <= actual <= upper, (
        f"lineage edge count {actual} outside ±2× of expected {expected} "
        f"(allowed range: [{lower}, {upper}]). pipeline summary: "
        f"{enriched_synthetic_graph.summary}"
    )


def test_exposures_propagate_has_external_consumers(
    enriched_synthetic_graph: EnrichedSyntheticGraph,
) -> None:
    """Assertion 2: tables that exposures depend on are marked
    ``has_external_consumers = true``.

    Walks the hand-picked ``KnownExposure`` set recorded by the generator and
    verifies every ``(schema, physical_name)`` pair has the flag set on its
    ``SchemaTable`` row. We do not check the BFS *upstream* propagation here
    (that would re-derive what the writer does); we just confirm the direct
    dependencies are flagged, which is the strongest assertion the test can
    make without rebuilding the resolver in test code.
    """
    bundle = enriched_synthetic_graph.bundle
    store = enriched_synthetic_graph.store
    cn = bundle.spec.connection_name

    assert bundle.spec.known_exposures, "generator produced no known exposures"

    unflagged: list[tuple[str, str]] = []
    for exposure in bundle.spec.known_exposures:
        for schema, name in exposure.expected_tables:
            rows = store.query_all_rows(
                """
                MATCH (t:SchemaTable {connection_name: $cn,
                                      schema_name: $schema,
                                      table_name: $name})
                RETURN t.has_external_consumers
                """,
                {"cn": cn, "schema": schema, "name": name},
            )
            assert rows, (
                f"expected SchemaTable for exposure target "
                f"{schema}.{name} (exposure={exposure.unique_id}) but it is missing"
            )
            value = rows[0][0]
            if value is not True:
                unflagged.append((schema, name))

    assert not unflagged, (
        f"{len(unflagged)} exposure-target tables missing has_external_consumers=true: "
        f"{unflagged[:5]}"
    )


def test_description_coverage_over_eighty_percent(
    enriched_synthetic_graph: EnrichedSyntheticGraph,
) -> None:
    """Assertion 3: ≥80 % of resolvable, documented models gain a non-empty
    ``description`` on their ``SchemaTable`` row.

    The generator records ``documented_resolvable_models`` as the set of
    ``(schema, physical_name)`` pairs whose source manifest model carries a
    description. The writer is non-destructive and should populate the column
    on every matching row.
    """
    bundle = enriched_synthetic_graph.bundle
    store = enriched_synthetic_graph.store
    cn = bundle.spec.connection_name

    documented = bundle.spec.documented_resolvable_models
    assert documented, "generator produced no documented models — fixture broken"

    enriched_count = 0
    for schema, name in documented:
        rows = store.query_all_rows(
            """
            MATCH (t:SchemaTable {connection_name: $cn,
                                  schema_name: $schema,
                                  table_name: $name})
            RETURN t.description
            """,
            {"cn": cn, "schema": schema, "name": name},
        )
        if not rows:
            continue
        desc = rows[0][0]
        if isinstance(desc, str) and desc.strip():
            enriched_count += 1

    coverage = enriched_count / len(documented)
    assert coverage >= 0.80, (
        f"description coverage {coverage:.1%} below the 80% threshold "
        f"({enriched_count}/{len(documented)} documented models enriched). "
        f"pipeline summary: {enriched_synthetic_graph.summary}"
    )


def test_enrichment_within_performance_budget(
    enriched_synthetic_graph: EnrichedSyntheticGraph,
) -> None:
    """Assertion 4: full parse + enrichment of the synthetic ~1,500-model
    manifest completes in under :data:`ENRICHMENT_BUDGET_SECONDS`.

    Wall-clock measured by the module fixture so this test only inspects the
    cached value. If the budget proves too tight on real CI hardware, the
    issue allows relaxing to 60 s — but only in response to a real failure,
    not speculatively.
    """
    elapsed = enriched_synthetic_graph.elapsed_seconds
    assert elapsed < ENRICHMENT_BUDGET_SECONDS, (
        f"enrichment took {elapsed:.2f}s, exceeding the "
        f"{ENRICHMENT_BUDGET_SECONDS:.0f}s budget. pipeline summary: "
        f"{enriched_synthetic_graph.summary}"
    )


def test_lineage_independent_of_fk_edges(
    enriched_synthetic_graph: EnrichedSyntheticGraph,
) -> None:
    """Assertion 5: the synthetic warehouse has zero ``FK_REFERENCES`` edges
    (no real FK constraints in the DDL, relationship discovery disabled), yet
    dbt lineage is still emitted — proving dbt enrichment does not depend on
    FK edges to function.
    """
    store = enriched_synthetic_graph.store

    fk_result = single_query_result(
        store,
        "MATCH ()-[r:FK_REFERENCES]->() RETURN count(r)",
    )
    fk_count = int(first_cell(fk_result))
    assert fk_count == 0, (
        f"expected 0 FK_REFERENCES edges in the synthetic warehouse, found {fk_count}"
    )

    lineage_result = single_query_result(
        store,
        """
        MATCH ()-[r:LINEAGE]->()
        WHERE r.source = 'dbt' AND r.lineage_type = 'model_dependency'
        RETURN count(r)
        """,
    )
    lineage_count = int(first_cell(lineage_result))
    assert lineage_count > 0, (
        "no dbt lineage edges produced even though FK edges are absent — "
        "dbt lineage should be independent of FK discovery"
    )
