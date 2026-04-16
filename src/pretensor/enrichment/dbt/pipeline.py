"""Orchestrate dbt manifest enrichment passes (lineage, metadata, signals)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pretensor.core.store import KuzuStore
from pretensor.enrichment.dbt.lineage import write_dbt_lineage
from pretensor.enrichment.dbt.manifest import DbtManifest
from pretensor.enrichment.dbt.metadata import write_dbt_metadata
from pretensor.enrichment.dbt.signals import write_dbt_signals

__all__ = ["DbtEnrichmentSummary", "run_dbt_enrichment", "run_dbt_enrichment_from_manifest"]


@dataclass(frozen=True, slots=True)
class DbtEnrichmentSummary:
    """Aggregate counts after running all dbt enrichment writers."""

    lineage_edges: int
    tables_enriched: int
    tags_set: int
    exposures_marked: int
    freshness_rows_applied: int
    tests_counted: int


def run_dbt_enrichment_from_manifest(
    manifest: DbtManifest,
    sources_path: Path | None,
    store: KuzuStore,
    connection_name: str,
) -> DbtEnrichmentSummary:
    """Run lineage → metadata → signals on an already-parsed dbt manifest.

    Exists so CLI commands can preload/validate the manifest *before* mutating
    the graph and still share the same orchestration as ``run_dbt_enrichment``.
    """
    lineage_edges = write_dbt_lineage(manifest, store, connection_name)
    meta_stats = write_dbt_metadata(manifest, store, connection_name)
    signal_stats = write_dbt_signals(
        manifest, store, connection_name, sources_path=sources_path
    )
    return DbtEnrichmentSummary(
        lineage_edges=lineage_edges,
        tables_enriched=meta_stats.tables_enriched,
        tags_set=meta_stats.tags_set,
        exposures_marked=signal_stats.exposures_marked,
        freshness_rows_applied=signal_stats.freshness_rows_applied,
        tests_counted=signal_stats.tests_counted,
    )


def run_dbt_enrichment(
    manifest_path: Path,
    sources_path: Path | None,
    store: KuzuStore,
    connection_name: str,
) -> DbtEnrichmentSummary:
    """Load a dbt manifest and run lineage, metadata, then exposure/freshness signals.

    Order matches the dbt enrichment pipeline sequence.

    Args:
        manifest_path: Path to dbt ``target/manifest.json`` (after ``dbt compile`` / ``dbt run``).
        sources_path: Optional ``sources.json`` from ``dbt source freshness`` (or ``None``).
        store: Open Kuzu graph store for the indexed connection.
        connection_name: Logical Pretensor connection name for this database.

    Returns:
        Counts for CLI summary lines.
    """
    manifest = DbtManifest.load(manifest_path)
    return run_dbt_enrichment_from_manifest(
        manifest, sources_path, store, connection_name
    )
