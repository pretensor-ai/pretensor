"""Tests for CLI dbt path helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from rich.console import Console

from pretensor.cli.dbt_enrichment import (
    apply_dbt_enrichment_cli,
    preload_dbt_manifest,
    resolve_dbt_sources_path,
    validate_dbt_manifest_path,
)
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.graph_models.node import GraphNode


def _minimal_table(connection: str, schema: str, name: str) -> GraphNode:
    return GraphNode(
        node_id=table_node_id(connection, schema, name),
        connection_name=connection,
        database="wh",
        schema_name=schema,
        table_name=name,
        row_count=None,
        comment=None,
        entity_type=None,
        table_type="table",
        seq_scan_count=None,
        idx_scan_count=None,
        insert_count=None,
        update_count=None,
        delete_count=None,
        is_partitioned=None,
        partition_key=None,
        grants_json=None,
        access_read_count=None,
        access_write_count=None,
        days_since_last_access=None,
        potentially_unused=None,
        table_bytes=None,
        clustering_key=None,
    )


def test_resolve_dbt_sources_path_default_next_to_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    sources = tmp_path / "sources.json"
    sources.write_text("{}", encoding="utf-8")
    assert resolve_dbt_sources_path(manifest, None) == sources


def test_resolve_dbt_sources_path_explicit_must_exist(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="dbt sources file not found"):
        resolve_dbt_sources_path(manifest, missing)


def test_validate_dbt_manifest_path_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dbt manifest not found"):
        validate_dbt_manifest_path(tmp_path / "missing.json")


def test_preload_dbt_manifest_exits_on_bad_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("not json {{{", encoding="utf-8")
    console = Console(record=True, width=120)
    with pytest.raises(typer.Exit) as excinfo:
        preload_dbt_manifest(manifest, None, console=console)
    assert excinfo.value.exit_code == 1
    text = console.export_text()
    assert "dbt manifest is not valid JSON" in text


def test_preload_dbt_manifest_exits_on_missing_manifest(tmp_path: Path) -> None:
    console = Console(record=True, width=120)
    with pytest.raises(typer.Exit) as excinfo:
        preload_dbt_manifest(tmp_path / "missing.json", None, console=console)
    assert excinfo.value.exit_code == 1
    text = console.export_text()
    assert "dbt manifest not found" in text


def test_apply_dbt_enrichment_cli_summary_line(
    tmp_path: Path,
) -> None:
    cn = "warehouse"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "metadata": {
                    "dbt_schema_version": (
                        "https://schemas.getdbt.com/dbt/manifest/v10.json"
                    )
                },
                "nodes": {
                    "model.p.t": {
                        "resource_type": "model",
                        "name": "t",
                        "schema": "public",
                        "tags": ["a"],
                    }
                },
                "sources": {},
                "exposures": {},
                "parent_map": {"model.p.t": []},
            }
        ),
        encoding="utf-8",
    )
    store = KuzuStore(Path(":memory:"))
    store.ensure_schema()
    try:
        store.upsert_table(_minimal_table(cn, "public", "t"))
        console = Console(record=True, width=120)
        preloaded, sources = preload_dbt_manifest(manifest, None, console=console)
        apply_dbt_enrichment_cli(
            manifest=preloaded,
            sources_path=sources,
            store=store,
            connection_name=cn,
            console=console,
        )
        text = console.export_text()
        assert "dbt enrichment:" in text
        assert "lineage_edges=0" in text
        assert "tables_enriched=1" in text
        assert "tags_set=1" in text
        assert "exposures_marked=0" in text
        assert "freshness_rows=0" in text
        assert "tests_counted=0" in text
    finally:
        store.close()
