"""Tests for dbt manifest loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pretensor.enrichment.dbt.manifest import DbtManifest, DbtManifestError

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dbt"


def test_load_v9_manifest() -> None:
    manifest = DbtManifest.load(FIXTURES / "manifest_v9.json")
    assert manifest.schema_version is not None
    assert "v9" in manifest.schema_version.lower()

    assert len(manifest.nodes) == 1
    orders = manifest.nodes["model.my_pkg.orders"]
    assert orders.name == "orders"
    assert orders.description == "Orders fact table"
    assert orders.tags == ("core", "finance")
    assert orders.database == "analytics"
    assert orders.schema_name == "marts"
    assert orders.alias == "orders"
    assert orders.column_descriptions.get("id") == "Primary key"

    assert len(manifest.sources) == 1
    src = manifest.sources["source.my_pkg.raw.orders"]
    assert src.source_name == "raw"
    assert src.identifier == "orders"

    assert len(manifest.exposures) == 1
    exp = manifest.exposures["exposure.my_pkg.looker_dashboard"]
    assert exp.depends_on_nodes == ("model.my_pkg.orders",)

    assert len(manifest.tests) == 1
    test = manifest.tests["test.my_pkg.unique_orders_id"]
    assert test.attached_node == "model.my_pkg.orders"

    assert manifest.parent_map["model.my_pkg.orders"] == ["source.my_pkg.raw.orders"]


def test_load_v10_manifest() -> None:
    manifest = DbtManifest.load(FIXTURES / "manifest_v10.json")
    assert manifest.schema_version is not None
    assert "v10" in manifest.schema_version.lower()
    assert "model.pkg2.dim_customer" in manifest.nodes
    dim = manifest.nodes["model.pkg2.dim_customer"]
    assert dim.name == "dim_customer"
    assert dim.description is None
    assert dim.tags == ()


def test_missing_file_raises() -> None:
    missing = FIXTURES / "does_not_exist.json"
    with pytest.raises(DbtManifestError, match="not found"):
        DbtManifest.load(missing)


def test_malformed_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DbtManifestError, match="not valid JSON"):
        DbtManifest.load(bad)


def test_graceful_defaults_for_minimal_node(tmp_path: Path) -> None:
    minimal = {
        "metadata": {},
        "nodes": {
            "model.x.y": {"resource_type": "model"},
        },
        "sources": {},
        "exposures": {},
    }
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps(minimal), encoding="utf-8")
    manifest = DbtManifest.load(path)
    model = manifest.nodes["model.x.y"]
    assert model.name == "model.x.y"
    assert model.description is None
    assert model.tags == ()
    assert model.database is None


def test_unsupported_schema_version_logs_warning(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    import logging

    caplog.set_level(logging.WARNING)
    weird = {
        "metadata": {
            "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v8.json"
        },
        "nodes": {},
        "sources": {},
        "exposures": {},
    }
    path = tmp_path / "manifest_weird_schema.json"
    path.write_text(json.dumps(weird), encoding="utf-8")
    DbtManifest.load(path)
    assert any("unsupported" in r.message.lower() for r in caplog.records)
