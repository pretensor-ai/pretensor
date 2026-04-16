"""E2E tests — VisibilityFilter hides tables and schemas at serve time."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pretensor.mcp.tools.context import context_payload
from pretensor.mcp.tools.list import list_databases_payload
from pretensor.visibility.config import VisibilityConfig, load_visibility_config
from pretensor.visibility.filter import VisibilityFilter

if not os.getenv("PRETENSOR_E2E"):
    pytest.skip("set PRETENSOR_E2E=1", allow_module_level=True)

pytestmark = pytest.mark.e2e


def test_hidden_table_not_in_context(graph_dir: Path) -> None:
    vf = VisibilityFilter.from_config(VisibilityConfig(hidden_tables=["public.film"]))
    result = context_payload(graph_dir, table="film", db="pagila", visibility_filter=vf)
    assert "error" in result, (
        f"Expected 'error' when film is hidden; got: {list(result)}"
    )


def test_hidden_table_reduces_list_count(graph_dir: Path) -> None:
    baseline = list_databases_payload(graph_dir)
    baseline_count = sum(
        db.get("table_count", 0) for db in baseline.get("databases", [])
    )

    vf = VisibilityFilter.from_config(VisibilityConfig(hidden_tables=["public.film"]))
    filtered = list_databases_payload(graph_dir, visibility_filter=vf)
    filtered_count = sum(
        db.get("table_count", 0) for db in filtered.get("databases", [])
    )

    assert filtered_count < baseline_count, (
        f"Expected filtered count ({filtered_count}) < baseline ({baseline_count})"
    )


def test_hidden_schema_excludes_all_tables(graph_dir: Path) -> None:
    # Hide public schema (7 tables); only staff.staff (1 table) should remain
    vf = VisibilityFilter.from_config(VisibilityConfig(hidden_schemas=["public"]))
    result = list_databases_payload(graph_dir, visibility_filter=vf)
    remaining = sum(
        db.get("table_count", 0) for db in result.get("databases", [])
    )
    assert remaining == 1, (
        f"Expected only staff.staff visible after hiding public schema; got {remaining}"
    )


def test_allowed_schemas_restricts_results(graph_dir: Path) -> None:
    vf = VisibilityFilter.from_config(VisibilityConfig(allowed_schemas=["public"]))
    result = list_databases_payload(graph_dir, visibility_filter=vf)
    count = sum(db.get("table_count", 0) for db in result.get("databases", []))
    assert count > 0, "Expected >0 tables when only public schema is allowed"


def test_visibility_yml_on_disk_respected(graph_dir: Path) -> None:
    vis_path = graph_dir / "visibility.yml"
    vis_path.write_text(
        "hidden_tables:\n  - public.payment\n", encoding="utf-8"
    )
    try:
        cfg = load_visibility_config(vis_path)
        vf = VisibilityFilter.from_config(cfg)
        result = context_payload(graph_dir, table="payment", db="pagila", visibility_filter=vf)
        assert "error" in result, (
            f"Expected 'error' when payment is hidden via visibility.yml; got: {list(result)}"
        )
    finally:
        vis_path.unlink(missing_ok=True)
