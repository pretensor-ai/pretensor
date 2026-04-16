"""Tests for visibility YAML parsing and glob matching."""

from __future__ import annotations

from pathlib import Path

import pytest

from pretensor.visibility.config import (
    load_visibility_config,
    merge_profile_into_base,
)
from pretensor.visibility.filter import VisibilityFilter


def test_load_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "visibility.yml"
    p.write_text("", encoding="utf-8")
    cfg = load_visibility_config(p)
    assert cfg.hidden_tables == []


def test_merge_profile_via_helpers(tmp_path: Path) -> None:
    p = tmp_path / "v.yml"
    p.write_text(
        """
hidden_tables: ["pii_*"]
profiles:
  external:
    allowed_schemas: ["public"]
""",
        encoding="utf-8",
    )
    base = load_visibility_config(p)
    merged = merge_profile_into_base(base, "external")
    vf = VisibilityFilter.from_config(merged)
    assert vf.is_table_visible("demo", "public", "users")
    assert not vf.is_table_visible("demo", "hr", "pii_salary")


def test_unknown_profile_raises(tmp_path: Path) -> None:
    p = tmp_path / "v.yml"
    p.write_text("hidden_tables: []\n", encoding="utf-8")
    base = load_visibility_config(p)
    with pytest.raises(ValueError, match="Unknown visibility profile"):
        merge_profile_into_base(base, "nope")


def test_fnmatch_hidden_columns() -> None:
    from pretensor.visibility.config import VisibilityConfig

    cfg = VisibilityConfig(hidden_columns=["*.ssn", "public.users.email"])
    vf = VisibilityFilter.from_config(cfg)
    assert vf.visible_columns("c", "public", "users", ["id", "email"]) == ["id"]
    assert "ssn" not in vf.visible_columns("c", "hr", "employees", ["ssn", "name"])


def test_connection_qualified_table_pattern() -> None:
    from pretensor.visibility.config import VisibilityConfig

    cfg = VisibilityConfig(hidden_tables=["demo::public.secret"])
    vf = VisibilityFilter.from_config(cfg)
    assert not vf.is_table_visible("demo", "public", "secret")
    assert vf.is_table_visible("other", "public", "secret")


def test_allowed_schemas_union_with_profile() -> None:
    """Base + profile allowed_schemas are merged (union), not replaced."""
    from pretensor.visibility.config import VisibilityConfig

    base = VisibilityConfig(
        allowed_schemas=["public"],
        profiles={},
    )
    from pretensor.visibility.config import VisibilityProfileRule

    base = base.model_copy(
        update={
            "profiles": {
                "extra": VisibilityProfileRule(allowed_schemas=["analytics"])
            }
        }
    )
    merged = merge_profile_into_base(base, "extra")
    vf = VisibilityFilter.from_config(merged)
    assert vf.is_table_visible("c", "public", "orders")
    assert vf.is_table_visible("c", "analytics", "metrics")
    assert not vf.is_table_visible("c", "private", "secrets")


def test_allowed_schemas_profile_only_acts_as_whitelist() -> None:
    """When only the profile sets allowed_schemas, it becomes the whitelist."""
    from pretensor.visibility.config import VisibilityConfig, VisibilityProfileRule

    base = VisibilityConfig(
        profiles={"restricted": VisibilityProfileRule(allowed_schemas=["public"])}
    )
    merged = merge_profile_into_base(base, "restricted")
    vf = VisibilityFilter.from_config(merged)
    assert vf.is_table_visible("c", "public", "orders")
    assert not vf.is_table_visible("c", "hr", "employees")


def test_allowed_tables_empty_no_behavior_change() -> None:
    """Empty allowed_tables does not restrict visibility (hide lists still apply)."""
    from pretensor.visibility.config import VisibilityConfig

    cfg = VisibilityConfig(
        allowed_tables=[],
        hidden_tables=["analytics.secret"],
    )
    vf = VisibilityFilter.from_config(cfg)
    assert vf.is_table_visible("c", "analytics", "orders")
    assert not vf.is_table_visible("c", "analytics", "secret")


def test_allowed_tables_exact_hides_other_tables_in_schema() -> None:
    from pretensor.visibility.config import VisibilityConfig

    vf = VisibilityFilter.from_config(
        VisibilityConfig(allowed_tables=["analytics.orders"])
    )
    assert vf.is_table_visible("c", "analytics", "orders")
    assert not vf.is_table_visible("c", "analytics", "users")
    assert not vf.is_table_visible("c", "public", "anything")


def test_allowed_tables_glob_pattern() -> None:
    from pretensor.visibility.config import VisibilityConfig

    vf = VisibilityFilter.from_config(
        VisibilityConfig(allowed_tables=["analytics.*_staging"])
    )
    assert vf.is_table_visible("c", "analytics", "users_staging")
    assert not vf.is_table_visible("c", "analytics", "users_prod")


def test_allowed_tables_union_with_profile() -> None:
    """Base + profile allowed_tables are union-merged when the profile contributes."""
    from pretensor.visibility.config import VisibilityConfig, VisibilityProfileRule

    base = VisibilityConfig(
        allowed_tables=["public.orders"],
        profiles={
            "extra": VisibilityProfileRule(allowed_tables=["analytics.metrics"])
        },
    )
    merged = merge_profile_into_base(base, "extra")
    vf = VisibilityFilter.from_config(merged)
    assert vf.is_table_visible("c", "public", "orders")
    assert vf.is_table_visible("c", "analytics", "metrics")
    assert not vf.is_table_visible("c", "public", "users")


def test_allowed_tables_profile_only_acts_as_whitelist() -> None:
    from pretensor.visibility.config import VisibilityConfig, VisibilityProfileRule

    base = VisibilityConfig(
        profiles={"t": VisibilityProfileRule(allowed_tables=["x.y"])}
    )
    merged = merge_profile_into_base(base, "t")
    vf = VisibilityFilter.from_config(merged)
    assert vf.is_table_visible("c", "x", "y")
    assert not vf.is_table_visible("c", "x", "z")


def test_allowed_tables_and_allowed_schemas_both_required() -> None:
    """When both allowlists are non-empty, a table must match both."""
    from pretensor.visibility.config import VisibilityConfig

    vf = VisibilityFilter.from_config(
        VisibilityConfig(
            allowed_schemas=["analytics"],
            allowed_tables=["analytics.orders", "public.users"],
        )
    )
    assert vf.is_table_visible("c", "analytics", "orders")
    assert not vf.is_table_visible("c", "public", "users")
    assert not vf.is_table_visible("c", "analytics", "other")


def test_allowed_tables_connection_qualified_pattern() -> None:
    from pretensor.visibility.config import VisibilityConfig

    vf = VisibilityFilter.from_config(
        VisibilityConfig(allowed_tables=["demo::public.orders"])
    )
    assert vf.is_table_visible("demo", "public", "orders")
    assert not vf.is_table_visible("other", "public", "orders")


# ---------------------------------------------------------------------------
# _change_visible — unit tests (pure function, no DB required)
# ---------------------------------------------------------------------------


def _make_change_visible_vf() -> VisibilityFilter:
    from pretensor.visibility.config import VisibilityConfig

    return VisibilityFilter.from_config(
        VisibilityConfig(hidden_tables=["public.users", "private.*"])
    )


def test_change_visible_table_hidden() -> None:
    from pretensor.connectors.snapshot import ChangeTarget, ChangeType, SchemaChange
    from pretensor.mcp.tools.detect_changes import _change_visible

    vf = _make_change_visible_vf()
    ch = SchemaChange(
        change_type=ChangeType.ADDED,
        target=ChangeTarget.TABLE,
        schema_name="public",
        table_name="users",
    )
    assert not _change_visible(ch, connection_name="demo", vf=vf)


def test_change_visible_table_visible() -> None:
    from pretensor.connectors.snapshot import ChangeTarget, ChangeType, SchemaChange
    from pretensor.mcp.tools.detect_changes import _change_visible

    vf = _make_change_visible_vf()
    ch = SchemaChange(
        change_type=ChangeType.ADDED,
        target=ChangeTarget.TABLE,
        schema_name="public",
        table_name="orders",
    )
    assert _change_visible(ch, connection_name="demo", vf=vf)


def test_change_visible_column_hidden_table() -> None:
    from pretensor.connectors.snapshot import ChangeTarget, ChangeType, SchemaChange
    from pretensor.mcp.tools.detect_changes import _change_visible

    vf = _make_change_visible_vf()
    ch = SchemaChange(
        change_type=ChangeType.ADDED,
        target=ChangeTarget.COLUMN,
        schema_name="public",
        table_name="users",
        column_name="email",
    )
    assert not _change_visible(ch, connection_name="demo", vf=vf)


def test_change_visible_lineage_source_hidden() -> None:
    from pretensor.connectors.snapshot import ChangeTarget, ChangeType, SchemaChange
    from pretensor.mcp.tools.detect_changes import _change_visible

    vf = _make_change_visible_vf()
    ch = SchemaChange(
        change_type=ChangeType.ADDED,
        target=ChangeTarget.LINEAGE,
        schema_name="public",
        table_name="users",  # source is hidden
        details="VIEW: public.users → public.user_summary",
    )
    assert not _change_visible(ch, connection_name="demo", vf=vf)


def test_change_visible_lineage_target_hidden() -> None:
    from pretensor.connectors.snapshot import ChangeTarget, ChangeType, SchemaChange
    from pretensor.mcp.tools.detect_changes import _change_visible

    vf = _make_change_visible_vf()
    ch = SchemaChange(
        change_type=ChangeType.ADDED,
        target=ChangeTarget.LINEAGE,
        schema_name="public",
        table_name="orders",  # source is visible
        details="VIEW: public.orders → public.users",  # target is hidden
    )
    assert not _change_visible(ch, connection_name="demo", vf=vf)


def test_change_visible_lineage_both_visible() -> None:
    from pretensor.connectors.snapshot import ChangeTarget, ChangeType, SchemaChange
    from pretensor.mcp.tools.detect_changes import _change_visible

    vf = _make_change_visible_vf()
    ch = SchemaChange(
        change_type=ChangeType.ADDED,
        target=ChangeTarget.LINEAGE,
        schema_name="public",
        table_name="orders",
        details="VIEW: public.orders → public.order_summary",
    )
    assert _change_visible(ch, connection_name="demo", vf=vf)


def test_change_visible_lineage_no_arrow_passthrough() -> None:
    """When details lacks the expected arrow format, the change passes through."""
    from pretensor.connectors.snapshot import ChangeTarget, ChangeType, SchemaChange
    from pretensor.mcp.tools.detect_changes import _change_visible

    vf = _make_change_visible_vf()
    ch = SchemaChange(
        change_type=ChangeType.ADDED,
        target=ChangeTarget.LINEAGE,
        schema_name="public",
        table_name="orders",
        details="some unexpected format without arrow",
    )
    assert _change_visible(ch, connection_name="demo", vf=vf)


def test_change_visible_wildcard_schema() -> None:
    from pretensor.connectors.snapshot import ChangeTarget, ChangeType, SchemaChange
    from pretensor.mcp.tools.detect_changes import _change_visible

    vf = _make_change_visible_vf()
    ch = SchemaChange(
        change_type=ChangeType.REMOVED,
        target=ChangeTarget.TABLE,
        schema_name="private",
        table_name="salary",
    )
    assert not _change_visible(ch, connection_name="demo", vf=vf)


# ---------------------------------------------------------------------------
# hidden_table_types
# ---------------------------------------------------------------------------


def test_hidden_table_types_hides_views() -> None:
    from pretensor.visibility.config import VisibilityConfig

    cfg = VisibilityConfig(hidden_table_types=["view"])
    vf = VisibilityFilter.from_config(cfg)
    assert not vf.is_table_visible("c", "public", "v_orders", table_type="view")
    assert vf.is_table_visible("c", "public", "orders", table_type="table")


def test_hidden_table_types_case_insensitive() -> None:
    from pretensor.visibility.config import VisibilityConfig

    cfg = VisibilityConfig(hidden_table_types=["VIEW"])
    vf = VisibilityFilter.from_config(cfg)
    assert not vf.is_table_visible("c", "public", "v", table_type="view")
    assert not vf.is_table_visible("c", "public", "v", table_type="View")


def test_hidden_table_types_no_effect_without_table_type() -> None:
    """When table_type is not passed, hidden_table_types has no effect."""
    from pretensor.visibility.config import VisibilityConfig

    cfg = VisibilityConfig(hidden_table_types=["view"])
    vf = VisibilityFilter.from_config(cfg)
    assert vf.is_table_visible("c", "public", "v_orders")


def test_hidden_table_types_merged_from_profile() -> None:
    from pretensor.visibility.config import VisibilityConfig, VisibilityProfileRule

    base = VisibilityConfig(
        hidden_table_types=["materialized_view"],
        profiles={"strict": VisibilityProfileRule(hidden_table_types=["view"])},
    )
    merged = merge_profile_into_base(base, "strict")
    vf = VisibilityFilter.from_config(merged)
    assert not vf.is_table_visible("c", "s", "t", table_type="view")
    assert not vf.is_table_visible("c", "s", "t", table_type="materialized_view")
    assert vf.is_table_visible("c", "s", "t", table_type="table")


def test_hidden_table_types_yaml_loading(tmp_path: Path) -> None:
    p = tmp_path / "v.yml"
    p.write_text(
        "hidden_table_types:\n  - view\n  - materialized_view\n",
        encoding="utf-8",
    )
    cfg = load_visibility_config(p)
    assert cfg.hidden_table_types == ["view", "materialized_view"]
