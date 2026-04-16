"""Tests for snapshot store and graph patcher."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
    ViewDependency,
)
from pretensor.connectors.snapshot import ChangeTarget, ChangeType, diff_snapshots
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import column_node_id
from pretensor.core.store import KuzuStore
from pretensor.staleness.graph_patcher import GraphPatcher
from pretensor.staleness.snapshot_store import SnapshotStore


def _base_tables() -> list[Table]:
    return [
        Table(
            name="orders",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="user_id", data_type="int", is_foreign_key=True),
            ],
            foreign_keys=[
                ForeignKey(
                    source_schema="public",
                    source_table="orders",
                    source_column="user_id",
                    target_schema="public",
                    target_table="users",
                    target_column="id",
                )
            ],
        ),
        Table(
            name="users",
            schema_name="public",
            columns=[Column(name="id", data_type="int", is_primary_key=True)],
            foreign_keys=[],
        ),
    ]


def test_snapshot_store_roundtrip(tmp_path: Path) -> None:
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=_base_tables(),
        introspected_at=datetime.now(timezone.utc),
    )
    store = SnapshotStore(tmp_path)
    path = store.save("demo", snap)
    assert path.exists()
    loaded = store.load("demo")
    assert loaded is not None
    assert loaded.connection_name == "demo"
    assert len(loaded.tables) == 2


def test_graph_patcher_adds_column(tmp_path: Path) -> None:
    old = SchemaSnapshot(
        connection_name="conn",
        database="dbk",
        schemas=["public"],
        tables=_base_tables(),
        introspected_at=datetime.now(timezone.utc),
    )
    base = _base_tables()
    orders = base[0]
    users = base[1]
    orders_new = orders.model_copy(
        update={
            "columns": list(orders.columns)
            + [Column(name="extra_col", data_type="text", nullable=True)]
        }
    )
    new = old.model_copy(update={"tables": [orders_new, users]})

    graph = tmp_path / "g.kuzu"
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(old, store, run_relationship_discovery=False)
        changes = diff_snapshots(old, new)
        lineage_changes = [c for c in changes if c.target == ChangeTarget.LINEAGE]
        assert len(lineage_changes) == 0
        assert len(changes) == 1
        patcher = GraphPatcher(store)
        result = patcher.apply(changes, new, dry_run=False)
        assert result.columns_added == 1
        cid = column_node_id("conn", "public", "orders", "extra_col")
        rows = store.query_all_rows(
            "MATCH (c:SchemaColumn {node_id: $id}) RETURN c.column_name",
            {"id": cid},
        )
        assert rows and str(rows[0][0]) == "extra_col"
    finally:
        store.close()


def _view_dep(src_table: str, tgt_table: str, lineage_type: str = "VIEW") -> ViewDependency:
    return ViewDependency(
        source_schema="public",
        source_table=src_table,
        target_schema="public",
        target_table=tgt_table,
        lineage_type=lineage_type,
        object_name=f"public.{tgt_table}",
    )


def test_diff_snapshots_detects_lineage_added_and_removed() -> None:
    base = SchemaSnapshot(
        connection_name="c",
        database="d",
        schemas=["public"],
        tables=_base_tables(),
        introspected_at=datetime.now(timezone.utc),
        view_dependencies=[_view_dep("orders", "v_orders")],
    )
    # new snapshot: v_orders lineage removed, new v_users lineage added
    updated = base.model_copy(
        update={
            "view_dependencies": [_view_dep("users", "v_users")],
        }
    )
    changes = diff_snapshots(base, updated)
    lineage_changes = [c for c in changes if c.target == ChangeTarget.LINEAGE]
    assert len(lineage_changes) == 2

    added = [c for c in lineage_changes if c.change_type == ChangeType.ADDED]
    removed = [c for c in lineage_changes if c.change_type == ChangeType.REMOVED]
    assert len(added) == 1
    assert len(removed) == 1
    assert "v_users" in added[0].details
    assert "v_orders" in removed[0].details


def test_diff_snapshots_no_lineage_changes_when_identical() -> None:
    dep = _view_dep("orders", "v_orders")
    snap = SchemaSnapshot(
        connection_name="c",
        database="d",
        schemas=["public"],
        tables=_base_tables(),
        introspected_at=datetime.now(timezone.utc),
        view_dependencies=[dep],
    )
    changes = diff_snapshots(snap, snap)
    lineage_changes = [c for c in changes if c.target == ChangeTarget.LINEAGE]
    assert lineage_changes == []
