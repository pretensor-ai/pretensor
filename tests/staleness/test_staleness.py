"""Tests for snapshot store and graph patcher."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

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


def test_graph_patcher_classifies_added_table(tmp_path: Path) -> None:
    """A table added by a fast reindex gets a heuristic role, not role=NULL."""
    old = SchemaSnapshot(
        connection_name="conn",
        database="dbk",
        schemas=["public"],
        tables=_base_tables(),
        introspected_at=datetime.now(timezone.utc),
    )
    fct_sales = Table(
        name="fct_sales",
        schema_name="public",
        row_count=1000,
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="user_id", data_type="int", is_foreign_key=True),
            Column(name="amount", data_type="numeric"),
            Column(name="created_at", data_type="timestamp"),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="fct_sales",
                source_column="user_id",
                target_schema="public",
                target_table="users",
                target_column="id",
            )
        ],
    )
    new = old.model_copy(update={"tables": [*_base_tables(), fct_sales]})

    graph = tmp_path / "g.kuzu"
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(old, store, run_relationship_discovery=False)
        changes = diff_snapshots(old, new)
        added = [c for c in changes if c.change_type == ChangeType.ADDED]
        assert any(c.table_name == "fct_sales" for c in added)
        result = GraphPatcher(store).apply(changes, new, dry_run=False)
        assert result.tables_added == 1
        assert result.tables_classified == 1
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {table_name: 'fct_sales'})"
            " RETURN t.role, t.role_confidence",
            {},
        )
        assert rows and rows[0][0] is not None
        assert rows[0][1] is not None
    finally:
        store.close()


def _added_pair_snapshots() -> tuple[SchemaSnapshot, SchemaSnapshot]:
    """old → new adds fct_events (FK out to dim_channel) and dim_channel."""
    old = SchemaSnapshot(
        connection_name="conn",
        database="dbk",
        schemas=["public"],
        tables=_base_tables(),
        introspected_at=datetime.now(timezone.utc),
    )
    dim_channel = Table(
        name="dim_channel",
        schema_name="public",
        row_count=10,
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="name", data_type="text"),
        ],
    )
    fct_events = Table(
        name="fct_events",
        schema_name="public",
        row_count=1000,
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="channel_id", data_type="int", is_foreign_key=True),
            Column(name="amount", data_type="numeric"),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="fct_events",
                source_column="channel_id",
                target_schema="public",
                target_table="dim_channel",
                target_column="id",
            )
        ],
    )
    new = old.model_copy(update={"tables": [*_base_tables(), dim_channel, fct_events]})
    return old, new


def test_classify_added_tables_computes_fk_degrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FK in/out degrees are derived from the snapshot, per added table."""
    from pretensor.entities.classifier import TableClassifier

    old, new = _added_pair_snapshots()
    captured = []
    orig = TableClassifier.classify

    def _spy(self, table, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(table)
        return orig(self, table, **kwargs)

    monkeypatch.setattr(TableClassifier, "classify", _spy)

    graph = tmp_path / "g.kuzu"
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(old, store, run_relationship_discovery=False)
        changes = diff_snapshots(old, new)
        result = GraphPatcher(store).apply(changes, new, dry_run=False)
        assert result.tables_classified == 2
    finally:
        store.close()

    by_name = {t.name: t for t in captured}
    assert by_name["fct_events"].fk_out_degree == 1
    assert by_name["fct_events"].fk_in_degree == 0
    assert by_name["dim_channel"].fk_out_degree == 0
    assert by_name["dim_channel"].fk_in_degree == 1


def test_graph_patcher_dry_run_classifies_nothing(tmp_path: Path) -> None:
    """A dry run must not write roles (or any node) as a side effect."""
    old, new = _added_pair_snapshots()
    graph = tmp_path / "g.kuzu"
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(old, store, run_relationship_discovery=False)
        changes = diff_snapshots(old, new)
        result = GraphPatcher(store).apply(changes, new, dry_run=True)
        assert result.tables_classified == 0
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {table_name: 'fct_events'}) RETURN t.node_id",
            {},
        )
        assert rows == []
    finally:
        store.close()


def test_fast_path_classification_matches_full_recompute(tmp_path: Path) -> None:
    """Fast-path role/confidence must equal what --recompute-intelligence writes.

    The added table has two FK columns to the same target plus a dangling FK
    (target not in the snapshot) — exactly the cases where counting raw FK
    constraints instead of distinct present neighbors would diverge from the
    pipeline's count(DISTINCT neighbor) query.
    """
    from pretensor.core.ids import table_node_id
    from pretensor.intelligence.schema_classification import (
        classify_database_tables,
    )

    old = SchemaSnapshot(
        connection_name="conn",
        database="dbk",
        schemas=["public"],
        tables=_base_tables(),
        introspected_at=datetime.now(timezone.utc),
    )
    fct_multi = Table(
        name="fct_multi",
        schema_name="public",
        row_count=500,
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="buyer_id", data_type="int", is_foreign_key=True),
            Column(name="seller_id", data_type="int", is_foreign_key=True),
            Column(name="ghost_id", data_type="int", is_foreign_key=True),
            Column(name="amount", data_type="numeric"),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="fct_multi",
                source_column="buyer_id",
                target_schema="public",
                target_table="users",
                target_column="id",
            ),
            ForeignKey(
                source_schema="public",
                source_table="fct_multi",
                source_column="seller_id",
                target_schema="public",
                target_table="users",
                target_column="id",
            ),
            ForeignKey(
                source_schema="public",
                source_table="fct_multi",
                source_column="ghost_id",
                target_schema="public",
                target_table="not_in_snapshot",
                target_column="id",
            ),
        ],
    )
    new = old.model_copy(update={"tables": [*_base_tables(), fct_multi]})

    graph = tmp_path / "g.kuzu"
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(old, store, run_relationship_discovery=False)
        changes = diff_snapshots(old, new)
        GraphPatcher(store).apply(changes, new, dry_run=False)
        tid = table_node_id("conn", "public", "fct_multi")
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {node_id: $tid}) RETURN t.role, t.role_confidence",
            {"tid": tid},
        )
        assert rows and rows[0][0] is not None
        fast_role, fast_conf = str(rows[0][0]), float(rows[0][1])

        full = classify_database_tables(store, "dbk")[tid]
        assert fast_role == str(full.role)
        assert fast_conf == pytest.approx(full.confidence)
    finally:
        store.close()


def test_impact_summary_names_recompute_intelligence_flag(tmp_path: Path) -> None:
    """The advice string must name the flag that actually fixes staleness."""
    from pretensor.connectors.snapshot import SchemaChange
    from pretensor.staleness.impact_analyzer import ImpactAnalyzer

    change = SchemaChange(
        change_type=ChangeType.ADDED,
        target=ChangeTarget.TABLE,
        table_name="fct_events",
        schema_name="public",
    )
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        report = ImpactAnalyzer(store).analyze(
            [change], connection_name="conn", database_key="dbk"
        )
        assert "--recompute-intelligence" in report.summary
    finally:
        store.close()


def _view_dep(
    src_table: str, tgt_table: str, lineage_type: str = "VIEW"
) -> ViewDependency:
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


def _snapshot(tables: list[Table]) -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name="conn",
        database="dbk",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )


def test_diff_snapshots_ignores_volatile_table_stats() -> None:
    """Query activity (pg_stat counters, row counts, size) is not schema drift."""
    old = _snapshot(_base_tables())
    tables = _base_tables()
    tables[0] = tables[0].model_copy(
        update={
            "row_count": 4200,
            "seq_scan_count": 40,
            "idx_scan_count": 17,
            "insert_count": 100,
            "update_count": 5,
            "delete_count": 1,
            "table_bytes": 8_192_000,
        }
    )
    new = old.model_copy(update={"tables": tables})
    assert diff_snapshots(old, new) == []


def test_diff_snapshots_ignores_volatile_column_stats() -> None:
    """Planner stats rewritten by ANALYZE/autovacuum are not schema drift."""
    old = _snapshot(_base_tables())
    tables = _base_tables()
    orders = tables[0]
    cols = [
        orders.columns[0].model_copy(
            update={
                "most_common_values": ["1", "2"],
                "histogram_bounds": ["0", "50", "100"],
                "stats_correlation": 0.97,
            }
        ),
        *orders.columns[1:],
    ]
    tables[0] = orders.model_copy(update={"columns": cols})
    new = old.model_copy(update={"tables": tables})
    assert diff_snapshots(old, new) == []


def test_diff_snapshots_detects_structural_table_change() -> None:
    """Positive control: DDL-level fields still produce MODIFIED/TABLE."""
    old = _snapshot(_base_tables())
    tables = _base_tables()
    tables[0] = tables[0].model_copy(update={"comment": "orders fact table"})
    new = old.model_copy(update={"tables": tables})
    changes = diff_snapshots(old, new)
    assert len(changes) == 1
    assert changes[0].change_type == ChangeType.MODIFIED
    assert changes[0].target == ChangeTarget.TABLE
    assert "comment" in changes[0].details


def test_volatile_field_lists_stay_in_sync() -> None:
    """The diff-exclusion set and the graph-refresh surface must not drift.

    ``_refresh_volatile_stats`` builds kwargs directly from
    ``VOLATILE_TABLE_FIELDS``, so a field added to the constant without a
    matching parameter on ``update_table_volatile_stats`` must fail loudly
    here (and at patch time) instead of silently leaking into schema diffs
    or silently never being refreshed.
    """
    import inspect as inspect_mod

    from pretensor.connectors.snapshot import (
        _STRUCTURAL_COLUMN_FIELDS,
        _STRUCTURAL_TABLE_FIELDS,
        VOLATILE_COLUMN_FIELDS,
        VOLATILE_TABLE_FIELDS,
    )
    from pretensor.core.graph_store import GraphStore

    params = set(
        inspect_mod.signature(GraphStore.update_table_volatile_stats).parameters
    ) - {"self", "table_node_id"}
    assert params == set(VOLATILE_TABLE_FIELDS)

    # Every volatile field is a real model field (getattr in the patcher works),
    # and the structural/volatile partitions never overlap.
    tbl = _base_tables()[0]
    for f in VOLATILE_TABLE_FIELDS:
        getattr(tbl, f)
    col = tbl.columns[0]
    for f in VOLATILE_COLUMN_FIELDS:
        getattr(col, f)
    assert not set(VOLATILE_TABLE_FIELDS) & set(_STRUCTURAL_TABLE_FIELDS)
    assert not set(VOLATILE_COLUMN_FIELDS) & set(_STRUCTURAL_COLUMN_FIELDS)


def test_update_table_volatile_stats_missing_node_is_noop(tmp_path: Path) -> None:
    """A refresh against a node that doesn't exist returns False and writes nothing."""
    from pretensor.connectors.snapshot import VOLATILE_TABLE_FIELDS

    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        store.ensure_schema()
        stats = {f: None for f in VOLATILE_TABLE_FIELDS}
        assert store.update_table_volatile_stats("no-such-node", **stats) is False
    finally:
        store.close()


def test_graph_patcher_dry_run_skips_volatile_refresh(tmp_path: Path) -> None:
    """dry_run must not touch the graph: counter stays 0, values unchanged."""
    old = _snapshot(_base_tables())
    tables = _base_tables()
    tables[0] = tables[0].model_copy(update={"row_count": 4200})
    new = old.model_copy(update={"tables": tables})

    graph = tmp_path / "g.kuzu"
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(old, store, run_relationship_discovery=False)
        result = GraphPatcher(store).apply([], new, dry_run=True)
        assert result.table_stats_refreshed == 0
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {table_name: 'orders'}) RETURN t.row_count",
            {},
        )
        assert rows and rows[0][0] != 4200
    finally:
        store.close()


def test_graph_patcher_refreshes_volatile_stats(tmp_path: Path) -> None:
    """Volatile stats reach the graph on reindex even with an empty diff."""
    old = _snapshot(_base_tables())
    tables = _base_tables()
    tables[0] = tables[0].model_copy(update={"row_count": 4200, "seq_scan_count": 40})
    new = old.model_copy(update={"tables": tables})

    graph = tmp_path / "g.kuzu"
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(old, store, run_relationship_discovery=False)
        changes = diff_snapshots(old, new)
        assert changes == []
        result = GraphPatcher(store).apply(changes, new, dry_run=False)
        assert result.table_stats_refreshed == 2
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {table_name: 'orders'})"
            " RETURN t.row_count, t.seq_scan_count",
            {},
        )
        assert rows and int(rows[0][0]) == 4200
        assert int(rows[0][1]) == 40
    finally:
        store.close()
