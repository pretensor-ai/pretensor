"""GraphBuilder writes LINEAGE edges from snapshot.view_dependencies."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, SchemaSnapshot, Table, ViewDependency
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore


def test_builder_merges_duplicate_lineage_by_edge_id(tmp_path: Path) -> None:
    tables = [
        Table(
            name="src",
            schema_name="public",
            columns=[Column(name="id", data_type="int")],
        ),
        Table(
            name="dst",
            schema_name="public",
            columns=[Column(name="id", data_type="int")],
        ),
    ]
    # Same logical edge from two "sources" — builder keeps higher confidence
    deps = [
        ViewDependency(
            source_schema="public",
            source_table="src",
            target_schema="public",
            target_table="dst",
            lineage_type="VIEW",
            object_name="public.dst",
            confidence=0.5,
        ),
        ViewDependency(
            source_schema="public",
            source_table="src",
            target_schema="public",
            target_table="dst",
            lineage_type="VIEW",
            object_name="public.dst:alt",
            confidence=1.0,
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="c",
        database="d",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
        view_dependencies=deps,
    )
    graph = tmp_path / "t.kuzu"
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        rows = store.query_all_rows(
            "MATCH ()-[r:LINEAGE]->() RETURN r.confidence, r.source ORDER BY r.confidence DESC"
        )
        assert len(rows) == 1
        assert float(rows[0][0]) == 1.0
    finally:
        store.close()
