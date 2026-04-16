"""Unit tests for intelligence/cluster_labeler.py — heuristic labeling."""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path

from pretensor.connectors.models import Column, ForeignKey, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.intelligence.cluster_labeler import (
    HEURISTIC_CLUSTER_DESCRIPTION,
    ClusterLabeler,
    LabeledCluster,
)
from pretensor.intelligence.clustering import Cluster


def _ts() -> datetime.datetime:
    return datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)


def _build_store(tmp_path: Path, n_tables: int = 3) -> tuple[KuzuStore, list[str]]:
    """Build a graph with n_tables and return (store, list_of_node_ids)."""
    tables = [
        Table(
            name=f"table{i}",
            schema_name="public",
            columns=[Column(name="id", data_type="int", is_primary_key=True)],
        )
        for i in range(n_tables)
    ]
    snap = SchemaSnapshot(
        connection_name="demo",
        database="db",
        schemas=["public"],
        tables=tables,
        introspected_at=_ts(),
    )
    store = KuzuStore(tmp_path / "cl.kuzu")
    GraphBuilder().build(snap, store, run_relationship_discovery=False)

    node_ids = [table_node_id("demo", "public", f"table{i}") for i in range(n_tables)]
    return store, node_ids


def _build_hr_store(tmp_path: Path) -> tuple[KuzuStore, list[str]]:
    """HR-shaped cluster where Employee is the central node by FK degree."""
    cols = [Column(name="id", data_type="int", is_primary_key=True)]
    tables = [
        Table(name="Employee", schema_name="HumanResources", columns=cols),
        Table(
            name="EmployeeDepartmentHistory",
            schema_name="HumanResources",
            columns=cols,
            foreign_keys=[
                ForeignKey(
                    source_schema="HumanResources",
                    source_table="EmployeeDepartmentHistory",
                    source_column="id",
                    target_schema="HumanResources",
                    target_table="Employee",
                    target_column="id",
                ),
                ForeignKey(
                    source_schema="HumanResources",
                    source_table="EmployeeDepartmentHistory",
                    source_column="id",
                    target_schema="HumanResources",
                    target_table="Department",
                    target_column="id",
                ),
            ],
        ),
        Table(
            name="EmployeePayHistory",
            schema_name="HumanResources",
            columns=cols,
            foreign_keys=[
                ForeignKey(
                    source_schema="HumanResources",
                    source_table="EmployeePayHistory",
                    source_column="id",
                    target_schema="HumanResources",
                    target_table="Employee",
                    target_column="id",
                ),
            ],
        ),
        Table(name="Department", schema_name="HumanResources", columns=cols),
    ]
    snap = SchemaSnapshot(
        connection_name="demo",
        database="db",
        schemas=["HumanResources"],
        tables=tables,
        introspected_at=_ts(),
    )
    store = KuzuStore(tmp_path / "hr.kuzu")
    GraphBuilder().build(snap, store, run_relationship_discovery=False)
    node_ids = [
        table_node_id("demo", "HumanResources", name)
        for name in (
            "Employee",
            "EmployeeDepartmentHistory",
            "EmployeePayHistory",
            "Department",
        )
    ]
    return store, node_ids


def _clusters(node_ids: list[str]) -> list[Cluster]:
    return [Cluster(table_ids=[tid], cohesion_score=0.0) for tid in node_ids]


def test_heuristic_label_no_llm(tmp_path: Path) -> None:
    store, node_ids = _build_store(tmp_path, 2)
    try:
        labeler = ClusterLabeler(store, llm_client=None)
        results = asyncio.run(labeler.label_and_persist(_clusters(node_ids), "db"))
    finally:
        store.close()

    assert len(results) == 2
    assert all(HEURISTIC_CLUSTER_DESCRIPTION in r.description for r in results)


def test_heuristic_labels_are_labeled_clusters(tmp_path: Path) -> None:
    store, node_ids = _build_store(tmp_path, 2)
    try:
        labeler = ClusterLabeler(store)
        results = asyncio.run(labeler.label_and_persist(_clusters(node_ids), "db"))
    finally:
        store.close()

    assert all(isinstance(r, LabeledCluster) for r in results)
    assert all(r.label for r in results)
    assert all(r.cluster_id.startswith("db::cluster::") for r in results)


def test_empty_cluster_list(tmp_path: Path) -> None:
    store, _ = _build_store(tmp_path, 1)
    try:
        labeler = ClusterLabeler(store, llm_client=None)
        results = asyncio.run(labeler.label_and_persist([], "db"))
    finally:
        store.close()

    assert results == []


def test_batch_size_multiple_batches(tmp_path: Path) -> None:
    """Six clusters produce two batches (batch_size=5) — all heuristic."""
    store, node_ids = _build_store(tmp_path, 6)
    try:
        labeler = ClusterLabeler(store)
        results = asyncio.run(labeler.label_and_persist(_clusters(node_ids), "db"))
    finally:
        store.close()

    assert len(results) == 6
    assert all(HEURISTIC_CLUSTER_DESCRIPTION in r.description for r in results)


def test_heuristic_label_picks_central_node_and_frequent_noun(tmp_path: Path) -> None:
    """HR cluster: Employee has highest in-cluster FK degree; 'employee' is frequent noun."""
    store, node_ids = _build_hr_store(tmp_path)
    try:
        labeler = ClusterLabeler(store, llm_client=None)
        # Single cluster spanning all four HR tables. Order intentionally NOT
        # central-first, to prove the heuristic is no longer order-dependent.
        cluster = Cluster(
            table_ids=[node_ids[3], node_ids[1], node_ids[2], node_ids[0]],
            cohesion_score=0.5,
        )
        results = asyncio.run(labeler.label_and_persist([cluster], "db"))
    finally:
        store.close()

    assert len(results) == 1
    assert results[0].label == "HumanResources.Employee · Employee cluster (4 tables)"


def test_heuristic_label_is_deterministic(tmp_path: Path) -> None:
    """Same graph + cluster → identical labels across runs and permutations."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    store_a, ids_a = _build_hr_store(dir_a)
    store_b, ids_b = _build_hr_store(dir_b)
    try:
        cluster_a = Cluster(table_ids=list(ids_a), cohesion_score=0.5)
        cluster_b = Cluster(table_ids=list(reversed(ids_b)), cohesion_score=0.5)
        labels_a = asyncio.run(ClusterLabeler(store_a).label_and_persist([cluster_a], "db"))
        labels_b = asyncio.run(ClusterLabeler(store_b).label_and_persist([cluster_b], "db"))
    finally:
        store_a.close()
        store_b.close()

    assert labels_a[0].label == labels_b[0].label


def test_heuristic_label_fallback_when_missing_nodes(tmp_path: Path) -> None:
    store, _ = _build_store(tmp_path, 1)
    try:
        labeler = ClusterLabeler(store, llm_client=None)
        cluster = Cluster(table_ids=["Table:db:public:ghost"], cohesion_score=0.0)
        results = asyncio.run(labeler.label_and_persist([cluster], "db"))
    finally:
        store.close()

    assert results[0].label == "Schema domain"


def test_frequent_noun_skips_suffix_denylist_hub_tokens() -> None:
    """``*entity``/``*history``/``*type`` stems must not beat real head nouns.

    Regression: AW ``person.person`` cluster had ``Businessentity`` winning the
    weighted noun vote over ``Person`` because ``entity`` appeared in multiple
    hub tables. The suffix deny now rejects those candidates entirely, so
    ``person`` — the real semantic head — surfaces even when it has slightly
    less degree than the hub.
    """
    from pretensor.intelligence.cluster_labeler import _frequent_noun

    # Person appears 3x with modest weight; Businessentity appears 2x with
    # higher weight per row (mimicking the AW hub). Pre-fix, Businessentity
    # would win the weighted total; post-fix it's filtered out.
    entries: list[tuple[str, str, float]] = [
        ("person", "Person", 1.0),
        ("person", "PersonPhone", 1.0),
        ("person", "PersonCreditCard", 1.0),
        ("person", "BusinessEntity", 2.5),
        ("person", "BusinessEntityContact", 2.5),
        ("person", "AddressType", 2.0),
        ("person", "CustomerHistory", 2.0),
    ]
    noun = _frequent_noun(entries, schema_stopwords={"person"})
    assert noun is not None
    # Not Businessentity, Addresstype, Customerhistory — all suffix-denied.
    assert noun.lower() not in {
        "businessentity",
        "addresstype",
        "customerhistory",
        "entity",
        "type",
        "history",
    }


def _build_hub_store(tmp_path: Path) -> tuple[KuzuStore, list[str]]:
    """Hub cluster where a bridge-like hub has slightly higher FK degree than the dimension.

    Shape (mirrors AW ``person.person``):
      - ``person`` (dimension): in-degree = 2 (``address`` → person, ``phone`` → person)
      - ``businessentity`` (bridge): in-degree = 3 (three spoke tables point at it)
      - Spokes point at both, so without role weighting ``businessentity`` wins.
    """
    cols = [Column(name="id", data_type="int", is_primary_key=True)]
    tables = [
        Table(name="person", schema_name="person", columns=cols),
        Table(name="businessentity", schema_name="person", columns=cols),
        Table(
            name="address",
            schema_name="person",
            columns=cols,
            foreign_keys=[
                ForeignKey(
                    source_schema="person",
                    source_table="address",
                    source_column="id",
                    target_schema="person",
                    target_table="person",
                    target_column="id",
                ),
                ForeignKey(
                    source_schema="person",
                    source_table="address",
                    source_column="id",
                    target_schema="person",
                    target_table="businessentity",
                    target_column="id",
                ),
            ],
        ),
        Table(
            name="phone",
            schema_name="person",
            columns=cols,
            foreign_keys=[
                ForeignKey(
                    source_schema="person",
                    source_table="phone",
                    source_column="id",
                    target_schema="person",
                    target_table="person",
                    target_column="id",
                ),
                ForeignKey(
                    source_schema="person",
                    source_table="phone",
                    source_column="id",
                    target_schema="person",
                    target_table="businessentity",
                    target_column="id",
                ),
            ],
        ),
        Table(
            name="contact",
            schema_name="person",
            columns=cols,
            foreign_keys=[
                ForeignKey(
                    source_schema="person",
                    source_table="contact",
                    source_column="id",
                    target_schema="person",
                    target_table="businessentity",
                    target_column="id",
                ),
            ],
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="demo",
        database="db",
        schemas=["person"],
        tables=tables,
        introspected_at=_ts(),
    )
    store = KuzuStore(tmp_path / "hub.kuzu")
    GraphBuilder().build(snap, store, run_relationship_discovery=False)
    ids = [
        table_node_id("demo", "person", name)
        for name in ("person", "businessentity", "address", "phone", "contact")
    ]
    return store, ids


def test_role_weight_prefers_dimension_over_bridge(tmp_path: Path) -> None:
    """Dimension ``person`` wins the head-noun vote over bridge ``businessentity``.

    Pre-fix the labeler ranked by FK degree alone, so the 3-degree bridge
    ``businessentity`` beat the 2-degree dimension ``person``. Post-fix the
    role weight (dimension 1.5×, bridge 0.4×) pushes ``person`` to win and
    also swings the anchor tiebreak when degrees tie.
    """
    store, ids = _build_hub_store(tmp_path)
    try:
        # Mark roles: dimension for `person`, bridge for `businessentity`.
        store.set_table_classification(
            ids[0], role="dimension", role_confidence=0.9, classification_signals_json="[]"
        )
        store.set_table_classification(
            ids[1], role="bridge", role_confidence=0.9, classification_signals_json="[]"
        )
        labeler = ClusterLabeler(store, llm_client=None)
        cluster = Cluster(table_ids=list(ids), cohesion_score=0.5)
        results = asyncio.run(labeler.label_and_persist([cluster], "db"))
    finally:
        store.close()

    assert len(results) == 1
    label = results[0].label
    # Head noun must be ``Person`` — the dimension — not ``Businessentity``
    # (also prevented by the suffix deny, but we assert the stronger property
    # that the dimension won the anchor tiebreak too).
    assert "Person cluster" in label
    assert "businessentity" not in label.lower()


def test_anchor_tiebreak_prefers_dimension_on_equal_degree(tmp_path: Path) -> None:
    """Equal raw FK degree → dimension beats bridge as the anchor table.

    Mutual FKs give each table degree 2 (1 in, 1 out). Role weight lifts the
    dimension above the bridge in the effective-degree sort.
    """
    cols = [Column(name="id", data_type="int", is_primary_key=True)]
    tables = [
        Table(
            name="product",
            schema_name="sales",
            columns=cols,
            foreign_keys=[
                ForeignKey(
                    source_schema="sales",
                    source_table="product",
                    source_column="id",
                    target_schema="sales",
                    target_table="productbridge",
                    target_column="id",
                ),
            ],
        ),
        Table(
            name="productbridge",
            schema_name="sales",
            columns=cols,
            foreign_keys=[
                ForeignKey(
                    source_schema="sales",
                    source_table="productbridge",
                    source_column="id",
                    target_schema="sales",
                    target_table="product",
                    target_column="id",
                ),
            ],
        ),
    ]
    snap = SchemaSnapshot(
        connection_name="demo",
        database="db",
        schemas=["sales"],
        tables=tables,
        introspected_at=_ts(),
    )
    store = KuzuStore(tmp_path / "tie.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        ids = [
            table_node_id("demo", "sales", name)
            for name in ("product", "productbridge")
        ]
        store.set_table_classification(
            ids[0], role="dimension", role_confidence=0.9, classification_signals_json="[]"
        )
        store.set_table_classification(
            ids[1], role="bridge", role_confidence=0.9, classification_signals_json="[]"
        )
        labeler = ClusterLabeler(store, llm_client=None)
        cluster = Cluster(table_ids=list(ids), cohesion_score=0.5)
        results = asyncio.run(labeler.label_and_persist([cluster], "db"))
    finally:
        store.close()

    # Anchor is the dimension, not the bridge, even though raw degree ties.
    assert "sales.product ·" in results[0].label
    assert "productbridge" not in results[0].label.lower()


def test_schema_pattern_passed_through(tmp_path: Path) -> None:
    store, node_ids = _build_store(tmp_path, 2)
    try:
        labeler = ClusterLabeler(store)
        results = asyncio.run(
            labeler.label_and_persist(
                _clusters(node_ids),
                "db",
                cluster_schema_patterns={0: "public", 1: "analytics"},
            )
        )
    finally:
        store.close()

    assert len(results) == 2
