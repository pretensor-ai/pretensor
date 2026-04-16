"""Tests for YamlSemanticLayer (OSS SemanticLayer impl)."""

from __future__ import annotations

from datetime import datetime, timezone

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.introspection.models.semantic import (
    Attribute,
    AttributeRole,
    Domain,
    Entity,
    Metric,
    MetricType,
)
from pretensor.introspection.models.semantic import (
    SemanticLayer as SemanticLayerModel,
)
from pretensor.semantic import SemanticLayer, YamlSemanticLayer


def _snapshot() -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[
            Table(
                name="orders",
                schema_name="public",
                columns=[
                    Column(name="id", data_type="int", is_primary_key=True),
                    Column(name="user_id", data_type="int", is_foreign_key=True),
                    Column(name="amount", data_type="numeric"),
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
                columns=[
                    Column(name="id", data_type="int", is_primary_key=True)
                ],
                foreign_keys=[],
            ),
        ],
        introspected_at=datetime.now(timezone.utc),
    )


def _layer_model() -> SemanticLayerModel:
    orders = Entity(
        name="orders",
        description="orders",
        source_table="public.orders",
        attributes=[
            Attribute(
                name="id",
                description="pk",
                role=AttributeRole.IDENTIFIER,
                source_column="id",
            ),
            Attribute(
                name="created_at",
                description="time",
                role=AttributeRole.TIME_DIMENSION,
                source_column="created_at",
            ),
            Attribute(
                name="status",
                description="dim",
                role=AttributeRole.DIMENSION,
                source_column="status",
            ),
            Attribute(
                name="amount",
                description="measure",
                role=AttributeRole.MEASURE,
                source_column="amount",
            ),
        ],
        metrics=[
            Metric(
                name="total_revenue",
                description="sum amount",
                type=MetricType.SUM,
                field="amount",
            )
        ],
    )
    return SemanticLayerModel(
        connection_name="demo",
        domains=[Domain(name="sales", description="sales", entities=[orders])],
    )


def _build_layer(tmp_path) -> YamlSemanticLayer:
    store = KuzuStore(tmp_path / "g.kuzu")
    GraphBuilder().build(_snapshot(), store, run_relationship_discovery=False)
    return YamlSemanticLayer(_layer_model(), store=store, database_key="demo")


def test_is_semantic_layer(tmp_path) -> None:
    layer = _build_layer(tmp_path)
    try:
        assert isinstance(layer, SemanticLayer)
    finally:
        layer._store.close()


def test_get_entity_returns_dump(tmp_path) -> None:
    layer = _build_layer(tmp_path)
    try:
        entity = layer.get_entity("orders")
        assert entity is not None
        assert entity["name"] == "orders"
        assert entity["source_table"] == "public.orders"
        assert layer.get_entity("missing") is None
    finally:
        layer._store.close()


def test_get_metric_includes_entity(tmp_path) -> None:
    layer = _build_layer(tmp_path)
    try:
        metric = layer.get_metric("total_revenue")
        assert metric is not None
        assert metric["entity"] == "orders"
        assert metric["source_table"] == "public.orders"
        assert layer.get_metric("missing") is None
    finally:
        layer._store.close()


def test_get_dimensions_filters_measures(tmp_path) -> None:
    layer = _build_layer(tmp_path)
    try:
        dims = layer.get_dimensions("orders")
        names = {d["name"] for d in dims}
        assert "status" in names
        assert "created_at" in names
        assert "id" in names
        assert "amount" not in names  # measure excluded
        assert layer.get_dimensions("missing") == []
    finally:
        layer._store.close()


def test_get_rules_empty(tmp_path) -> None:
    layer = _build_layer(tmp_path)
    try:
        assert layer.get_rules("orders") == []
    finally:
        layer._store.close()


def test_validate_query_delegates(tmp_path) -> None:
    layer = _build_layer(tmp_path)
    try:
        result = layer.validate_query(
            "SELECT o.id FROM public.orders o "
            "JOIN public.users u ON o.user_id = u.id",
            connection_name="demo",
        )
        assert result["valid"] is True
        assert result["errors"] == []

        bad = layer.validate_query(
            "SELECT * FROM public.nope",
            connection_name="demo",
        )
        assert bad["valid"] is False
        assert any("missing_table" in e for e in bad["errors"])
    finally:
        layer._store.close()


def test_validate_query_mismatched_connection(tmp_path) -> None:
    layer = _build_layer(tmp_path)
    try:
        result = layer.validate_query("SELECT 1", connection_name="other")
        assert result["valid"] is False
        assert any("does not match" in e for e in result["errors"])
    finally:
        layer._store.close()


def test_impact_semantic_finds_owning_metric(tmp_path) -> None:
    layer = _build_layer(tmp_path)
    store = layer._store
    try:
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {schema_name: 'public', table_name: 'orders'}) "
            "RETURN t.node_id LIMIT 1"
        )
        assert rows
        node_id = str(rows[0][0])
        impact = layer.impact_semantic(node_id)
        assert any(
            item["metric"] == "total_revenue"
            and item["reason"] == "owns_source_table"
            for item in impact
        )
    finally:
        store.close()


def test_impact_semantic_missing_node(tmp_path) -> None:
    layer = _build_layer(tmp_path)
    try:
        assert layer.impact_semantic("schema_table::does_not_exist") == []
    finally:
        layer._store.close()
