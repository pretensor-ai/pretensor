"""Tests for MetricSqlCompiler (YAML → validated SQL)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

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
from pretensor.semantic.compiler import MetricCompileError, MetricSqlCompiler


def _snapshot(*, extra_tables: list[Table] | None = None) -> SchemaSnapshot:
    t_orders = Table(
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
    )
    t_users = Table(
        name="users",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    tables = [t_orders, t_users]
    if extra_tables:
        tables.extend(extra_tables)
    return SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=tables,
        introspected_at=datetime.now(timezone.utc),
    )


def _layer(*, metrics: list[Metric]) -> SemanticLayerModel:
    entity = Entity(
        name="orders",
        description="Orders",
        source_table="public.orders",
        attributes=[
            Attribute(
                name="id",
                description="pk",
                role=AttributeRole.IDENTIFIER,
                source_column="id",
            ),
            Attribute(
                name="amount",
                description="measure",
                role=AttributeRole.MEASURE,
                source_column="amount",
            ),
        ],
        metrics=metrics,
    )
    return SemanticLayerModel(
        connection_name="demo",
        domains=[Domain(name="sales", description="sales", entities=[entity])],
    )


def _build_store(tmp_path) -> KuzuStore:
    store = KuzuStore(tmp_path / "g.kuzu")
    GraphBuilder().build(_snapshot(), store, run_relationship_discovery=False)
    return store


def test_compile_sum_metric(tmp_path) -> None:
    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="total_revenue",
                    description="sum amount",
                    type=MetricType.SUM,
                    field="amount",
                )
            ]
        )
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        compiled = compiler.compile(layer, "total_revenue")
        assert compiled.metric == "total_revenue"
        assert compiled.entity == "orders"
        assert 'SUM("amount")' in compiled.sql
        assert '"public"."orders"' in compiled.sql
        assert compiled.validation.valid
    finally:
        store.close()


def test_compile_count_distinct_metric(tmp_path) -> None:
    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="order_count",
                    description="distinct ids",
                    type=MetricType.COUNT_DISTINCT,
                    field="id",
                )
            ]
        )
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        compiled = compiler.compile(layer, "order_count")
        assert 'COUNT(DISTINCT "id")' in compiled.sql
        assert compiled.validation.valid
    finally:
        store.close()


def test_compile_average_metric_with_filters(tmp_path) -> None:
    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="avg_amt",
                    description="average",
                    type=MetricType.AVERAGE,
                    field="amount",
                    filters=['"amount" > 0'],
                )
            ]
        )
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        compiled = compiler.compile(layer, "avg_amt")
        assert 'AVG("amount")' in compiled.sql
        assert "WHERE" in compiled.sql
        assert '"amount" > 0' in compiled.sql
    finally:
        store.close()


def test_compile_unknown_field_raises(tmp_path) -> None:
    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="bad",
                    description="nope",
                    type=MetricType.SUM,
                    field="no_such_column",
                )
            ]
        )
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        with pytest.raises(MetricCompileError, match="no_such_column"):
            compiler.compile(layer, "bad")
    finally:
        store.close()


def test_compile_missing_metric_raises(tmp_path) -> None:
    store = _build_store(tmp_path)
    try:
        layer = _layer(metrics=[])
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        with pytest.raises(MetricCompileError, match="not found"):
            compiler.compile(layer, "ghost")
    finally:
        store.close()


def test_compile_aggregate_missing_field_raises(tmp_path) -> None:
    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="bad",
                    description="no field",
                    type=MetricType.SUM,
                    field=None,
                )
            ]
        )
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        with pytest.raises(MetricCompileError, match="requires `field`"):
            compiler.compile(layer, "bad")
    finally:
        store.close()


def test_compile_derived_valid_join(tmp_path) -> None:
    store = _build_store(tmp_path)
    try:
        expr = (
            "SELECT AVG(o.amount) AS avg_ticket "
            "FROM public.orders o "
            "JOIN public.users u ON o.user_id = u.id"
        )
        layer = _layer(
            metrics=[
                Metric(
                    name="avg_ticket",
                    description="derived",
                    type=MetricType.DERIVED,
                    expression=expr,
                )
            ]
        )
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        compiled = compiler.compile(layer, "avg_ticket")
        assert compiled.warnings == []  # orders→users FK exists
        assert "AVG(o.amount)" in compiled.sql
        assert compiled.validation.valid
    finally:
        store.close()


def test_compile_derived_unknown_table_raises(tmp_path) -> None:
    store = _build_store(tmp_path)
    try:
        expr = "SELECT COUNT(*) FROM public.missing_table"
        layer = _layer(
            metrics=[
                Metric(
                    name="bad_derived",
                    description="bad",
                    type=MetricType.DERIVED,
                    expression=expr,
                )
            ]
        )
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        with pytest.raises(MetricCompileError, match="missing_table"):
            compiler.compile(layer, "bad_derived")
    finally:
        store.close()


def test_compile_derived_warns_on_graphless_join(tmp_path) -> None:
    t_products = Table(
        name="products",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    store = KuzuStore(tmp_path / "g.kuzu")
    try:
        GraphBuilder().build(
            _snapshot(extra_tables=[t_products]),
            store,
            run_relationship_discovery=False,
        )
        expr = (
            "SELECT COUNT(*) FROM public.orders o "
            "JOIN public.products p ON o.id = p.id"
        )
        layer = _layer(
            metrics=[
                Metric(
                    name="cross_join",
                    description="weird",
                    type=MetricType.DERIVED,
                    expression=expr,
                )
            ]
        )
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        compiled = compiler.compile(layer, "cross_join")
        assert compiled.warnings
        assert any("products" in w for w in compiled.warnings)
    finally:
        store.close()


def test_compile_derived_missing_expression_raises(tmp_path) -> None:
    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="empty_derived",
                    description="nope",
                    type=MetricType.DERIVED,
                )
            ]
        )
        compiler = MetricSqlCompiler(
            store, connection_name="demo", database_key="demo"
        )
        with pytest.raises(MetricCompileError, match="requires `expression`"):
            compiler.compile(layer, "empty_derived")
    finally:
        store.close()
