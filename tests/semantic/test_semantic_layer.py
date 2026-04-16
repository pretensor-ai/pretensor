"""Tests for SemanticLayer ABC and NullSemanticLayer."""

from __future__ import annotations

import pytest

from pretensor.semantic import NullSemanticLayer, SemanticLayer

# ── Instantiation ─────────────────────────────────────────────────────────────


def test_null_semantic_layer_is_semantic_layer() -> None:
    layer = NullSemanticLayer()
    assert isinstance(layer, SemanticLayer)


def test_semantic_layer_is_abstract() -> None:
    with pytest.raises(TypeError):
        SemanticLayer()  # type: ignore[abstract]


# ── NullSemanticLayer return values ───────────────────────────────────────────


def test_get_entity_returns_none() -> None:
    layer = NullSemanticLayer()
    assert layer.get_entity("customers") is None


def test_get_metric_returns_none() -> None:
    layer = NullSemanticLayer()
    assert layer.get_metric("monthly_revenue") is None


def test_get_dimensions_returns_empty_list() -> None:
    layer = NullSemanticLayer()
    result = layer.get_dimensions("customers")
    assert result == []
    assert isinstance(result, list)


def test_get_rules_returns_empty_list() -> None:
    layer = NullSemanticLayer()
    result = layer.get_rules("orders")
    assert result == []
    assert isinstance(result, list)


def test_validate_query_returns_valid() -> None:
    layer = NullSemanticLayer()
    result = layer.validate_query("SELECT 1", connection_name="demo")
    assert result["valid"] is True
    assert result["errors"] == []


def test_impact_semantic_returns_empty_list() -> None:
    layer = NullSemanticLayer()
    result = layer.impact_semantic("table::public.orders")
    assert result == []
    assert isinstance(result, list)


# ── Subclass compliance ───────────────────────────────────────────────────────


def test_incomplete_subclass_raises_type_error() -> None:
    """A subclass missing any abstract method must not be instantiable."""

    class PartialLayer(SemanticLayer):
        def get_entity(self, entity_id: str):  # type: ignore[override]
            return None

        # Missing: get_metric, get_dimensions, get_rules, validate_query, impact_semantic

    with pytest.raises(TypeError):
        PartialLayer()  # type: ignore[abstract]
