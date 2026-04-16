"""Contract tests for SemanticLayer extension point.

``SemanticLayerContractTest`` verifies that any :class:`~pretensor.semantic.base.SemanticLayer`
implementation satisfies the full interface contract.  Cloud implementations
import this class and bind ``make_layer`` to their own factory.
"""

from __future__ import annotations

import abc
from typing import Any

import pytest

from pretensor.semantic import NullSemanticLayer, SemanticLayer

# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


class SemanticLayerContractTest(abc.ABC):
    """Reusable contract suite for :class:`SemanticLayer` implementations.

    Subclass and implement :meth:`make_layer` to verify any semantic layer.

    Example (Cloud)::

        class TestMyCloudSemanticLayer(SemanticLayerContractTest):
            def make_layer(self) -> SemanticLayer:
                return MyCloudSemanticLayer(config=...)
    """

    @abc.abstractmethod
    def make_layer(self) -> SemanticLayer:
        """Return an instance of the semantic layer under test."""

    # -- Interface shape -------------------------------------------------------

    def test_is_semantic_layer_instance(self) -> None:
        """Layer must be a SemanticLayer subclass."""
        layer = self.make_layer()
        assert isinstance(layer, SemanticLayer)

    # -- get_entity() ----------------------------------------------------------

    def test_get_entity_accepts_string(self) -> None:
        """get_entity() must accept a string argument without raising."""
        layer = self.make_layer()
        result = layer.get_entity("customers")
        assert result is None or isinstance(result, dict)

    def test_get_entity_returns_dict_or_none(self) -> None:
        """get_entity() must return a dict or None."""
        layer = self.make_layer()
        result = layer.get_entity("nonexistent_entity")
        assert result is None or isinstance(result, dict)

    def test_get_entity_empty_string(self) -> None:
        """get_entity('') must not raise."""
        layer = self.make_layer()
        result = layer.get_entity("")
        assert result is None or isinstance(result, dict)

    # -- get_metric() ----------------------------------------------------------

    def test_get_metric_accepts_string(self) -> None:
        """get_metric() must accept a string argument without raising."""
        layer = self.make_layer()
        result = layer.get_metric("monthly_revenue")
        assert result is None or isinstance(result, dict)

    def test_get_metric_returns_dict_or_none(self) -> None:
        """get_metric() must return a dict or None."""
        layer = self.make_layer()
        result = layer.get_metric("nonexistent_metric")
        assert result is None or isinstance(result, dict)

    def test_get_metric_empty_string(self) -> None:
        """get_metric('') must not raise."""
        layer = self.make_layer()
        result = layer.get_metric("")
        assert result is None or isinstance(result, dict)

    # -- get_dimensions() ------------------------------------------------------

    def test_get_dimensions_returns_list(self) -> None:
        """get_dimensions() must return a list."""
        layer = self.make_layer()
        result = layer.get_dimensions("customers")
        assert isinstance(result, list)

    def test_get_dimensions_items_are_dicts(self) -> None:
        """Every item returned by get_dimensions() must be a dict."""
        layer = self.make_layer()
        for item in layer.get_dimensions("customers"):
            assert isinstance(item, dict), f"Expected dict, got {type(item)}"

    def test_get_dimensions_empty_entity(self) -> None:
        """get_dimensions('') must return a list without raising."""
        layer = self.make_layer()
        result = layer.get_dimensions("")
        assert isinstance(result, list)

    # -- get_rules() -----------------------------------------------------------

    def test_get_rules_returns_list(self) -> None:
        """get_rules() must return a list."""
        layer = self.make_layer()
        result = layer.get_rules("orders")
        assert isinstance(result, list)

    def test_get_rules_items_are_dicts(self) -> None:
        """Every item returned by get_rules() must be a dict."""
        layer = self.make_layer()
        for item in layer.get_rules("orders"):
            assert isinstance(item, dict), f"Expected dict, got {type(item)}"

    def test_get_rules_empty_entity(self) -> None:
        """get_rules('') must return a list without raising."""
        layer = self.make_layer()
        result = layer.get_rules("")
        assert isinstance(result, list)

    # -- validate_query() ------------------------------------------------------

    def test_validate_query_returns_dict(self) -> None:
        """validate_query() must return a dict."""
        layer = self.make_layer()
        result = layer.validate_query("SELECT 1", connection_name="demo")
        assert isinstance(result, dict)

    def test_validate_query_has_valid_key(self) -> None:
        """validate_query() result must contain a 'valid' boolean key."""
        layer = self.make_layer()
        result = layer.validate_query("SELECT 1", connection_name="demo")
        assert "valid" in result, "'valid' key missing from validate_query result"
        assert isinstance(result["valid"], bool)

    def test_validate_query_has_errors_key(self) -> None:
        """validate_query() result must contain an 'errors' list key."""
        layer = self.make_layer()
        result = layer.validate_query("SELECT 1", connection_name="demo")
        assert "errors" in result, "'errors' key missing from validate_query result"
        assert isinstance(result["errors"], list)

    def test_validate_query_empty_sql(self) -> None:
        """validate_query('', ...) must return a dict without raising."""
        layer = self.make_layer()
        result = layer.validate_query("", connection_name="demo")
        assert isinstance(result, dict)

    # -- impact_semantic() -----------------------------------------------------

    def test_impact_semantic_returns_list(self) -> None:
        """impact_semantic() must return a list."""
        layer = self.make_layer()
        result = layer.impact_semantic("table::public.orders")
        assert isinstance(result, list)

    def test_impact_semantic_items_are_dicts(self) -> None:
        """Every item returned by impact_semantic() must be a dict."""
        layer = self.make_layer()
        for item in layer.impact_semantic("table::public.orders"):
            assert isinstance(item, dict), f"Expected dict, got {type(item)}"

    def test_impact_semantic_empty_node_id(self) -> None:
        """impact_semantic('') must return a list without raising."""
        layer = self.make_layer()
        result = layer.impact_semantic("")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestSemanticLayerAbstract:
    """SemanticLayer itself must remain abstract."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            SemanticLayer()  # type: ignore[abstract]

    def test_incomplete_subclass_not_instantiable(self) -> None:
        class Partial(SemanticLayer):
            def get_entity(self, entity_id: str) -> dict[str, Any] | None:
                return None

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Concrete: NullSemanticLayer
# ---------------------------------------------------------------------------


class TestNullSemanticLayerContract(SemanticLayerContractTest):
    """Run the full semantic layer contract against :class:`NullSemanticLayer`."""

    def make_layer(self) -> SemanticLayer:
        return NullSemanticLayer()

    # -- NullSemanticLayer-specific behaviour ----------------------------------

    def test_get_entity_is_none(self) -> None:
        assert self.make_layer().get_entity("any") is None

    def test_get_metric_is_none(self) -> None:
        assert self.make_layer().get_metric("any") is None

    def test_get_dimensions_is_empty(self) -> None:
        assert self.make_layer().get_dimensions("any") == []

    def test_get_rules_is_empty(self) -> None:
        assert self.make_layer().get_rules("any") == []

    def test_validate_query_is_valid_no_errors(self) -> None:
        result = self.make_layer().validate_query("SELECT 1", connection_name="demo")
        assert result == {"valid": True, "errors": []}

    def test_impact_semantic_is_empty(self) -> None:
        assert self.make_layer().impact_semantic("node::x") == []
