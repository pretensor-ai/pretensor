"""SemanticLayer interface and NullSemanticLayer stub.

The ``SemanticLayer`` ABC defines the extension point for semantic-layer
implementations. ``NullSemanticLayer`` is the default no-op — all methods
return ``None`` or empty collections so graph operations and MCP tool
responses compile and run without any semantic-layer implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["SemanticLayer", "NullSemanticLayer"]


class SemanticLayer(ABC):
    """Abstract interface for the Pretensor semantic layer.

    Cloud implementations populate this with business-level knowledge
    (metrics, dimensions, business rules). OSS callers always receive
    ``NullSemanticLayer``, which returns ``None`` or empty collections
    for every method.

    All methods are intentionally narrow: they accept identifiers (strings
    or simple primitives) and return plain Python types so callers never
    depend on Cloud-specific model classes.
    """

    @abstractmethod
    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Return business entity metadata by ID.

        Args:
            entity_id: Canonical entity identifier (e.g. ``"customers"``).

        Returns:
            Mapping of entity attributes, or ``None`` if not found.
        """

    @abstractmethod
    def get_metric(self, metric_name: str) -> dict[str, Any] | None:
        """Return a named metric definition.

        Args:
            metric_name: Unique metric name (e.g. ``"monthly_revenue"``).

        Returns:
            Mapping with metric metadata and SQL template, or ``None``.
        """

    @abstractmethod
    def get_dimensions(self, entity_id: str) -> list[dict[str, Any]]:
        """Return dimension definitions associated with an entity.

        Args:
            entity_id: Canonical entity identifier.

        Returns:
            List of dimension attribute mappings; empty list if none exist.
        """

    @abstractmethod
    def get_rules(self, entity_id: str) -> list[dict[str, Any]]:
        """Return business rules that apply to an entity.

        Args:
            entity_id: Canonical entity identifier.

        Returns:
            List of business rule mappings; empty list if none exist.
        """

    @abstractmethod
    def validate_query(self, sql: str, *, connection_name: str) -> dict[str, Any]:
        """Validate a SQL query against the semantic layer.

        Args:
            sql: SQL statement to validate.
            connection_name: Target connection for dialect and schema context.

        Returns:
            Mapping with at minimum ``{"valid": bool, "errors": list[str]}``.
        """

    @abstractmethod
    def impact_semantic(self, table_node_id: str) -> list[dict[str, Any]]:
        """Return semantic objects impacted by changes to a table.

        Args:
            table_node_id: Kuzu node ID of the affected ``SchemaTable``.

        Returns:
            List of impacted semantic object descriptors (metrics, rules,
            dimensions); empty list if none exist.
        """


class NullSemanticLayer(SemanticLayer):
    """No-op semantic layer for OSS deployments.

    Every method returns ``None`` or an empty collection. This lets the
    graph core and MCP tools reference a ``SemanticLayer`` without
    requiring Cloud credentials or implementations.
    """

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        return None

    def get_metric(self, metric_name: str) -> dict[str, Any] | None:
        return None

    def get_dimensions(self, entity_id: str) -> list[dict[str, Any]]:
        return []

    def get_rules(self, entity_id: str) -> list[dict[str, Any]]:
        return []

    def validate_query(self, sql: str, *, connection_name: str) -> dict[str, Any]:
        return {"valid": True, "errors": []}

    def impact_semantic(self, table_node_id: str) -> list[dict[str, Any]]:
        return []
