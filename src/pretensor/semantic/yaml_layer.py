"""``YamlSemanticLayer`` — OSS :class:`SemanticLayer` backed by a YAML file.

Loads the Pydantic :class:`pretensor.introspection.models.semantic.SemanticLayer`
container and answers the :class:`pretensor.semantic.base.SemanticLayer` ABC
using the loaded definitions plus the indexed Kuzu graph. Query validation is
delegated to :class:`pretensor.validation.query_validator.QueryValidator`; the
compile path lives in :mod:`pretensor.semantic.compiler`.
"""

from __future__ import annotations

from typing import Any

from pretensor.core.store import KuzuStore
from pretensor.introspection.models.semantic import (
    AttributeRole,
    Entity,
)
from pretensor.introspection.models.semantic import (
    SemanticLayer as SemanticLayerModel,
)
from pretensor.semantic.base import SemanticLayer
from pretensor.semantic.compiler import MetricSqlCompiler
from pretensor.validation.query_validator import QueryValidator

__all__ = ["YamlSemanticLayer"]


_DEFAULT_DIALECT = "postgres"


class YamlSemanticLayer(SemanticLayer):
    """File-based semantic layer for OSS deployments.

    Constructed against a loaded :class:`SemanticLayerModel` and an open
    :class:`KuzuStore`. Methods on the ABC return plain ``dict`` / ``list``
    payloads as documented on :class:`SemanticLayer`; no Cloud model classes
    leak through.
    """

    def __init__(
        self,
        layer: SemanticLayerModel,
        *,
        store: KuzuStore,
        database_key: str,
        dialect: str = _DEFAULT_DIALECT,
    ) -> None:
        self._layer = layer
        self._store = store
        self._database_key = database_key
        self._dialect = dialect
        self._validator = QueryValidator(
            store,
            connection_name=layer.connection_name,
            database_key=database_key,
            dialect=dialect,
        )
        self._compiler = MetricSqlCompiler(
            store,
            connection_name=layer.connection_name,
            database_key=database_key,
            dialect=dialect,
        )

    # ---- accessors --------------------------------------------------------

    @property
    def layer(self) -> SemanticLayerModel:
        return self._layer

    @property
    def compiler(self) -> MetricSqlCompiler:
        return self._compiler

    # ---- SemanticLayer ABC -------------------------------------------------

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        entity = self._find_entity(entity_id)
        if entity is None:
            return None
        return entity.model_dump(mode="json")

    def get_metric(self, metric_name: str) -> dict[str, Any] | None:
        for domain in self._layer.domains:
            for entity in domain.entities:
                for metric in entity.metrics:
                    if metric.name == metric_name:
                        payload = metric.model_dump(mode="json")
                        payload["entity"] = entity.name
                        payload["source_table"] = entity.source_table
                        return payload
        return None

    def get_dimensions(self, entity_id: str) -> list[dict[str, Any]]:
        entity = self._find_entity(entity_id)
        if entity is None:
            return []
        out: list[dict[str, Any]] = []
        for attr in entity.attributes:
            if attr.role in {
                AttributeRole.DIMENSION,
                AttributeRole.TIME_DIMENSION,
                AttributeRole.IDENTIFIER,
            }:
                out.append(attr.model_dump(mode="json"))
        return out

    def get_rules(self, entity_id: str) -> list[dict[str, Any]]:
        # OSS has no business-rule surface. Cloud overrides.
        return []

    def validate_query(
        self, sql: str, *, connection_name: str
    ) -> dict[str, Any]:
        if connection_name and connection_name != self._layer.connection_name:
            return {
                "valid": False,
                "errors": [
                    f"connection {connection_name!r} does not match "
                    f"semantic layer connection {self._layer.connection_name!r}"
                ],
            }
        result = self._validator.validate(sql)
        errors: list[str] = []
        errors.extend(result.syntax_errors)
        errors.extend(f"missing_table: {t}" for t in result.missing_tables)
        errors.extend(f"missing_column: {c}" for c in result.missing_columns)
        errors.extend(f"invalid_join: {j.message}" for j in result.invalid_joins)
        return {"valid": result.valid, "errors": errors}

    def impact_semantic(self, table_node_id: str) -> list[dict[str, Any]]:
        rows = self._store.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $nid})
            RETURN t.schema_name, t.table_name
            LIMIT 1
            """,
            {"nid": table_node_id},
        )
        if not rows:
            return []
        sn, tn = rows[0]
        schema_name = str(sn)
        table_name = str(tn)
        qualified_full = f"{schema_name}.{table_name}"
        bare = table_name
        out: list[dict[str, Any]] = []
        for domain in self._layer.domains:
            for entity in domain.entities:
                owns = entity.source_table in (qualified_full, bare)
                for metric in entity.metrics:
                    if owns:
                        out.append(
                            {
                                "kind": "metric",
                                "metric": metric.name,
                                "entity": entity.name,
                                "reason": "owns_source_table",
                            }
                        )
                        continue
                    expr = metric.expression or ""
                    if expr and (qualified_full in expr or f" {bare} " in f" {expr} "):
                        out.append(
                            {
                                "kind": "metric",
                                "metric": metric.name,
                                "entity": entity.name,
                                "reason": "expression_references_table",
                            }
                        )
        return out

    # ---- internals --------------------------------------------------------

    def _find_entity(self, entity_id: str) -> Entity | None:
        for domain in self._layer.domains:
            for entity in domain.entities:
                if entity.name == entity_id:
                    return entity
        return None
