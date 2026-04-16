"""Semantic layer extension point for Pretensor.

OSS ships ``NullSemanticLayer`` (all methods return None/empty).
Cloud implementations override ``SemanticLayer`` to populate the
reserved Kuzu node types (``Metric``, ``Dimension``, ``BusinessRule``)
and their associated edge types.
"""

from pretensor.semantic.base import NullSemanticLayer, SemanticLayer
from pretensor.semantic.compiler import (
    CompiledMetric,
    MetricCompileError,
    MetricSqlCompiler,
)
from pretensor.semantic.yaml_layer import YamlSemanticLayer

__all__ = [
    "CompiledMetric",
    "MetricCompileError",
    "MetricSqlCompiler",
    "NullSemanticLayer",
    "SemanticLayer",
    "YamlSemanticLayer",
]
