"""Pretensor Graph — schema knowledge graph backed by Kuzu.

Heavy imports (e.g. :class:`GraphBuilder`) are lazy so ``import pretensor``
works in tests that only touch the registry.
"""

# pyright: reportUnsupportedDunderAll=false
# Names in __all__ are provided via __getattr__; Pyright does not model that.

from __future__ import annotations

from typing import Any

__all__ = [
    "GraphBuilder",
    "GraphEdge",
    "GraphNode",
    "GraphRegistry",
    "KuzuStore",
    "RelationshipCandidate",
]


def __getattr__(name: str) -> Any:
    if name == "GraphBuilder":
        from pretensor.core.builder import GraphBuilder

        return GraphBuilder
    if name == "GraphRegistry":
        from pretensor.core.registry import GraphRegistry

        return GraphRegistry
    if name == "KuzuStore":
        from pretensor.core.store import KuzuStore

        return KuzuStore
    if name == "GraphNode":
        from pretensor.graph_models.node import GraphNode

        return GraphNode
    if name == "GraphEdge":
        from pretensor.graph_models.edge import GraphEdge

        return GraphEdge
    if name == "RelationshipCandidate":
        from pretensor.graph_models.relationship import RelationshipCandidate

        return RelationshipCandidate
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
