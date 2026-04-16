"""Abstract base for search index implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pretensor.core.registry import GraphRegistry

__all__ = ["BaseSearchIndex", "SearchResult"]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked hit from a search index."""

    node_type: str
    name: str
    database_name: str
    connection_name: str
    description: str
    score: float


class BaseSearchIndex(ABC):
    """Abstract search index over graph metadata.

    Concrete implementations may use keyword (BM25), vector (embeddings),
    or hybrid strategies.  OSS ships ``KeywordSearchIndex``; Cloud can register
    a ``HybridSearchIndex`` subclass.
    """

    @abstractmethod
    def search(
        self,
        q: str,
        *,
        db: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search indexed metadata.

        Args:
            q: Free-text query string.
            db: Optional connection or logical database name filter.
            limit: Maximum number of results to return.

        Returns:
            Ranked list of search results.
        """

    @abstractmethod
    def similar(
        self,
        name: str,
        *,
        db: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return nodes related to the named node.

        OSS implementations use graph proximity (FK / join edges).  Cloud
        implementations may additionally use vector similarity.

        Args:
            name: Fully-qualified node name, e.g. ``schema.table``.
            db: Optional connection or logical database name filter.
            limit: Maximum number of results to return.

        Returns:
            Related nodes ranked by proximity or similarity.
        """

    @abstractmethod
    def index_graph(self, registry: GraphRegistry) -> None:
        """Build or rebuild the search index from all graphs in *registry*.

        Args:
            registry: Loaded registry describing available graph files.
        """
