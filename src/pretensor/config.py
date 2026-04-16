"""Graph package configuration (clustering, intelligence defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pretensor.core.store import KuzuStore
    from pretensor.intelligence.combining import ConfidenceCombiner
    from pretensor.intelligence.scoring import ScorerRegistry
    from pretensor.search.base import BaseSearchIndex
    from pretensor.semantic.base import SemanticLayer
    from pretensor.semantic.yaml_layer import YamlSemanticLayer

__all__ = ["GraphConfig", "PretensorConfig", "load_semantic_layer"]


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Tunable parameters for graph intelligence (clustering, join paths)."""

    stale_index_warning_days: int = 7
    """Warn in MCP responses when ``last_indexed`` is older than this many days."""

    clustering_resolution_override: float | None = None
    """If set, skip automatic resolution heuristic and use this value for Leiden."""

    join_path_max_depth: int = 4
    """Maximum hop count for precomputed intra-cluster join paths."""

    min_cluster_size_merge: int = 3
    """Clusters with fewer tables are merged into a neighbor cluster."""

    collapse_shadow_aliases: bool = True
    """Treat single-source views (``table_type='view'`` with exactly one
    incoming ``LINEAGE`` edge) as transparent aliases of their base table.
    When enabled, intelligence consumers (clustering, INFERRED_JOIN, impact,
    traverse, context) suppress or pass through these aliases."""

    lineage_in_max_for_alias: int = 1
    """Maximum incoming LINEAGE edge count for a view to be considered a
    shadow alias.  The default of 1 matches pure 1:1 projections."""


def _default_scorer_registry() -> ScorerRegistry:
    from pretensor.intelligence.heuristic import HeuristicScorer
    from pretensor.intelligence.scoring import ScorerRegistry

    return ScorerRegistry([HeuristicScorer()])


def _default_combiner() -> ConfidenceCombiner:
    from pretensor.intelligence.combining import MaxScoreCombiner

    return MaxScoreCombiner()


def _default_search_index_cls() -> type[BaseSearchIndex]:
    from pretensor.search.index import KeywordSearchIndex

    return KeywordSearchIndex


def _default_semantic_layer() -> SemanticLayer:
    from pretensor.semantic.base import NullSemanticLayer

    return NullSemanticLayer()


@dataclass
class PretensorConfig:
    """Central pluggable configuration for the Pretensor graph system.

    Bundles all tunable and swappable components so that Cloud can ship a
    ``CloudConfig(PretensorConfig)`` subclass with its own implementations
    without patching individual files.

    All fields default to the OSS implementations, so constructing
    ``PretensorConfig()`` with no arguments is always valid for basic use.

    Attributes:
        graph: Graph intelligence tuning parameters (clustering, join paths).
        scorer_registry: Ordered registry of relationship scorers used by discovery.
            Default: ``ScorerRegistry([HeuristicScorer()])``.
        combiner: Strategy for merging scored relationship candidates.
            Default: ``MaxScoreCombiner()``.
        search_index_cls: Class used to build or load the keyword/vector search index.
            Default: ``KeywordSearchIndex``.
        semantic_layer: Semantic enrichment layer for entities, metrics, and dimensions.
            Default: ``NullSemanticLayer()`` (no-op for OSS).
    """

    graph: GraphConfig = field(default_factory=GraphConfig)
    scorer_registry: ScorerRegistry = field(
        default_factory=_default_scorer_registry
    )
    combiner: ConfidenceCombiner = field(default_factory=_default_combiner)
    search_index_cls: type[BaseSearchIndex] = field(
        default_factory=_default_search_index_cls
    )
    semantic_layer: SemanticLayer = field(default_factory=_default_semantic_layer)


def load_semantic_layer(
    yaml_path: Path,
    *,
    store: KuzuStore,
    database_key: str,
    dialect: str = "postgres",
) -> YamlSemanticLayer:
    """Read a YAML file and return a ready-to-use :class:`YamlSemanticLayer`.

    The YAML must match the
    :class:`pretensor.introspection.models.semantic.SemanticLayer` schema.
    The returned layer uses ``store`` for graph lookups (metric compilation,
    query validation, impact) scoped to ``database_key``.

    Args:
        yaml_path: Filesystem path to the semantic layer YAML.
        store: Open Kuzu store for the same ``database_key``.
        database_key: Logical database key as recorded on ``SchemaTable``.
        dialect: sqlglot dialect for SQL parsing (default PostgreSQL).

    Returns:
        A :class:`YamlSemanticLayer` instance.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        pydantic.ValidationError: If the YAML fails schema validation.
    """
    from pretensor.introspection.models.semantic import (
        SemanticLayer as SemanticLayerModel,
    )
    from pretensor.semantic.yaml_layer import YamlSemanticLayer

    text = yaml_path.read_text(encoding="utf-8")
    model = SemanticLayerModel.from_yaml(text)
    return YamlSemanticLayer(
        model,
        store=store,
        database_key=database_key,
        dialect=dialect,
    )
