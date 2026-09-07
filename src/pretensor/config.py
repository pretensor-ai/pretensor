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

__all__ = [
    "EmbeddingsConfig",
    "GraphConfig",
    "PretensorConfig",
    "load_semantic_layer",
]


@dataclass(frozen=True, slots=True)
class EmbeddingsConfig:
    """Opt-in toggles for the ``[embeddings]`` extra."""

    index_tables: bool = False
    """When True and ``[embeddings]`` is installed, ``_EmbeddingIndexStep``
    computes one embedding per :class:`SchemaTable` during intelligence
    indexing.  Default ``False`` keeps the null path byte-identical to the
    pre-embedding pipeline output."""

    cluster_blend: float = 0.0
    """Weight (in [0.0, 1.0]) for blending embedding cosine similarity into
    existing FK / INFERRED_JOIN edge weights before community detection.

    For each existing structural edge ``(a, b)`` where both endpoints
    carry an embedding, the edge weight is incremented by
    ``cluster_blend * cosine(emb_a, emb_b)``.  No new edges are introduced
    from cosine alone — the blend only nudges already-structurally-adjacent
    pairs.  Default ``0.0`` keeps cluster membership byte-identical to the
    pre-embedding pipeline output (Invariant #6).  A small positive value
    (~0.25) lets domain-related tables with weak lexical overlap drift
    into the same cluster.
    """

    join_threshold: float | None = None
    """Cosine threshold for proposing ``INFERRED_JOIN`` candidates from
    embedding similarity.  ``None`` (default) disables the
    embedding scorer entirely → null-path identical.  When set (typical
    value ~0.85), an ``EmbeddingRelationshipScorer`` is registered after
    the heuristic scorer; pairs whose cosine ≥ threshold are emitted as
    ``status="suggested"`` candidates with ``source="embedding"``.  The
    scorer applies a type-family compatibility gate per column pair
    (reusing the heuristic's check) so high-cosine pairs with
    incompatible types are dropped.
    """

    role_weight: float = 0.0
    """Weight for the embedding-based role-classification vote.
    ``0.0`` (default) skips the embedding signal entirely → null-path
    identical.  When > 0, ``classify_database_tables_async`` embeds each
    table and adds ``role_weight * cosine(table, role_centroid)`` to
    every role's heuristic score; the heuristic remains the primary
    signal.  Centroids are computed once per process from a small curated
    exemplar set in ``pretensor.intelligence.role_exemplars``.
    """

    def __post_init__(self) -> None:
        # Validate the float knobs at construction time so misuse surfaces
        # close to its source rather than mid-pipeline. All three numeric
        # toggles are checked here for one consistent contract on the
        # dataclass; ``EmbeddingRelationshipScorer.__init__`` keeps its own
        # belt-and-braces check for callers that construct the scorer
        # directly with an out-of-band threshold.
        if not 0.0 <= self.cluster_blend <= 1.0:
            msg = (
                f"EmbeddingsConfig.cluster_blend must be in [0.0, 1.0]; "
                f"got {self.cluster_blend!r}"
            )
            raise ValueError(msg)
        if not 0.0 <= self.role_weight <= 1.0:
            msg = (
                f"EmbeddingsConfig.role_weight must be in [0.0, 1.0]; "
                f"got {self.role_weight!r}"
            )
            raise ValueError(msg)
        if self.join_threshold is not None and not (0.0 <= self.join_threshold <= 1.0):
            msg = (
                f"EmbeddingsConfig.join_threshold must be in [0.0, 1.0] "
                f"or None; got {self.join_threshold!r}"
            )
            raise ValueError(msg)


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

    same_name_max_tables: int | None = 8
    """Maximum number of distinct tables sharing a column name for the
    same-name join heuristic to fire on that column.  A name shared across
    more tables (``customer_id`` on every fact table, ``date_id`` on 40
    tables) is a generic key whose name alone is too weak a join signal —
    pairing all of them generates O(tables²) inferred edges and floods the
    join-path precompute.  ``None`` disables the gate."""

    def __post_init__(self) -> None:
        if self.same_name_max_tables is not None and self.same_name_max_tables < 2:
            raise ValueError(
                "same_name_max_tables must be >= 2 or None, got "
                f"{self.same_name_max_tables}"
            )


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

    Bundles all tunable and swappable components so that downstream packages can ship a
    ``PretensorConfig`` subclass with its own implementations
    without patching individual files.

    All fields default to the OSS implementations, so constructing
    ``PretensorConfig()`` with no arguments is always valid for basic use.

    Attributes:
        graph: Graph intelligence tuning parameters (clustering, join paths).
        embeddings: Opt-in toggles for the ``[embeddings]`` extra.
            Default: ``EmbeddingsConfig()`` (all toggles off).
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
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    scorer_registry: ScorerRegistry = field(default_factory=_default_scorer_registry)
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
