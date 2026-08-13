"""Kuzu-backed persistence for the schema graph."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import kuzu

from pretensor.core.graph_schema_manager import (
    _SCHEMA_TABLE_EMBEDDING_DIM,  # noqa: F401 – re-exported for existing callers
    GraphSchemaManager,
)
from pretensor.core.graph_store import GraphStore, TableEmbeddingRow
from pretensor.core.query_runner import QueryRunner
from pretensor.graph_models.consumer import ConsumesEdge, ExternalConsumerNode
from pretensor.graph_models.edge import GraphEdge, LineageEdge
from pretensor.graph_models.entity import EntityNode
from pretensor.graph_models.node import GraphNode
from pretensor.graph_models.relationship import RelationshipCandidate

__all__ = ["KuzuStore", "TableEmbeddingRow"]

logger = logging.getLogger(__name__)


class KuzuStore:
    """Open or create a Kuzu database file and upsert schema graph nodes/edges.

    Facade that composes :class:`~pretensor.core.query_runner.QueryRunner`,
    :class:`~pretensor.core.graph_store.GraphStore`, and
    :class:`~pretensor.core.graph_schema_manager.GraphSchemaManager`.
    All public methods preserve their original signatures and behavior.
    """

    def __init__(self, database_path: Path, *, read_only: bool = False) -> None:
        self._path = database_path
        # ``read_only`` is enforced by the Kuzu engine itself (write statements
        # raise); it is not re-stored as an attribute because nothing downstream
        # needs to branch on it.
        self._db = kuzu.Database(str(database_path), read_only=read_only)
        self._conn = kuzu.Connection(self._db)
        self._runner = QueryRunner(self._conn, self._path)
        self._schema = GraphSchemaManager(self._conn)
        self._graph = GraphStore(self._runner)

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        """Close the underlying database handle."""
        self._db.close()

    def ensure_schema(self) -> None:
        """Create node and relationship tables if they do not exist."""
        self._schema.ensure_schema()

    # ── Execution methods (delegated to QueryRunner) ──────────────────────────

    def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a write (mutation) Cypher query. Use this for DELETE / SET operations."""
        return self._runner.execute_write(query, params)

    def execute(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> kuzu.QueryResult | list[kuzu.QueryResult]:
        """Run a read query (or arbitrary Cypher) with optional parameters."""
        return self._runner.execute(cypher, parameters)

    def query_all_rows(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute Cypher and materialize all result rows as tuples.

        Keeps callers (MCP, search index) off raw ``kuzu.QueryResult`` handles
        while still routing reads through :class:`KuzuStore`.
        """
        return self._runner.query_all_rows(cypher, parameters)

    # ── Graph methods (delegated to GraphStore) ───────────────────────────────

    def clear_graph(self) -> None:
        """Remove all FK edges and table nodes (full re-index)."""
        return self._graph.clear_graph()

    def clear_connection_subgraph(self, connection_name: str) -> None:
        """Remove all nodes and edges scoped to one indexed connection (unified graph merge)."""
        return self._graph.clear_connection_subgraph(connection_name)

    def upsert_table(self, node: GraphNode) -> None:
        """Insert or update a single ``SchemaTable`` node."""
        return self._graph.upsert_table(node)

    def upsert_entity(self, node: EntityNode) -> None:
        """Insert or update a single ``Entity`` node."""
        return self._graph.upsert_entity(node)

    def upsert_represents(self, entity_node_id: str, table_node_id: str) -> None:
        """Link an entity to a table (idempotent; one edge per pair)."""
        return self._graph.upsert_represents(entity_node_id, table_node_id)

    def set_table_entity_type(self, table_node_id: str, entity_type: str) -> None:
        """Set the business entity label on a ``SchemaTable`` node."""
        return self._graph.set_table_entity_type(table_node_id, entity_type)

    def set_table_classification(
        self,
        table_node_id: str,
        *,
        role: str,
        role_confidence: float,
        classification_signals_json: str,
    ) -> None:
        """Persist classifier output on a ``SchemaTable`` node."""
        return self._graph.set_table_classification(
            table_node_id,
            role=role,
            role_confidence=role_confidence,
            classification_signals_json=classification_signals_json,
        )

    def set_table_embedding(
        self, table_node_id: str, vector: list[float] | None
    ) -> None:
        """Set (or clear with ``None``) the embedding on a ``SchemaTable`` node."""
        return self._graph.set_table_embedding(table_node_id, vector)

    def has_any_table_embeddings(self) -> bool:
        """Return True iff any ``SchemaTable`` row carries a non-null embedding.

        Cheap LIMIT-1 probe used by the builder to warn before a full
        rebuild silently drops previously computed vectors.
        """
        return self._graph.has_any_table_embeddings()

    def iter_table_embeddings(
        self,
        *,
        database: str | None = None,
        cluster: str | None = None,
    ) -> Iterator[TableEmbeddingRow]:
        """Yield ``SchemaTable`` rows for cosine retrieval, honoring optional filters.

        ``database`` matches either ``SchemaTable.connection_name`` or
        ``SchemaTable.database`` (so callers may pass a registry connection
        name or a logical database name, mirroring ``query.db``).  ``cluster``
        requires an ``IN_CLUSTER`` edge to a :class:`Cluster` with the given
        ``cluster_id``; when unset, the cluster membership is looked up via
        ``OPTIONAL MATCH`` and populated on the row when present.

        Rows with ``embedding is None`` are still yielded so the caller can
        detect the "no tables carry vectors" case and return the documented
        ``fallback_bm25`` envelope without a second store round-trip.
        """
        return self._graph.iter_table_embeddings(database=database, cluster=cluster)

    def upsert_fk_edge(self, edge: GraphEdge) -> None:
        """Insert a foreign-key relationship edge (assumes endpoints exist)."""
        return self._graph.upsert_fk_edge(edge)

    def merge_fk_edge(self, edge: GraphEdge) -> None:
        """Idempotent FK edge upsert by ``edge_id`` (avoids duplicates on reindex)."""
        return self._graph.merge_fk_edge(edge)

    def upsert_lineage_edge(self, edge: LineageEdge) -> None:
        """Insert or update a ``LINEAGE`` edge (idempotent by ``edge_id``)."""
        return self._graph.upsert_lineage_edge(edge)

    def clear_lineage_edges(self, connection_name: str) -> None:
        """Remove all ``LINEAGE`` edges touching tables on one connection."""
        return self._graph.clear_lineage_edges(connection_name)

    def clear_dbt_model_dependency_lineage(self, connection_name: str) -> None:
        """Remove dbt ``parent_map`` lineage edges for one connection (see enrichment dbt)."""
        return self._graph.clear_dbt_model_dependency_lineage(connection_name)

    def clear_intelligence_artifacts(self) -> None:
        """Remove clusters and precomputed join paths (keeps tables and FK edges)."""
        return self._graph.clear_intelligence_artifacts()

    def upsert_external_consumer(self, node: ExternalConsumerNode) -> None:
        """Insert or update an ``ExternalConsumer`` node (idempotent by ``node_id``)."""
        return self._graph.upsert_external_consumer(node)

    def upsert_consumes_edge(self, edge: ConsumesEdge) -> None:
        """Insert or update a ``CONSUMES`` edge (idempotent by ``edge_id``)."""
        return self._graph.upsert_consumes_edge(edge)

    def mark_tables_have_external_consumers(self, table_node_ids: list[str]) -> None:
        """Set ``has_external_consumers = true`` on the given ``SchemaTable`` nodes."""
        return self._graph.mark_tables_have_external_consumers(table_node_ids)

    def sweep_stale_consumers(
        self, service_name: str, connection_name: str, current_scan_run_id: str
    ) -> None:
        """Delete stale consumers/edges from prior runs of one service and clear
        ``has_external_consumers`` on tables that lost their last consumer."""
        return self._graph.sweep_stale_consumers(
            service_name, connection_name, current_scan_run_id
        )

    def upsert_column_for_table(
        self,
        *,
        column_node_id: str,
        connection_name: str,
        database: str,
        schema_name: str,
        table_name: str,
        column_name: str,
        data_type: str,
        nullable: bool,
        is_primary_key: bool,
        is_foreign_key: bool,
        table_node_id: str,
        comment: str | None = None,
        description: str | None = None,
        default_value: str | None = None,
        is_indexed: bool = False,
        check_constraints: list[str] | None = None,
        ordinal_position: int | None = None,
        most_common_values_json: str | None = None,
        histogram_bounds_json: str | None = None,
        stats_correlation: float | None = None,
        column_cardinality: int | None = None,
        index_type: str | None = None,
        index_is_unique: bool | None = None,
        parent_column_id: str | None = None,
        is_array: bool = False,
    ) -> None:
        """Insert or update a ``SchemaColumn`` and link it to table or parent column."""
        return self._graph.upsert_column_for_table(
            column_node_id=column_node_id,
            connection_name=connection_name,
            database=database,
            schema_name=schema_name,
            table_name=table_name,
            column_name=column_name,
            data_type=data_type,
            nullable=nullable,
            is_primary_key=is_primary_key,
            is_foreign_key=is_foreign_key,
            table_node_id=table_node_id,
            comment=comment,
            description=description,
            default_value=default_value,
            is_indexed=is_indexed,
            check_constraints=check_constraints,
            ordinal_position=ordinal_position,
            most_common_values_json=most_common_values_json,
            histogram_bounds_json=histogram_bounds_json,
            stats_correlation=stats_correlation,
            column_cardinality=column_cardinality,
            index_type=index_type,
            index_is_unique=index_is_unique,
            parent_column_id=parent_column_id,
            is_array=is_array,
        )

    def delete_column_node(self, column_node_id: str) -> None:
        """Remove a column node (``HAS_COLUMN`` edges removed by cascade)."""
        return self._graph.delete_column_node(column_node_id)

    def upsert_metric_template(
        self,
        *,
        node_id: str,
        connection_name: str,
        database: str,
        dialect: str = "postgresql",
        name: str,
        display_name: str,
        description: str,
        sql_template: str,
        tables_used: list[str],
        validated: bool,
        validation_errors: list[str],
        generated_at_iso: str,
        stale: bool,
        depends_on_table_node_ids: list[str],
    ) -> None:
        """Insert or replace a ``MetricTemplate`` and its ``METRIC_DEPENDS`` edges."""
        return self._graph.upsert_metric_template(
            node_id=node_id,
            connection_name=connection_name,
            database=database,
            dialect=dialect,
            name=name,
            display_name=display_name,
            description=description,
            sql_template=sql_template,
            tables_used=tables_used,
            validated=validated,
            validation_errors=validation_errors,
            generated_at_iso=generated_at_iso,
            stale=stale,
            depends_on_table_node_ids=depends_on_table_node_ids,
        )

    def delete_fk_edges_touching_column(
        self, table_node_id: str, column_name: str
    ) -> int:
        """Delete FK edges where the column appears as source or target. Returns count."""
        return self._graph.delete_fk_edges_touching_column(table_node_id, column_name)

    def delete_inferred_joins_touching_column(
        self, table_node_id: str, column_name: str
    ) -> int:
        """Delete inferred join edges touching a column on either endpoint."""
        return self._graph.delete_inferred_joins_touching_column(
            table_node_id, column_name
        )

    def delete_table_node_cascade(self, table_node_id: str) -> None:
        """Remove a table and its column nodes; caller should handle intelligence cleanup."""
        return self._graph.delete_table_node_cascade(table_node_id)

    def mark_clusters_stale_for_table(
        self, table_node_id: str, database_key: str
    ) -> None:
        """Set ``stale`` on clusters linked to the table."""
        return self._graph.mark_clusters_stale_for_table(table_node_id, database_key)

    def remove_table_from_clusters(self, table_node_id: str) -> None:
        """Drop ``IN_CLUSTER`` edges for a table (cluster nodes kept)."""
        return self._graph.remove_table_from_clusters(table_node_id)

    def mark_join_paths_stale_for_table(
        self, table_node_id: str, database_key: str
    ) -> None:
        """Flag precomputed paths that start or end at this table."""
        return self._graph.mark_join_paths_stale_for_table(table_node_id, database_key)

    def delete_join_paths_for_table(
        self, table_node_id: str, database_key: str
    ) -> None:
        """Remove precomputed paths touching a table (used when removing a table)."""
        return self._graph.delete_join_paths_for_table(table_node_id, database_key)

    def upsert_inferred_join(self, candidate: RelationshipCandidate) -> None:
        """Insert an inferred join edge (assumes endpoint table nodes exist)."""
        return self._graph.upsert_inferred_join(candidate)

    def upsert_cluster(
        self,
        *,
        node_id: str,
        database_key: str,
        label: str,
        description: str,
        cohesion_score: float,
        table_count: int,
        stale: bool = False,
        schema_pattern: str | None = None,
    ) -> None:
        """Insert or replace a ``Cluster`` node."""
        return self._graph.upsert_cluster(
            node_id=node_id,
            database_key=database_key,
            label=label,
            description=description,
            cohesion_score=cohesion_score,
            table_count=table_count,
            stale=stale,
            schema_pattern=schema_pattern,
        )

    def upsert_in_cluster(self, table_node_id: str, cluster_node_id: str) -> None:
        """Link a table to a cluster (idempotent)."""
        return self._graph.upsert_in_cluster(table_node_id, cluster_node_id)

    def set_cluster_schema_pattern(
        self, cluster_node_id: str, schema_pattern: str
    ) -> None:
        """Set ``schema_pattern`` on an existing ``Cluster`` node."""
        return self._graph.set_cluster_schema_pattern(cluster_node_id, schema_pattern)

    def upsert_join_path(
        self,
        *,
        node_id: str,
        database_key: str,
        from_table_id: str,
        to_table_id: str,
        depth: int,
        confidence: float,
        ambiguous: bool,
        steps_json: str,
        semantic_label: str,
        stale: bool = False,
    ) -> None:
        """Insert or replace a precomputed ``JoinPath`` node."""
        return self._graph.upsert_join_path(
            node_id=node_id,
            database_key=database_key,
            from_table_id=from_table_id,
            to_table_id=to_table_id,
            depth=depth,
            confidence=confidence,
            ambiguous=ambiguous,
            steps_json=steps_json,
            semantic_label=semantic_label,
            stale=stale,
        )

    def merge_same_entity_edge(
        self,
        *,
        edge_id: str,
        from_entity_node_id: str,
        to_entity_node_id: str,
        status: str,
        score: float,
        join_columns: str | None,
        reasoning: str | None,
        created_at: str,
        confirmed_at: str | None = None,
        confirmed_by: str | None = None,
    ) -> None:
        """Upsert a ``SAME_ENTITY`` edge between two ``Entity`` nodes (idempotent by ``edge_id``)."""
        return self._graph.merge_same_entity_edge(
            edge_id=edge_id,
            from_entity_node_id=from_entity_node_id,
            to_entity_node_id=to_entity_node_id,
            status=status,
            score=score,
            join_columns=join_columns,
            reasoning=reasoning,
            created_at=created_at,
            confirmed_at=confirmed_at,
            confirmed_by=confirmed_by,
        )

    def delete_same_entity_edge(self, edge_id: str) -> None:
        """Remove a ``SAME_ENTITY`` edge by ``edge_id``."""
        return self._graph.delete_same_entity_edge(edge_id)

    def same_entity_edge_endpoints(self, edge_id: str) -> tuple[str, str] | None:
        """Return ``(from_entity_node_id, to_entity_node_id)`` for an existing edge."""
        return self._graph.same_entity_edge_endpoints(edge_id)

    def list_same_entity_edges(
        self, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Return same-entity links with table context for CLI and MCP."""
        return self._graph.list_same_entity_edges(status=status)

    def list_entities_with_primary_table(
        self, connection_name: str
    ) -> list[tuple[str, str, str]]:
        """Return ``(entity_name, schema.table, entity_node_id)`` per entity."""
        return self._graph.list_entities_with_primary_table(connection_name)

    def table_node_id_for_entity(self, entity_node_id: str) -> str | None:
        """First ``SchemaTable`` linked by ``REPRESENTS`` (ordered by name)."""
        return self._graph.table_node_id_for_entity(entity_node_id)

    def columns_for_table(
        self, table_node_id: str
    ) -> list[tuple[str, str, bool, bool]]:
        """Return ``(column_name, data_type, nullable, is_pk)`` rows."""
        return self._graph.columns_for_table(table_node_id)

    def count_same_entity_by_status(self, status: str) -> int:
        """Count ``SAME_ENTITY`` edges with the given status."""
        return self._graph.count_same_entity_by_status(status)
