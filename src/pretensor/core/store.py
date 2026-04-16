"""Kuzu-backed persistence for the schema graph."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import kuzu

from pretensor.core import schema as graph_schema
from pretensor.graph_models.edge import GraphEdge, LineageEdge
from pretensor.graph_models.entity import EntityNode
from pretensor.graph_models.node import GraphNode
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.observability import log_timed_operation

__all__ = ["KuzuStore"]

logger = logging.getLogger(__name__)


class KuzuStore:
    """Open or create a Kuzu database file and upsert schema graph nodes/edges."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self._db = kuzu.Database(str(database_path))
        self._conn = kuzu.Connection(self._db)

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        """Close the underlying database handle."""
        self._db.close()

    def ensure_schema(self) -> None:
        """Create node and relationship tables if they do not exist."""
        self._conn.execute(graph_schema.DDL_CREATE_TABLE_NODE)
        self._ensure_schema_table_entity_type_column()
        self._ensure_schema_table_table_type_column()
        self._ensure_schema_table_catalog_enrichment_columns()
        self._ensure_schema_table_classification_columns()
        self._ensure_schema_table_dbt_enrichment_columns()
        self._ensure_schema_table_description_tags_columns()
        self._ensure_schema_table_staleness_columns()
        self._ensure_schema_table_row_count_source_column()
        self._drop_schema_table_legacy_dbt_columns()
        self._conn.execute(graph_schema.DDL_CREATE_COLUMN_NODE)
        self._ensure_schema_column_description_column()
        self._ensure_schema_column_enrichment_columns()
        self._ensure_schema_column_stats_columns()
        self._ensure_schema_column_catalog_enrichment_columns()
        self._ensure_schema_column_nested_columns()
        self._conn.execute(graph_schema.DDL_CREATE_HAS_COLUMN_REL)
        self._conn.execute(graph_schema.DDL_CREATE_HAS_SUBCOLUMN_REL)
        self._conn.execute(graph_schema.DDL_CREATE_ENTITY_NODE)
        self._conn.execute(graph_schema.DDL_CREATE_REPRESENTS_REL)
        self._conn.execute(graph_schema.DDL_CREATE_FK_REL)
        self._ensure_fk_constraint_name_column()
        self._conn.execute(graph_schema.DDL_CREATE_INFERRED_JOIN_REL)
        self._conn.execute(graph_schema.DDL_CREATE_LINEAGE_REL)
        self._conn.execute(graph_schema.DDL_CREATE_SAME_ENTITY_REL)
        self._conn.execute(graph_schema.DDL_CREATE_CLUSTER_NODE)
        self._ensure_cluster_stale_column()
        self._ensure_cluster_schema_pattern_column()
        self._conn.execute(graph_schema.DDL_CREATE_IN_CLUSTER_REL)
        self._conn.execute(graph_schema.DDL_CREATE_JOIN_PATH_NODE)
        self._ensure_join_path_stale_column()
        self._conn.execute(graph_schema.DDL_CREATE_METRIC_TEMPLATE_NODE)
        self._ensure_metric_template_stale_column()
        self._ensure_metric_template_dialect_column()
        self._conn.execute(graph_schema.DDL_CREATE_METRIC_DEPENDS_REL)
        self._conn.execute(graph_schema.DDL_CREATE_SEMANTIC_METRIC_NODE)
        self._conn.execute(graph_schema.DDL_CREATE_SEMANTIC_DIMENSION_NODE)
        self._conn.execute(graph_schema.DDL_CREATE_SEMANTIC_BUSINESS_RULE_NODE)
        self._conn.execute(graph_schema.DDL_CREATE_SEMANTIC_METRIC_DEPENDS_REL)
        self._conn.execute(graph_schema.DDL_CREATE_SEMANTIC_DIMENSION_LEVEL_REL)
        self._conn.execute(graph_schema.DDL_CREATE_SEMANTIC_RULE_APPLIES_TO_REL)

    def _ensure_schema_table_entity_type_column(self) -> None:
        """Add ``entity_type`` to ``SchemaTable`` when upgrading older graph files."""
        try:
            self._conn.execute("ALTER TABLE SchemaTable ADD entity_type STRING")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "entity_type" in message and "already has" in message:
                return
            logger.warning("SchemaTable ALTER for entity_type: %s", exc)

    def _ensure_schema_table_table_type_column(self) -> None:
        """Add ``table_type`` to ``SchemaTable`` when upgrading older graph files."""
        try:
            self._conn.execute("ALTER TABLE SchemaTable ADD table_type STRING")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "table_type" in message and "already has" in message:
                return
            logger.warning("SchemaTable ALTER for table_type: %s", exc)

    def _ensure_schema_table_catalog_enrichment_columns(self) -> None:
        """Add usage, partition, grant, and Snowflake catalog fields for older graphs."""
        for col_name, col_type in (
            ("seq_scan_count", "INT64"),
            ("idx_scan_count", "INT64"),
            ("insert_count", "INT64"),
            ("update_count", "INT64"),
            ("delete_count", "INT64"),
            ("is_partitioned", "BOOL"),
            ("partition_key", "STRING"),
            ("grants_json", "STRING"),
            ("access_read_count", "INT64"),
            ("access_write_count", "INT64"),
            ("days_since_last_access", "INT64"),
            ("potentially_unused", "BOOL"),
            ("table_bytes", "INT64"),
            ("clustering_key", "STRING"),
        ):
            try:
                self._conn.execute(f"ALTER TABLE SchemaTable ADD {col_name} {col_type}")
            except RuntimeError as exc:
                message = str(exc).lower()
                if col_name in message and "already has" in message:
                    continue
                logger.warning("SchemaTable ALTER for %s: %s", col_name, exc)

    def _ensure_schema_table_classification_columns(self) -> None:
        """Add classifier fields for older graph files."""
        for col_name, col_type in (
            ("role", "STRING"),
            ("role_confidence", "DOUBLE"),
            ("classification_signals", "STRING"),
        ):
            try:
                self._conn.execute(f"ALTER TABLE SchemaTable ADD {col_name} {col_type}")
            except RuntimeError as exc:
                message = str(exc).lower()
                if col_name in message and "already has" in message:
                    continue
                logger.warning("SchemaTable ALTER for %s: %s", col_name, exc)

    def _ensure_schema_table_dbt_enrichment_columns(self) -> None:
        """Add dbt semantic-layer fields for older graph files."""
        for col_name, col_type in (
            ("has_external_consumers", "BOOL"),
            ("test_count", "INT64"),
        ):
            try:
                self._conn.execute(f"ALTER TABLE SchemaTable ADD {col_name} {col_type}")
            except RuntimeError as exc:
                message = str(exc).lower()
                if col_name in message and "already has" in message:
                    continue
                logger.warning("SchemaTable ALTER for %s: %s", col_name, exc)

    def _ensure_schema_table_row_count_source_column(self) -> None:
        """Add ``row_count_source`` for older graph files (view-count provenance)."""
        try:
            self._conn.execute("ALTER TABLE SchemaTable ADD row_count_source STRING")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "row_count_source" in message and "already has" in message:
                return
            logger.warning("SchemaTable ALTER for row_count_source: %s", exc)

    def _ensure_schema_table_staleness_columns(self) -> None:
        """Add dbt source-freshness fields (``staleness_status``/``staleness_as_of``)."""
        for col_name, col_type in (
            ("staleness_status", "STRING"),
            ("staleness_as_of", "STRING"),
        ):
            try:
                self._conn.execute(f"ALTER TABLE SchemaTable ADD {col_name} {col_type}")
            except RuntimeError as exc:
                message = str(exc).lower()
                if col_name in message and "already has" in message:
                    continue
                logger.warning("SchemaTable ALTER for %s: %s", col_name, exc)

    def _drop_schema_table_legacy_dbt_columns(self) -> None:
        """Drop legacy duplicates ``dbt_description`` and ``tags_json``.

        These were superseded by the first-class ``description`` / ``tags``
        fields added in the dbt enrichment update. Older graph files that were written before
        the consolidation may still have them; silently drop if present.
        """
        for col_name in ("dbt_description", "tags_json"):
            try:
                self._conn.execute(f"ALTER TABLE SchemaTable DROP {col_name}")
            except RuntimeError as exc:
                message = str(exc).lower()
                if "does not have" in message or "does not exist" in message:
                    continue
                logger.debug("SchemaTable DROP for %s: %s", col_name, exc)

    def _ensure_schema_table_description_tags_columns(self) -> None:
        """Add first-class ``description`` and ``tags`` for dbt / LLM context."""
        for col_name, col_type in (
            ("description", "STRING"),
            ("tags", "STRING[]"),
        ):
            try:
                self._conn.execute(f"ALTER TABLE SchemaTable ADD {col_name} {col_type}")
            except RuntimeError as exc:
                message = str(exc).lower()
                if col_name in message and "already has" in message:
                    continue
                logger.warning("SchemaTable ALTER for %s: %s", col_name, exc)

    def _ensure_schema_column_description_column(self) -> None:
        """Add ``description`` on ``SchemaColumn`` for dbt column docs."""
        try:
            self._conn.execute("ALTER TABLE SchemaColumn ADD description STRING")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "description" in message and "already has" in message:
                return
            logger.warning("SchemaColumn ALTER for description: %s", exc)

    def _ensure_schema_column_enrichment_columns(self) -> None:
        """Add column comment / constraint fields for older Kuzu files."""
        for col_name, col_type in (
            ("comment", "STRING"),
            ("default_value", "STRING"),
            ("is_indexed", "BOOL"),
            ("check_constraints_json", "STRING"),
            ("ordinal_position", "INT64"),
        ):
            try:
                self._conn.execute(
                    f"ALTER TABLE SchemaColumn ADD {col_name} {col_type}"
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if col_name in message and "already has" in message:
                    continue
                logger.warning("SchemaColumn ALTER for %s: %s", col_name, exc)

    def _ensure_schema_column_stats_columns(self) -> None:
        """Add planner statistics mirror fields for older Kuzu files."""
        for col_name, col_type in (
            ("most_common_values_json", "STRING"),
            ("histogram_bounds_json", "STRING"),
            ("stats_correlation", "DOUBLE"),
        ):
            try:
                self._conn.execute(
                    f"ALTER TABLE SchemaColumn ADD {col_name} {col_type}"
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if col_name in message and "already has" in message:
                    continue
                logger.warning("SchemaColumn ALTER for %s: %s", col_name, exc)

    def _ensure_schema_column_catalog_enrichment_columns(self) -> None:
        """Add catalog cardinality and index-detail fields for older Kuzu files."""
        for col_name, col_type in (
            ("column_cardinality", "INT64"),
            ("index_type", "STRING"),
            ("index_is_unique", "BOOL"),
        ):
            try:
                self._conn.execute(
                    f"ALTER TABLE SchemaColumn ADD {col_name} {col_type}"
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if col_name in message and "already has" in message:
                    continue
                logger.warning("SchemaColumn ALTER for %s: %s", col_name, exc)

    def _ensure_schema_column_nested_columns(self) -> None:
        """Add nested-column linkage fields for older Kuzu files."""
        for col_name, col_type in (
            ("parent_column_id", "STRING"),
            ("is_array", "BOOL"),
        ):
            try:
                self._conn.execute(
                    f"ALTER TABLE SchemaColumn ADD {col_name} {col_type}"
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if col_name in message and "already has" in message:
                    continue
                logger.warning("SchemaColumn ALTER for %s: %s", col_name, exc)

    def _ensure_fk_constraint_name_column(self) -> None:
        """Add ``constraint_name`` to ``FK_REFERENCES`` when upgrading older graph files."""
        try:
            self._conn.execute(
                "ALTER TABLE FK_REFERENCES ADD constraint_name STRING"
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "constraint_name" in message and "already has" in message:
                return
            logger.warning("FK_REFERENCES ALTER for constraint_name: %s", exc)

    def _ensure_cluster_stale_column(self) -> None:
        """Add ``stale`` to ``Cluster`` when upgrading older graph files."""
        try:
            self._conn.execute("ALTER TABLE Cluster ADD stale BOOL")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "stale" in message and "already has" in message:
                return
            logger.warning("Cluster ALTER for stale: %s", exc)

    def _ensure_cluster_schema_pattern_column(self) -> None:
        """Add ``schema_pattern`` to ``Cluster`` when upgrading older graph files."""
        try:
            self._conn.execute("ALTER TABLE Cluster ADD schema_pattern STRING")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "schema_pattern" in message and "already has" in message:
                return
            logger.warning("Cluster ALTER for schema_pattern: %s", exc)

    def _ensure_join_path_stale_column(self) -> None:
        """Add ``stale`` to ``JoinPath`` when upgrading older graph files."""
        try:
            self._conn.execute("ALTER TABLE JoinPath ADD stale BOOL")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "stale" in message and "already has" in message:
                return
            logger.warning("JoinPath ALTER for stale: %s", exc)

    def _ensure_metric_template_stale_column(self) -> None:
        """Add ``stale`` to ``MetricTemplate`` when upgrading older graph files."""
        try:
            self._conn.execute("ALTER TABLE MetricTemplate ADD stale BOOL")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "stale" in message and "already has" in message:
                return
            logger.warning("MetricTemplate ALTER for stale: %s", exc)

    def _ensure_metric_template_dialect_column(self) -> None:
        """Add ``dialect`` to ``MetricTemplate`` when upgrading older graph files."""
        try:
            self._conn.execute("ALTER TABLE MetricTemplate ADD dialect STRING")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "dialect" in message and "already has" in message:
                return
            logger.warning("MetricTemplate ALTER for dialect: %s", exc)

    def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a write (mutation) Cypher query. Use this for DELETE / SET operations."""
        with log_timed_operation(
            logger,
            event="graph.execute_write",
            level=logging.DEBUG,
            has_params=bool(params),
            query_preview=query.strip().splitlines()[0][:120] if query.strip() else "",
            graph_path=str(self._path),
        ):
            if params:
                self._conn.execute(query, params)
            else:
                self._conn.execute(query)

    def clear_graph(self) -> None:
        """Remove all FK edges and table nodes (full re-index)."""
        self._conn.execute("MATCH ()-[r:METRIC_DEPENDS]->() DELETE r")
        self._conn.execute("MATCH (m:MetricTemplate) DELETE m")
        self._conn.execute("MATCH ()-[r:IN_CLUSTER]->() DELETE r")
        self._conn.execute("MATCH (c:Cluster) DELETE c")
        self._conn.execute("MATCH (j:JoinPath) DELETE j")
        self._conn.execute("MATCH ()-[r:FK_REFERENCES]->() DELETE r")
        self._conn.execute("MATCH ()-[r:INFERRED_JOIN]->() DELETE r")
        self._conn.execute("MATCH ()-[r:LINEAGE]->() DELETE r")
        self._conn.execute("MATCH ()-[r:SAME_ENTITY]->() DELETE r")
        self._conn.execute("MATCH ()-[r:HAS_SUBCOLUMN]->() DELETE r")
        self._conn.execute("MATCH ()-[r:HAS_COLUMN]->() DELETE r")
        self._conn.execute("MATCH (col:SchemaColumn) DELETE col")
        self._conn.execute("MATCH ()-[r:REPRESENTS]->() DELETE r")
        self._conn.execute("MATCH (e:Entity) DELETE e")
        self._conn.execute("MATCH (t:SchemaTable) DELETE t")

    def clear_connection_subgraph(self, connection_name: str) -> None:
        """Remove all nodes and edges scoped to one indexed connection (unified graph merge)."""
        cn = connection_name
        self._conn.execute(
            """
            MATCH (e1:Entity)-[r:SAME_ENTITY]->(e2:Entity)
            WHERE e1.connection_name = $cn OR e2.connection_name = $cn
            DELETE r
            """,
            {"cn": cn},
        )
        self._conn.execute(
            """
            MATCH (t:SchemaTable {connection_name: $cn}), (p:JoinPath)
            WHERE p.from_table_id = t.node_id OR p.to_table_id = t.node_id
            DELETE p
            """,
            {"cn": cn},
        )
        self._conn.execute(
            """
            MATCH (m:MetricTemplate {connection_name: $cn})
            DETACH DELETE m
            """,
            {"cn": cn},
        )
        self._conn.execute(
            """
            MATCH (t:SchemaTable {connection_name: $cn})-[r:IN_CLUSTER]->()
            DELETE r
            """,
            {"cn": cn},
        )
        self._conn.execute(
            """
            MATCH (a:SchemaTable)-[r:FK_REFERENCES]->(b:SchemaTable)
            WHERE a.connection_name = $cn OR b.connection_name = $cn
            DELETE r
            """,
            {"cn": cn},
        )
        self._conn.execute(
            """
            MATCH (a:SchemaTable)-[r:INFERRED_JOIN]->(b:SchemaTable)
            WHERE a.connection_name = $cn OR b.connection_name = $cn
            DELETE r
            """,
            {"cn": cn},
        )
        self._conn.execute(
            """
            MATCH (a:SchemaTable)-[r:LINEAGE]->(b:SchemaTable)
            WHERE a.connection_name = $cn OR b.connection_name = $cn
            DELETE r
            """,
            {"cn": cn},
        )
        self._conn.execute(
            """
            MATCH (e:Entity {connection_name: $cn})-[r:REPRESENTS]->()
            DELETE r
            """,
            {"cn": cn},
        )
        self._conn.execute(
            "MATCH (e:Entity {connection_name: $cn}) DELETE e",
            {"cn": cn},
        )
        self._conn.execute(
            """
            MATCH (col:SchemaColumn {connection_name: $cn})
            DETACH DELETE col
            """,
            {"cn": cn},
        )
        self._conn.execute(
            "MATCH (t:SchemaTable {connection_name: $cn}) DETACH DELETE t",
            {"cn": cn},
        )

    def upsert_table(self, node: GraphNode) -> None:
        """Insert or update a single ``SchemaTable`` node."""
        assignments = [
            "t.connection_name = $connection_name",
            "t.database = $database",
            "t.schema_name = $schema_name",
            "t.table_name = $table_name",
            "t.row_count = $row_count",
            "t.row_count_source = $row_count_source",
            "t.comment = $comment",
            "t.description = CASE WHEN $apply_tbl_desc THEN $tbl_description ELSE t.description END",
            "t.entity_type = $entity_type",
            "t.table_type = $table_type",
            "t.seq_scan_count = $seq_scan_count",
            "t.idx_scan_count = $idx_scan_count",
            "t.insert_count = $insert_count",
            "t.update_count = $update_count",
            "t.delete_count = $delete_count",
            "t.is_partitioned = $is_partitioned",
            "t.partition_key = $partition_key",
            "t.grants_json = $grants_json",
            "t.access_read_count = $access_read_count",
            "t.access_write_count = $access_write_count",
            "t.days_since_last_access = $days_since_last_access",
            "t.potentially_unused = $potentially_unused",
            "t.table_bytes = $table_bytes",
            "t.clustering_key = $clustering_key",
        ]
        params: dict[str, Any] = {
            "node_id": node.node_id,
            "connection_name": node.connection_name,
            "database": node.database,
            "schema_name": node.schema_name,
            "table_name": node.table_name,
            "row_count": node.row_count,
            "row_count_source": node.row_count_source,
            "comment": node.comment,
            "apply_tbl_desc": node.description is not None,
            "tbl_description": node.description if node.description is not None else "",
            "entity_type": node.entity_type,
            "table_type": node.table_type,
            "seq_scan_count": node.seq_scan_count,
            "idx_scan_count": node.idx_scan_count,
            "insert_count": node.insert_count,
            "update_count": node.update_count,
            "delete_count": node.delete_count,
            "is_partitioned": node.is_partitioned,
            "partition_key": node.partition_key,
            "grants_json": node.grants_json,
            "access_read_count": node.access_read_count,
            "access_write_count": node.access_write_count,
            "days_since_last_access": node.days_since_last_access,
            "potentially_unused": node.potentially_unused,
            "table_bytes": node.table_bytes,
            "clustering_key": node.clustering_key,
        }
        if node.tags is not None:
            assignments.append(
                "t.tags = CASE WHEN $apply_tags THEN $tags ELSE t.tags END"
            )
            params["apply_tags"] = True
            params["tags"] = node.tags
        if node.has_external_consumers is not None:
            assignments.append("t.has_external_consumers = $has_external_consumers")
            params["has_external_consumers"] = node.has_external_consumers
        if node.test_count is not None:
            assignments.append("t.test_count = $test_count")
            params["test_count"] = node.test_count
        set_clause = ",\n                ".join(assignments)
        self._conn.execute(
            f"""
            MERGE (t:SchemaTable {{node_id: $node_id}})
            SET {set_clause}
            """,
            params,
        )

    def upsert_entity(self, node: EntityNode) -> None:
        """Insert or update a single ``Entity`` node."""
        self._conn.execute(
            """
            MERGE (e:Entity {node_id: $node_id})
            SET e.connection_name = $connection_name,
                e.database = $database,
                e.name = $name,
                e.description = $description
            """,
            {
                "node_id": node.node_id,
                "connection_name": node.connection_name,
                "database": node.database,
                "name": node.name,
                "description": node.description,
            },
        )

    def upsert_represents(self, entity_node_id: str, table_node_id: str) -> None:
        """Link an entity to a table (idempotent; one edge per pair)."""
        self._conn.execute(
            """
            MATCH (e:Entity {node_id: $eid}), (t:SchemaTable {node_id: $tid})
            MERGE (e)-[:REPRESENTS]->(t)
            """,
            {"eid": entity_node_id, "tid": table_node_id},
        )

    def set_table_entity_type(self, table_node_id: str, entity_type: str) -> None:
        """Set the business entity label on a ``SchemaTable`` node."""
        self._conn.execute(
            """
            MATCH (t:SchemaTable {node_id: $tid})
            SET t.entity_type = $etype
            """,
            {"tid": table_node_id, "etype": entity_type},
        )

    def set_table_classification(
        self,
        table_node_id: str,
        *,
        role: str,
        role_confidence: float,
        classification_signals_json: str,
    ) -> None:
        """Persist classifier output on a ``SchemaTable`` node."""
        self._conn.execute(
            """
            MATCH (t:SchemaTable {node_id: $tid})
            SET t.role = $role,
                t.role_confidence = $rc,
                t.classification_signals = $sig
            """,
            {
                "tid": table_node_id,
                "role": role,
                "rc": role_confidence,
                "sig": classification_signals_json,
            },
        )

    def upsert_fk_edge(self, edge: GraphEdge) -> None:
        """Insert a foreign-key relationship edge (assumes endpoints exist)."""
        self.merge_fk_edge(edge)

    def merge_fk_edge(self, edge: GraphEdge) -> None:
        """Idempotent FK edge upsert by ``edge_id`` (avoids duplicates on reindex)."""
        self._conn.execute(
            """
            MATCH (src:SchemaTable {node_id: $src}), (dst:SchemaTable {node_id: $dst})
            MERGE (src)-[r:FK_REFERENCES {edge_id: $edge_id}]->(dst)
            SET r.source_column = $source_column,
                r.target_column = $target_column,
                r.constraint_name = $constraint_name
            """,
            {
                "src": edge.source_node_id,
                "dst": edge.target_node_id,
                "edge_id": edge.edge_id,
                "source_column": edge.source_column,
                "target_column": edge.target_column,
                "constraint_name": edge.constraint_name,
            },
        )

    def upsert_lineage_edge(self, edge: LineageEdge) -> None:
        """Insert or update a ``LINEAGE`` edge (idempotent by ``edge_id``)."""
        self._conn.execute(
            """
            MATCH (src:SchemaTable {node_id: $src}), (dst:SchemaTable {node_id: $dst})
            MERGE (src)-[r:LINEAGE {edge_id: $edge_id}]->(dst)
            SET r.source = $source,
                r.lineage_type = $lineage_type,
                r.confidence = $confidence
            """,
            {
                "src": edge.source_node_id,
                "dst": edge.target_node_id,
                "edge_id": edge.edge_id,
                "source": edge.source,
                "lineage_type": edge.lineage_type,
                "confidence": edge.confidence,
            },
        )

    def clear_lineage_edges(self, connection_name: str) -> None:
        """Remove all ``LINEAGE`` edges touching tables on one connection."""
        self._conn.execute(
            """
            MATCH (a:SchemaTable)-[r:LINEAGE]->(b:SchemaTable)
            WHERE a.connection_name = $cn OR b.connection_name = $cn
            DELETE r
            """,
            {"cn": connection_name},
        )

    def clear_dbt_model_dependency_lineage(self, connection_name: str) -> None:
        """Remove dbt ``parent_map`` lineage edges for one connection (see enrichment dbt)."""
        self._conn.execute(
            """
            MATCH (a:SchemaTable)-[r:LINEAGE]->(b:SchemaTable)
            WHERE a.connection_name = $cn AND b.connection_name = $cn
              AND r.source = 'dbt' AND r.lineage_type = 'model_dependency'
            DELETE r
            """,
            {"cn": connection_name},
        )

    def clear_intelligence_artifacts(self) -> None:
        """Remove clusters and precomputed join paths (keeps tables and FK edges)."""
        self._conn.execute("MATCH ()-[r:IN_CLUSTER]->() DELETE r")
        self._conn.execute("MATCH (c:Cluster) DELETE c")
        self._conn.execute("MATCH (j:JoinPath) DELETE j")

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
        checks_json = json.dumps(check_constraints or [], ensure_ascii=False)
        apply_desc = description is not None
        self._conn.execute(
            """
            MERGE (col:SchemaColumn {node_id: $cid})
            SET col.connection_name = $cn,
                col.database = $db,
                col.schema_name = $sn,
                col.table_name = $tn,
                col.column_name = $colname,
                col.description = CASE WHEN $apply_desc THEN $cdesc ELSE col.description END,
                col.data_type = $dtype,
                col.nullable = $nullable,
                col.is_primary_key = $pk,
                col.is_foreign_key = $fk,
                col.comment = $ccomment,
                col.default_value = $cdefault,
                col.is_indexed = $cindexed,
                col.check_constraints_json = $cchecks,
                col.ordinal_position = $ord,
                col.most_common_values_json = $mcv,
                col.histogram_bounds_json = $hb,
                col.stats_correlation = $scorr,
                col.column_cardinality = $cardinality,
                col.index_type = $itype,
                col.index_is_unique = $iunique,
                col.parent_column_id = $pcid,
                col.is_array = $is_arr
            """,
            {
                "cid": column_node_id,
                "cn": connection_name,
                "db": database,
                "sn": schema_name,
                "tn": table_name,
                "colname": column_name,
                "apply_desc": apply_desc,
                "cdesc": description if description is not None else "",
                "dtype": data_type,
                "nullable": nullable,
                "pk": is_primary_key,
                "fk": is_foreign_key,
                "ccomment": comment,
                "cdefault": default_value,
                "cindexed": is_indexed,
                "cchecks": checks_json,
                "ord": ordinal_position,
                "mcv": most_common_values_json,
                "hb": histogram_bounds_json,
                "scorr": stats_correlation,
                "cardinality": column_cardinality,
                "itype": index_type,
                "iunique": index_is_unique,
                "pcid": parent_column_id,
                "is_arr": is_array,
            },
        )
        if parent_column_id is None:
            self._conn.execute(
                """
                MATCH (t:SchemaTable {node_id: $tid}), (col:SchemaColumn {node_id: $cid})
                MERGE (t)-[:HAS_COLUMN]->(col)
                """,
                {"tid": table_node_id, "cid": column_node_id},
            )
        else:
            self._conn.execute(
                """
                MATCH (parent:SchemaColumn {node_id: $pid}), (col:SchemaColumn {node_id: $cid})
                MERGE (parent)-[:HAS_SUBCOLUMN]->(col)
                """,
                {"pid": parent_column_id, "cid": column_node_id},
            )

    def delete_column_node(self, column_node_id: str) -> None:
        """Remove a column node (``HAS_COLUMN`` edges removed by cascade)."""
        self._conn.execute(
            "MATCH (col:SchemaColumn {node_id: $cid}) DETACH DELETE col",
            {"cid": column_node_id},
        )

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
        tables_json = json.dumps(tables_used, ensure_ascii=False)
        errors_json = json.dumps(validation_errors, ensure_ascii=False)
        self._conn.execute(
            """
            MATCH (m:MetricTemplate {node_id: $mid})-[r:METRIC_DEPENDS]->()
            DELETE r
            """,
            {"mid": node_id},
        )
        self._conn.execute(
            """
            MERGE (m:MetricTemplate {node_id: $mid})
            SET m.connection_name = $cn,
                m.database = $db,
                m.dialect = $dialect,
                m.name = $name,
                m.display_name = $dname,
                m.description = $mdesc,
                m.sql_template = $sql,
                m.tables_used_json = $tables,
                m.validated = $ok,
                m.validation_errors_json = $errs,
                m.generated_at_iso = $gen,
                m.stale = $stale
            """,
            {
                "mid": node_id,
                "cn": connection_name,
                "db": database,
                "dialect": dialect,
                "name": name,
                "dname": display_name,
                "mdesc": description,
                "sql": sql_template,
                "tables": tables_json,
                "ok": validated,
                "errs": errors_json,
                "gen": generated_at_iso,
                "stale": stale,
            },
        )
        for tid in depends_on_table_node_ids:
            self._conn.execute(
                """
                MATCH (m:MetricTemplate {node_id: $mid}), (t:SchemaTable {node_id: $tid})
                MERGE (m)-[:METRIC_DEPENDS]->(t)
                """,
                {"mid": node_id, "tid": tid},
            )

    def delete_fk_edges_touching_column(
        self, table_node_id: str, column_name: str
    ) -> int:
        """Delete FK edges where the column appears as source or target. Returns count."""
        rows = self.query_all_rows(
            """
            MATCH (src:SchemaTable)-[r:FK_REFERENCES]->(dst:SchemaTable)
            WHERE (src.node_id = $tid AND r.source_column = $col)
               OR (dst.node_id = $tid AND r.target_column = $col)
            RETURN count(*)
            """,
            {"tid": table_node_id, "col": column_name},
        )
        n = int(rows[0][0]) if rows and rows[0][0] is not None else 0
        self._conn.execute(
            """
            MATCH (src:SchemaTable)-[r:FK_REFERENCES]->(dst:SchemaTable)
            WHERE (src.node_id = $tid AND r.source_column = $col)
               OR (dst.node_id = $tid AND r.target_column = $col)
            DELETE r
            """,
            {"tid": table_node_id, "col": column_name},
        )
        return n

    def delete_inferred_joins_touching_column(
        self, table_node_id: str, column_name: str
    ) -> int:
        """Delete inferred join edges touching a column on either endpoint."""
        rows = self.query_all_rows(
            """
            MATCH (src:SchemaTable)-[r:INFERRED_JOIN]->(dst:SchemaTable)
            WHERE (src.node_id = $tid AND r.source_column = $col)
               OR (dst.node_id = $tid AND r.target_column = $col)
            RETURN count(*)
            """,
            {"tid": table_node_id, "col": column_name},
        )
        n = int(rows[0][0]) if rows and rows[0][0] is not None else 0
        self._conn.execute(
            """
            MATCH (src:SchemaTable)-[r:INFERRED_JOIN]->(dst:SchemaTable)
            WHERE (src.node_id = $tid AND r.source_column = $col)
               OR (dst.node_id = $tid AND r.target_column = $col)
            DELETE r
            """,
            {"tid": table_node_id, "col": column_name},
        )
        return n

    def delete_table_node_cascade(self, table_node_id: str) -> None:
        """Remove a table and its column nodes; caller should handle intelligence cleanup."""
        self._conn.execute(
            """
            MATCH (t:SchemaTable {node_id: $tid})-[:HAS_COLUMN]->(col:SchemaColumn)
            DETACH DELETE col
            """,
            {"tid": table_node_id},
        )
        self._conn.execute(
            "MATCH (t:SchemaTable {node_id: $tid}) DETACH DELETE t",
            {"tid": table_node_id},
        )

    def mark_clusters_stale_for_table(
        self, table_node_id: str, database_key: str
    ) -> None:
        """Set ``stale`` on clusters linked to the table."""
        self._conn.execute(
            """
            MATCH (t:SchemaTable {node_id: $tid})-[:IN_CLUSTER]->(c:Cluster)
            WHERE c.database_key = $db
            SET c.stale = true
            """,
            {"tid": table_node_id, "db": database_key},
        )

    def remove_table_from_clusters(self, table_node_id: str) -> None:
        """Drop ``IN_CLUSTER`` edges for a table (cluster nodes kept)."""
        self._conn.execute(
            """
            MATCH (t:SchemaTable {node_id: $tid})-[r:IN_CLUSTER]->()
            DELETE r
            """,
            {"tid": table_node_id},
        )

    def mark_join_paths_stale_for_table(
        self, table_node_id: str, database_key: str
    ) -> None:
        """Flag precomputed paths that start or end at this table."""
        self._conn.execute(
            """
            MATCH (p:JoinPath)
            WHERE p.database_key = $db
              AND (p.from_table_id = $tid OR p.to_table_id = $tid)
            SET p.stale = true
            """,
            {"db": database_key, "tid": table_node_id},
        )

    def delete_join_paths_for_table(
        self, table_node_id: str, database_key: str
    ) -> None:
        """Remove precomputed paths touching a table (used when removing a table)."""
        self._conn.execute(
            """
            MATCH (p:JoinPath)
            WHERE p.database_key = $db
              AND (p.from_table_id = $tid OR p.to_table_id = $tid)
            DELETE p
            """,
            {"db": database_key, "tid": table_node_id},
        )

    def upsert_inferred_join(self, candidate: RelationshipCandidate) -> None:
        """Insert an inferred join edge (assumes endpoint table nodes exist)."""
        self._conn.execute(
            """
            MATCH (src:SchemaTable {node_id: $src}), (dst:SchemaTable {node_id: $dst})
            CREATE (src)-[:INFERRED_JOIN {
                edge_id: $edge_id,
                source_column: $source_column,
                target_column: $target_column,
                source: $source,
                confidence: $confidence,
                reasoning: $reasoning
            }]->(dst)
            """,
            {
                "src": candidate.source_node_id,
                "dst": candidate.target_node_id,
                "edge_id": candidate.candidate_id,
                "source_column": candidate.source_column,
                "target_column": candidate.target_column,
                "source": candidate.source,
                "confidence": candidate.confidence,
                "reasoning": candidate.reasoning,
            },
        )

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
        self._conn.execute(
            """
            MERGE (c:Cluster {node_id: $node_id})
            SET c.database_key = $database_key,
                c.label = $label,
                c.description = $description,
                c.cohesion_score = $cohesion_score,
                c.table_count = $table_count,
                c.stale = $stale,
                c.schema_pattern = $schema_pattern
            """,
            {
                "node_id": node_id,
                "database_key": database_key,
                "label": label,
                "description": description,
                "cohesion_score": cohesion_score,
                "table_count": table_count,
                "stale": stale,
                "schema_pattern": schema_pattern or "unknown",
            },
        )

    def upsert_in_cluster(self, table_node_id: str, cluster_node_id: str) -> None:
        """Link a table to a cluster (idempotent)."""
        self._conn.execute(
            """
            MATCH (t:SchemaTable {node_id: $tid}), (c:Cluster {node_id: $cid})
            MERGE (t)-[:IN_CLUSTER]->(c)
            """,
            {"tid": table_node_id, "cid": cluster_node_id},
        )

    def set_cluster_schema_pattern(
        self, cluster_node_id: str, schema_pattern: str
    ) -> None:
        """Set ``schema_pattern`` on an existing ``Cluster`` node."""
        self._conn.execute(
            """
            MATCH (c:Cluster {node_id: $cid})
            SET c.schema_pattern = $sp
            """,
            {"cid": cluster_node_id, "sp": schema_pattern},
        )

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
        self._conn.execute(
            """
            MERGE (p:JoinPath {node_id: $node_id})
            SET p.database_key = $database_key,
                p.from_table_id = $from_table_id,
                p.to_table_id = $to_table_id,
                p.depth = $depth,
                p.confidence = $confidence,
                p.ambiguous = $ambiguous,
                p.steps_json = $steps_json,
                p.semantic_label = $semantic_label,
                p.stale = $stale
            """,
            {
                "node_id": node_id,
                "database_key": database_key,
                "from_table_id": from_table_id,
                "to_table_id": to_table_id,
                "depth": depth,
                "confidence": confidence,
                "ambiguous": ambiguous,
                "steps_json": steps_json,
                "semantic_label": semantic_label,
                "stale": stale,
            },
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
        self._conn.execute(
            """
            MATCH (a:Entity {node_id: $a}), (b:Entity {node_id: $b})
            MERGE (a)-[r:SAME_ENTITY {edge_id: $eid}]->(b)
            SET r.status = $status,
                r.score = $score,
                r.join_columns = $join_columns,
                r.reasoning = $reasoning,
                r.created_at = $created_at,
                r.confirmed_at = $confirmed_at,
                r.confirmed_by = $confirmed_by
            """,
            {
                "a": from_entity_node_id,
                "b": to_entity_node_id,
                "eid": edge_id,
                "status": status,
                "score": score,
                "join_columns": join_columns,
                "reasoning": reasoning,
                "created_at": created_at,
                "confirmed_at": confirmed_at,
                "confirmed_by": confirmed_by,
            },
        )

    def delete_same_entity_edge(self, edge_id: str) -> None:
        """Remove a ``SAME_ENTITY`` edge by ``edge_id``."""
        self._conn.execute(
            """
            MATCH ()-[r:SAME_ENTITY {edge_id: $eid}]->()
            DELETE r
            """,
            {"eid": edge_id},
        )

    def same_entity_edge_endpoints(self, edge_id: str) -> tuple[str, str] | None:
        """Return ``(from_entity_node_id, to_entity_node_id)`` for an existing edge."""
        rows = self.query_all_rows(
            """
            MATCH (a:Entity)-[r:SAME_ENTITY {edge_id: $eid}]->(b:Entity)
            RETURN a.node_id, b.node_id
            """,
            {"eid": edge_id},
        )
        if not rows:
            return None
        return str(rows[0][0]), str(rows[0][1])

    def list_same_entity_edges(
        self, *, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Return same-entity links with table context for CLI and MCP."""
        if status is None:
            rows = self.query_all_rows(
                """
                MATCH (a:Entity)-[r:SAME_ENTITY]->(b:Entity)
                OPTIONAL MATCH (a)-[:REPRESENTS]->(ta:SchemaTable)
                OPTIONAL MATCH (b)-[:REPRESENTS]->(tb:SchemaTable)
                RETURN r.edge_id, r.status, r.score, r.join_columns, r.reasoning,
                       r.created_at, r.confirmed_at, r.confirmed_by,
                       a.node_id, a.connection_name, a.database, a.name,
                       b.node_id, b.connection_name, b.database, b.name,
                       ta.schema_name, ta.table_name, tb.schema_name, tb.table_name
                """
            )
        else:
            rows = self.query_all_rows(
                """
                MATCH (a:Entity)-[r:SAME_ENTITY]->(b:Entity)
                WHERE r.status = $st
                OPTIONAL MATCH (a)-[:REPRESENTS]->(ta:SchemaTable)
                OPTIONAL MATCH (b)-[:REPRESENTS]->(tb:SchemaTable)
                RETURN r.edge_id, r.status, r.score, r.join_columns, r.reasoning,
                       r.created_at, r.confirmed_at, r.confirmed_by,
                       a.node_id, a.connection_name, a.database, a.name,
                       b.node_id, b.connection_name, b.database, b.name,
                       ta.schema_name, ta.table_name, tb.schema_name, tb.table_name
                """,
                {"st": status},
            )
        by_eid: dict[str, dict[str, Any]] = {}
        for row in rows:
            (
                eid,
                st,
                score,
                join_cols,
                reason,
                created,
                conf_at,
                conf_by,
                aid,
                acn,
                adb,
                aname,
                bid,
                bcn,
                bdb,
                bname,
                tas,
                tat,
                tbs,
                tbt,
            ) = row
            key = str(eid) if eid is not None else ""
            if key in by_eid:
                continue
            by_eid[key] = {
                "edge_id": key,
                "status": str(st) if st is not None else "",
                "score": float(score) if score is not None else 0.0,
                "join_columns": str(join_cols) if join_cols is not None else None,
                "reasoning": str(reason) if reason is not None else None,
                "created_at": str(created) if created is not None else "",
                "confirmed_at": str(conf_at) if conf_at is not None else None,
                "confirmed_by": str(conf_by) if conf_by is not None else None,
                "entity_a_id": str(aid) if aid is not None else "",
                "entity_a_connection": str(acn) if acn is not None else "",
                "entity_a_database": str(adb) if adb is not None else "",
                "entity_a_name": str(aname) if aname is not None else "",
                "entity_b_id": str(bid) if bid is not None else "",
                "entity_b_connection": str(bcn) if bcn is not None else "",
                "entity_b_database": str(bdb) if bdb is not None else "",
                "entity_b_name": str(bname) if bname is not None else "",
                "table_a_schema": str(tas) if tas is not None else None,
                "table_a_name": str(tat) if tat is not None else None,
                "table_b_schema": str(tbs) if tbs is not None else None,
                "table_b_name": str(tbt) if tbt is not None else None,
            }
        return list(by_eid.values())

    def list_entities_with_primary_table(
        self, connection_name: str
    ) -> list[tuple[str, str, str]]:
        """Return ``(entity_name, schema.table, entity_node_id)`` per entity."""
        rows = self.query_all_rows(
            """
            MATCH (e:Entity {connection_name: $cn})-[:REPRESENTS]->(t:SchemaTable)
            RETURN e.name, t.schema_name, t.table_name, e.node_id
            ORDER BY e.name, t.schema_name, t.table_name
            """,
            {"cn": connection_name},
        )
        by_entity: dict[str, tuple[str, str, str]] = {}
        for name, sn, tn, eid in rows:
            eid_s = str(eid)
            if eid_s in by_entity:
                continue
            qn = f"{sn}.{tn}"
            by_entity[eid_s] = (str(name), qn, eid_s)
        return list(by_entity.values())

    def table_node_id_for_entity(self, entity_node_id: str) -> str | None:
        """First ``SchemaTable`` linked by ``REPRESENTS`` (ordered by name)."""
        rows = self.query_all_rows(
            """
            MATCH (e:Entity {node_id: $eid})-[:REPRESENTS]->(t:SchemaTable)
            RETURN t.node_id
            ORDER BY t.schema_name, t.table_name
            LIMIT 1
            """,
            {"eid": entity_node_id},
        )
        if not rows:
            return None
        return str(rows[0][0]) if rows[0][0] is not None else None

    def columns_for_table(
        self, table_node_id: str
    ) -> list[tuple[str, str, bool, bool]]:
        """Return ``(column_name, data_type, nullable, is_pk)`` rows."""
        rows = self.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $tid})-[:HAS_COLUMN]->(c:SchemaColumn)
            RETURN c.column_name, c.data_type, c.nullable, c.is_primary_key
            ORDER BY c.column_name
            """,
            {"tid": table_node_id},
        )
        out: list[tuple[str, str, bool, bool]] = []
        for a, b, c, d in rows:
            out.append(
                (
                    str(a or ""),
                    str(b or ""),
                    bool(c),
                    bool(d),
                )
            )
        return out

    def count_same_entity_by_status(self, status: str) -> int:
        """Count ``SAME_ENTITY`` edges with the given status."""
        rows = self.query_all_rows(
            """
            MATCH ()-[r:SAME_ENTITY]->()
            WHERE r.status = $st
            RETURN count(*)
            """,
            {"st": status},
        )
        if not rows or rows[0][0] is None:
            return 0
        return int(rows[0][0])

    def execute(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> kuzu.QueryResult | list[kuzu.QueryResult]:
        """Run a read query (or arbitrary Cypher) with optional parameters."""
        return self._conn.execute(cypher, parameters)

    def query_all_rows(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute Cypher and materialize all result rows as tuples.

        Keeps callers (MCP, search index) off raw ``kuzu.QueryResult`` handles
        while still routing reads through :class:`KuzuStore`.
        """
        result = self.execute(cypher, parameters)
        if isinstance(result, list):
            raise TypeError("Expected a single QueryResult from query_all_rows")
        rows: list[tuple[Any, ...]] = []
        while result.has_next():
            nxt = result.get_next()
            rows.append(tuple(nxt) if not isinstance(nxt, tuple) else nxt)
        return rows
