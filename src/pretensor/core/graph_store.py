"""Graph node/edge/intelligence persistence and reads extracted from KuzuStore."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any, NamedTuple

from pretensor.core.graph_schema_manager import _SCHEMA_TABLE_EMBEDDING_DIM
from pretensor.core.query_runner import QueryRunner
from pretensor.graph_models.consumer import ConsumesEdge, ExternalConsumerNode
from pretensor.graph_models.edge import GraphEdge, LineageEdge
from pretensor.graph_models.entity import EntityNode
from pretensor.graph_models.node import GraphNode
from pretensor.graph_models.relationship import RelationshipCandidate

logger = logging.getLogger(__name__)


class TableEmbeddingRow(NamedTuple):
    """One ``SchemaTable`` row for embedding-based retrieval.

    Yielded by :meth:`GraphStore.iter_table_embeddings`; includes rows where
    ``embedding is None`` so callers can detect the "no tables carry vectors"
    fallback without a second store round-trip.
    """

    node_id: str
    connection_name: str
    database: str
    schema_name: str
    table_name: str
    description: str
    embedding: list[float] | None
    cluster_id: str | None


class GraphStore:
    """Owns all graph node/edge/intelligence CRUD and read operations.

    Requires a :class:`~pretensor.core.query_runner.QueryRunner` for all
    Cypher execution; holds no direct reference to a kuzu connection.
    """

    def __init__(self, runner: QueryRunner) -> None:
        self._runner = runner

    def clear_graph(self) -> None:
        """Remove all FK edges and table nodes (full re-index)."""
        self._runner.execute("MATCH ()-[r:METRIC_DEPENDS]->() DELETE r")
        self._runner.execute("MATCH (m:MetricTemplate) DELETE m")
        self._runner.execute("MATCH ()-[r:IN_CLUSTER]->() DELETE r")
        self._runner.execute("MATCH (c:Cluster) DELETE c")
        self._runner.execute("MATCH (j:JoinPath) DELETE j")
        self._runner.execute("MATCH ()-[r:FK_REFERENCES]->() DELETE r")
        self._runner.execute("MATCH ()-[r:INFERRED_JOIN]->() DELETE r")
        self._runner.execute("MATCH ()-[r:LINEAGE]->() DELETE r")
        self._runner.execute("MATCH ()-[r:SAME_ENTITY]->() DELETE r")
        self._runner.execute("MATCH ()-[r:HAS_SUBCOLUMN]->() DELETE r")
        self._runner.execute("MATCH ()-[r:HAS_COLUMN]->() DELETE r")
        self._runner.execute("MATCH (col:SchemaColumn) DELETE col")
        self._runner.execute("MATCH ()-[r:REPRESENTS]->() DELETE r")
        self._runner.execute("MATCH (e:Entity) DELETE e")
        self._runner.execute("MATCH (t:SchemaTable) DELETE t")

    def clear_connection_subgraph(self, connection_name: str) -> None:
        """Remove all nodes and edges scoped to one indexed connection (unified graph merge)."""
        cn = connection_name
        self._runner.execute(
            """
            MATCH (e1:Entity)-[r:SAME_ENTITY]->(e2:Entity)
            WHERE e1.connection_name = $cn OR e2.connection_name = $cn
            DELETE r
            """,
            {"cn": cn},
        )
        self._runner.execute(
            """
            MATCH (t:SchemaTable {connection_name: $cn}), (p:JoinPath)
            WHERE p.from_table_id = t.node_id OR p.to_table_id = t.node_id
            DELETE p
            """,
            {"cn": cn},
        )
        self._runner.execute(
            """
            MATCH (m:MetricTemplate {connection_name: $cn})
            DETACH DELETE m
            """,
            {"cn": cn},
        )
        self._runner.execute(
            """
            MATCH (t:SchemaTable {connection_name: $cn})-[r:IN_CLUSTER]->()
            DELETE r
            """,
            {"cn": cn},
        )
        self._runner.execute(
            """
            MATCH (a:SchemaTable)-[r:FK_REFERENCES]->(b:SchemaTable)
            WHERE a.connection_name = $cn OR b.connection_name = $cn
            DELETE r
            """,
            {"cn": cn},
        )
        self._runner.execute(
            """
            MATCH (a:SchemaTable)-[r:INFERRED_JOIN]->(b:SchemaTable)
            WHERE a.connection_name = $cn OR b.connection_name = $cn
            DELETE r
            """,
            {"cn": cn},
        )
        self._runner.execute(
            """
            MATCH (a:SchemaTable)-[r:LINEAGE]->(b:SchemaTable)
            WHERE a.connection_name = $cn OR b.connection_name = $cn
            DELETE r
            """,
            {"cn": cn},
        )
        self._runner.execute(
            """
            MATCH (e:Entity {connection_name: $cn})-[r:REPRESENTS]->()
            DELETE r
            """,
            {"cn": cn},
        )
        self._runner.execute(
            "MATCH (e:Entity {connection_name: $cn}) DELETE e",
            {"cn": cn},
        )
        self._runner.execute(
            """
            MATCH (col:SchemaColumn {connection_name: $cn})
            DETACH DELETE col
            """,
            {"cn": cn},
        )
        self._runner.execute(
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
        self._runner.execute(
            f"""
            MERGE (t:SchemaTable {{node_id: $node_id}})
            SET {set_clause}
            """,
            params,
        )

    def upsert_entity(self, node: EntityNode) -> None:
        """Insert or update a single ``Entity`` node."""
        self._runner.execute(
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
        self._runner.execute(
            """
            MATCH (e:Entity {node_id: $eid}), (t:SchemaTable {node_id: $tid})
            MERGE (e)-[:REPRESENTS]->(t)
            """,
            {"eid": entity_node_id, "tid": table_node_id},
        )

    def set_table_entity_type(self, table_node_id: str, entity_type: str) -> None:
        """Set the business entity label on a ``SchemaTable`` node."""
        self._runner.execute(
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
        self._runner.execute(
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

    def set_table_embedding(
        self, table_node_id: str, vector: list[float] | None
    ) -> None:
        """Set (or clear with ``None``) the embedding on a ``SchemaTable`` node."""
        if vector is not None and len(vector) != _SCHEMA_TABLE_EMBEDDING_DIM:
            raise ValueError(
                f"embedding must be length {_SCHEMA_TABLE_EMBEDDING_DIM}, "
                f"got {len(vector)}"
            )
        self._runner.execute(
            """
            MATCH (t:SchemaTable {node_id: $tid})
            SET t.embedding = $vec
            """,
            {"tid": table_node_id, "vec": vector},
        )

    def has_any_table_embeddings(self) -> bool:
        """Return True iff any ``SchemaTable`` row carries a non-null embedding.

        Cheap LIMIT-1 probe used by the builder to warn before a full
        rebuild silently drops previously computed vectors.
        """
        rows = self._runner.query_all_rows(
            """
            MATCH (t:SchemaTable)
            WHERE t.embedding IS NOT NULL
            RETURN t.node_id LIMIT 1
            """,
            {},
        )
        return bool(rows)

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
        params: dict[str, Any] = {}
        conditions: list[str] = []
        if database is not None:
            params["db"] = database
            conditions.append("(t.connection_name = $db OR t.database = $db)")
        where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        if cluster is not None:
            params["cid"] = cluster
            cypher = f"""
            MATCH (t:SchemaTable)-[:IN_CLUSTER]->(c:Cluster {{node_id: $cid}})
            {where_clause}
            RETURN t.node_id, t.connection_name, t.database, t.schema_name,
                   t.table_name, t.description, t.embedding, c.node_id
            """
        else:
            cypher = f"""
            MATCH (t:SchemaTable)
            {where_clause}
            OPTIONAL MATCH (t)-[:IN_CLUSTER]->(c:Cluster)
            RETURN t.node_id, t.connection_name, t.database, t.schema_name,
                   t.table_name, t.description, t.embedding, c.node_id
            """

        rows = self._runner.query_all_rows(cypher, params)
        for row in rows:
            embedding_raw = row[6]
            embedding: list[float] | None = (
                [float(x) for x in embedding_raw] if embedding_raw is not None else None
            )
            yield TableEmbeddingRow(
                node_id=str(row[0]),
                connection_name=str(row[1] or ""),
                database=str(row[2] or ""),
                schema_name=str(row[3] or ""),
                table_name=str(row[4] or ""),
                description=str(row[5] or ""),
                embedding=embedding,
                cluster_id=str(row[7]) if row[7] is not None else None,
            )

    def upsert_fk_edge(self, edge: GraphEdge) -> None:
        """Insert a foreign-key relationship edge (assumes endpoints exist)."""
        self.merge_fk_edge(edge)

    def merge_fk_edge(self, edge: GraphEdge) -> None:
        """Idempotent FK edge upsert by ``edge_id`` (avoids duplicates on reindex)."""
        self._runner.execute(
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
        self._runner.execute(
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
        self._runner.execute(
            """
            MATCH (a:SchemaTable)-[r:LINEAGE]->(b:SchemaTable)
            WHERE a.connection_name = $cn OR b.connection_name = $cn
            DELETE r
            """,
            {"cn": connection_name},
        )

    def clear_dbt_model_dependency_lineage(self, connection_name: str) -> None:
        """Remove dbt ``parent_map`` lineage edges for one connection (see enrichment dbt)."""
        self._runner.execute(
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
        self._runner.execute("MATCH ()-[r:IN_CLUSTER]->() DELETE r")
        self._runner.execute("MATCH (c:Cluster) DELETE c")
        self._runner.execute("MATCH (j:JoinPath) DELETE j")

    # ── Analyze enrichment: external consumers ────────────────────────────────

    def upsert_external_consumer(self, node: ExternalConsumerNode) -> None:
        """Insert or update an ``ExternalConsumer`` node (idempotent by ``node_id``)."""
        self._runner.execute(
            """
            MERGE (c:ExternalConsumer {node_id: $node_id})
            SET c.connection_name = $connection_name,
                c.service_name = $service_name,
                c.file_path = $file_path,
                c.language = $language,
                c.symbol = $symbol,
                c.kind = $kind,
                c.line_start = $line_start,
                c.line_end = $line_end,
                c.sql_fingerprint = $sql_fingerprint,
                c.confidence = $confidence,
                c.dialect_used = $dialect_used,
                c.scan_run_id = $scan_run_id
            """,
            {
                "node_id": node.node_id,
                "connection_name": node.connection_name,
                "service_name": node.service_name,
                "file_path": node.file_path,
                "language": node.language,
                "symbol": node.symbol,
                "kind": node.kind,
                "line_start": node.line_start,
                "line_end": node.line_end,
                "sql_fingerprint": node.sql_fingerprint,
                "confidence": node.confidence,
                "dialect_used": node.dialect_used,
                "scan_run_id": node.scan_run_id,
            },
        )

    def upsert_consumes_edge(self, edge: ConsumesEdge) -> None:
        """Insert or update a ``CONSUMES`` edge (idempotent by ``edge_id``)."""
        self._runner.execute(
            """
            MATCH (c:ExternalConsumer {node_id: $src}), (t:SchemaTable {node_id: $dst})
            MERGE (c)-[r:CONSUMES {edge_id: $edge_id}]->(t)
            SET r.op = $op,
                r.source = $source,
                r.confidence = $confidence,
                r.scan_run_id = $scan_run_id
            """,
            {
                "src": edge.source_node_id,
                "dst": edge.target_node_id,
                "edge_id": edge.edge_id,
                "op": edge.op,
                "source": edge.source,
                "confidence": edge.confidence,
                "scan_run_id": edge.scan_run_id,
            },
        )

    def mark_tables_have_external_consumers(self, table_node_ids: list[str]) -> None:
        """Set ``has_external_consumers = true`` on the given ``SchemaTable`` nodes."""
        if not table_node_ids:
            return
        self._runner.execute_write(
            """
            MATCH (t:SchemaTable)
            WHERE t.node_id IN $ids
            SET t.has_external_consumers = true
            """,
            {"ids": table_node_ids},
        )

    def sweep_stale_consumers(
        self, service_name: str, connection_name: str, current_scan_run_id: str
    ) -> None:
        """Delete stale ``ExternalConsumer`` rows and ``CONSUMES`` edges from prior
        scans of the same service/connection, then clear ``has_external_consumers``
        on tables that lost their last consumer.

        Edges are matched on their own ``scan_run_id`` as well as their node's, so
        a still-live consumer whose resolved-table set shrank between scans sheds
        its orphaned edges too. The flag is cleared only for tables that had an
        edge removed here and now have no ``CONSUMES`` edges at all — tables
        flagged solely by the database-side signal (dbt exposures) are never
        touched; a table carrying both signals regains the database-side flag at
        the next index run.
        """
        params = {"sn": service_name, "cn": connection_name, "sid": current_scan_run_id}
        stale_edge_predicate = """
            WHERE c.service_name = $sn
              AND c.connection_name = $cn
              AND (c.scan_run_id <> $sid OR r.scan_run_id <> $sid)
        """
        affected = self._runner.query_all_rows(
            f"""
            MATCH (c:ExternalConsumer)-[r:CONSUMES]->(t:SchemaTable)
            {stale_edge_predicate}
            RETURN DISTINCT t.node_id
            """,
            params,
        )
        self._runner.execute_write(
            f"""
            MATCH (c:ExternalConsumer)-[r:CONSUMES]->()
            {stale_edge_predicate}
            DELETE r
            """,
            params,
        )
        self._runner.execute_write(
            """
            MATCH (c:ExternalConsumer)
            WHERE c.service_name = $sn
              AND c.connection_name = $cn
              AND c.scan_run_id <> $sid
            DELETE c
            """,
            params,
        )
        affected_ids = [str(r[0]) for r in affected]
        if affected_ids:
            self._runner.execute_write(
                """
                MATCH (t:SchemaTable)
                WHERE t.node_id IN $ids
                OPTIONAL MATCH (:ExternalConsumer)-[r:CONSUMES]->(t)
                WITH t, count(r) AS remaining
                WHERE remaining = 0
                SET t.has_external_consumers = false
                """,
                {"ids": affected_ids},
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
        checks_json = json.dumps(check_constraints or [], ensure_ascii=False)
        apply_desc = description is not None
        self._runner.execute(
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
            self._runner.execute(
                """
                MATCH (t:SchemaTable {node_id: $tid}), (col:SchemaColumn {node_id: $cid})
                MERGE (t)-[:HAS_COLUMN]->(col)
                """,
                {"tid": table_node_id, "cid": column_node_id},
            )
        else:
            self._runner.execute(
                """
                MATCH (parent:SchemaColumn {node_id: $pid}), (col:SchemaColumn {node_id: $cid})
                MERGE (parent)-[:HAS_SUBCOLUMN]->(col)
                """,
                {"pid": parent_column_id, "cid": column_node_id},
            )

    def delete_column_node(self, column_node_id: str) -> None:
        """Remove a column node (``HAS_COLUMN`` edges removed by cascade)."""
        self._runner.execute(
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
        self._runner.execute(
            """
            MATCH (m:MetricTemplate {node_id: $mid})-[r:METRIC_DEPENDS]->()
            DELETE r
            """,
            {"mid": node_id},
        )
        self._runner.execute(
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
            self._runner.execute(
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
        rows = self._runner.query_all_rows(
            """
            MATCH (src:SchemaTable)-[r:FK_REFERENCES]->(dst:SchemaTable)
            WHERE (src.node_id = $tid AND r.source_column = $col)
               OR (dst.node_id = $tid AND r.target_column = $col)
            RETURN count(*)
            """,
            {"tid": table_node_id, "col": column_name},
        )
        n = int(rows[0][0]) if rows and rows[0][0] is not None else 0
        self._runner.execute(
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
        rows = self._runner.query_all_rows(
            """
            MATCH (src:SchemaTable)-[r:INFERRED_JOIN]->(dst:SchemaTable)
            WHERE (src.node_id = $tid AND r.source_column = $col)
               OR (dst.node_id = $tid AND r.target_column = $col)
            RETURN count(*)
            """,
            {"tid": table_node_id, "col": column_name},
        )
        n = int(rows[0][0]) if rows and rows[0][0] is not None else 0
        self._runner.execute(
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
        self._runner.execute(
            """
            MATCH (t:SchemaTable {node_id: $tid})-[:HAS_COLUMN]->(col:SchemaColumn)
            DETACH DELETE col
            """,
            {"tid": table_node_id},
        )
        self._runner.execute(
            "MATCH (t:SchemaTable {node_id: $tid}) DETACH DELETE t",
            {"tid": table_node_id},
        )

    def mark_clusters_stale_for_table(
        self, table_node_id: str, database_key: str
    ) -> None:
        """Set ``stale`` on clusters linked to the table."""
        self._runner.execute(
            """
            MATCH (t:SchemaTable {node_id: $tid})-[:IN_CLUSTER]->(c:Cluster)
            WHERE c.database_key = $db
            SET c.stale = true
            """,
            {"tid": table_node_id, "db": database_key},
        )

    def remove_table_from_clusters(self, table_node_id: str) -> None:
        """Drop ``IN_CLUSTER`` edges for a table (cluster nodes kept)."""
        self._runner.execute(
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
        self._runner.execute(
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
        self._runner.execute(
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
        self._runner.execute(
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
        self._runner.execute(
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
        self._runner.execute(
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
        self._runner.execute(
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
        self._runner.execute(
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
        self._runner.execute(
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
        self._runner.execute(
            """
            MATCH ()-[r:SAME_ENTITY {edge_id: $eid}]->()
            DELETE r
            """,
            {"eid": edge_id},
        )

    def same_entity_edge_endpoints(self, edge_id: str) -> tuple[str, str] | None:
        """Return ``(from_entity_node_id, to_entity_node_id)`` for an existing edge."""
        rows = self._runner.query_all_rows(
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
            rows = self._runner.query_all_rows(
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
            rows = self._runner.query_all_rows(
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
        rows = self._runner.query_all_rows(
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
        rows = self._runner.query_all_rows(
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
        rows = self._runner.query_all_rows(
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
        rows = self._runner.query_all_rows(
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
