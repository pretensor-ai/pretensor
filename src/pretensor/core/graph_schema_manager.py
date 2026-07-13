"""Schema creation and idempotent column migration for the Kuzu graph.

Extracted from :class:`~pretensor.core.store.KuzuStore` to isolate the
schema-migration concern from the rest of the store layer.
:class:`GraphSchemaManager` is composed inside ``KuzuStore``; it owns every
``ensure_schema`` DDL call and every ``_ensure_*`` / ``_drop_*`` column-migration
helper so that ``KuzuStore`` can focus on query, upsert, and read operations.
"""

from __future__ import annotations

import logging

import kuzu

from pretensor.core import schema as graph_schema

logger = logging.getLogger(__name__)

# Kept in sync with ``pretensor.intelligence.embeddings.EMBEDDING_DIM`` (384).
# Duplicated here to avoid the circular import created by
# ``intelligence/__init__.py`` eagerly loading ``cluster_labeler`` (which itself
# imports ``KuzuStore``). The schema DDL literal ``FLOAT[384]`` and this
# constant are both asserted against the real ``EMBEDDING_DIM`` in tests.
_SCHEMA_TABLE_EMBEDDING_DIM = 384


class GraphSchemaManager:
    """Owns schema creation and idempotent column migrations for the Kuzu graph."""

    def __init__(self, conn: kuzu.Connection) -> None:
        self._conn = conn

    def _add_column_if_missing(self, table: str, col_name: str, col_type: str) -> None:
        try:
            self._conn.execute(f"ALTER TABLE {table} ADD {col_name} {col_type}")
        except RuntimeError as exc:
            message = str(exc).lower()
            if col_name.lower() in message and "already has" in message:
                return
            logger.warning("%s ALTER for %s: %s", table, col_name, exc)

    def _drop_column_if_present(self, table: str, col_name: str) -> None:
        try:
            self._conn.execute(f"ALTER TABLE {table} DROP {col_name}")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "does not have" in message or "does not exist" in message:
                return
            logger.debug("%s DROP for %s: %s", table, col_name, exc)

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
        self._ensure_schema_table_embedding_column()
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
        self._add_column_if_missing("SchemaTable", "entity_type", "STRING")

    def _ensure_schema_table_table_type_column(self) -> None:
        """Add ``table_type`` to ``SchemaTable`` when upgrading older graph files."""
        self._add_column_if_missing("SchemaTable", "table_type", "STRING")

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
            self._add_column_if_missing("SchemaTable", col_name, col_type)

    def _ensure_schema_table_classification_columns(self) -> None:
        """Add classifier fields for older graph files."""
        for col_name, col_type in (
            ("role", "STRING"),
            ("role_confidence", "DOUBLE"),
            ("classification_signals", "STRING"),
        ):
            self._add_column_if_missing("SchemaTable", col_name, col_type)

    def _ensure_schema_table_dbt_enrichment_columns(self) -> None:
        """Add dbt semantic-layer fields for older graph files."""
        for col_name, col_type in (
            ("has_external_consumers", "BOOL"),
            ("test_count", "INT64"),
        ):
            self._add_column_if_missing("SchemaTable", col_name, col_type)

    def _ensure_schema_table_description_tags_columns(self) -> None:
        """Add first-class ``description`` and ``tags`` for dbt / LLM context."""
        for col_name, col_type in (
            ("description", "STRING"),
            ("tags", "STRING[]"),
        ):
            self._add_column_if_missing("SchemaTable", col_name, col_type)

    def _ensure_schema_table_staleness_columns(self) -> None:
        """Add dbt source-freshness fields (``staleness_status``/``staleness_as_of``)."""
        for col_name, col_type in (
            ("staleness_status", "STRING"),
            ("staleness_as_of", "STRING"),
        ):
            self._add_column_if_missing("SchemaTable", col_name, col_type)

    def _ensure_schema_table_row_count_source_column(self) -> None:
        """Add ``row_count_source`` for older graph files (view-count provenance)."""
        self._add_column_if_missing("SchemaTable", "row_count_source", "STRING")

    def _ensure_schema_table_embedding_column(self) -> None:
        """Add ``embedding FLOAT[384]`` to ``SchemaTable`` for older graph files."""
        self._add_column_if_missing(
            "SchemaTable",
            "embedding",
            f"FLOAT[{_SCHEMA_TABLE_EMBEDDING_DIM}]",
        )

    def _drop_schema_table_legacy_dbt_columns(self) -> None:
        """Drop legacy duplicates ``dbt_description`` and ``tags_json``.

        These were superseded by the first-class ``description`` / ``tags``
        fields added in the dbt enrichment update. Older graph files that were
        written before the consolidation may still have them; silently drop if
        present.
        """
        for col_name in ("dbt_description", "tags_json"):
            self._drop_column_if_present("SchemaTable", col_name)

    def _ensure_schema_column_description_column(self) -> None:
        """Add ``description`` on ``SchemaColumn`` for dbt column docs."""
        self._add_column_if_missing("SchemaColumn", "description", "STRING")

    def _ensure_schema_column_enrichment_columns(self) -> None:
        """Add column comment / constraint fields for older Kuzu files."""
        for col_name, col_type in (
            ("comment", "STRING"),
            ("default_value", "STRING"),
            ("is_indexed", "BOOL"),
            ("check_constraints_json", "STRING"),
            ("ordinal_position", "INT64"),
        ):
            self._add_column_if_missing("SchemaColumn", col_name, col_type)

    def _ensure_schema_column_stats_columns(self) -> None:
        """Add planner statistics mirror fields for older Kuzu files."""
        for col_name, col_type in (
            ("most_common_values_json", "STRING"),
            ("histogram_bounds_json", "STRING"),
            ("stats_correlation", "DOUBLE"),
        ):
            self._add_column_if_missing("SchemaColumn", col_name, col_type)

    def _ensure_schema_column_catalog_enrichment_columns(self) -> None:
        """Add catalog cardinality and index-detail fields for older Kuzu files."""
        for col_name, col_type in (
            ("column_cardinality", "INT64"),
            ("index_type", "STRING"),
            ("index_is_unique", "BOOL"),
        ):
            self._add_column_if_missing("SchemaColumn", col_name, col_type)

    def _ensure_schema_column_nested_columns(self) -> None:
        """Add nested-column linkage fields for older Kuzu files."""
        for col_name, col_type in (
            ("parent_column_id", "STRING"),
            ("is_array", "BOOL"),
        ):
            self._add_column_if_missing("SchemaColumn", col_name, col_type)

    def _ensure_fk_constraint_name_column(self) -> None:
        """Add ``constraint_name`` to ``FK_REFERENCES`` when upgrading older graph files."""
        self._add_column_if_missing("FK_REFERENCES", "constraint_name", "STRING")

    def _ensure_cluster_stale_column(self) -> None:
        """Add ``stale`` to ``Cluster`` when upgrading older graph files."""
        self._add_column_if_missing("Cluster", "stale", "BOOL")

    def _ensure_cluster_schema_pattern_column(self) -> None:
        """Add ``schema_pattern`` to ``Cluster`` when upgrading older graph files."""
        self._add_column_if_missing("Cluster", "schema_pattern", "STRING")

    def _ensure_join_path_stale_column(self) -> None:
        """Add ``stale`` to ``JoinPath`` when upgrading older graph files."""
        self._add_column_if_missing("JoinPath", "stale", "BOOL")

    def _ensure_metric_template_stale_column(self) -> None:
        """Add ``stale`` to ``MetricTemplate`` when upgrading older graph files."""
        self._add_column_if_missing("MetricTemplate", "stale", "BOOL")

    def _ensure_metric_template_dialect_column(self) -> None:
        """Add ``dialect`` to ``MetricTemplate`` when upgrading older graph files."""
        self._add_column_if_missing("MetricTemplate", "dialect", "STRING")
