"""Kuzu DDL for the Phase 1 schema graph.

All ``CREATE`` statements for node and relationship tables live here and are
applied only through :class:`pretensor.core.store.KuzuStore`.
"""

from __future__ import annotations

__all__ = [
    "CATALOG_NODE_LABELS",
    "CATALOG_EDGE_TYPES",
    "format_catalog_summary",
    "DDL_CREATE_TABLE_NODE",
    "DDL_CREATE_COLUMN_NODE",
    "DDL_CREATE_HAS_COLUMN_REL",
    "DDL_CREATE_HAS_SUBCOLUMN_REL",
    "DDL_CREATE_ENTITY_NODE",
    "DDL_CREATE_REPRESENTS_REL",
    "DDL_CREATE_FK_REL",
    "DDL_CREATE_INFERRED_JOIN_REL",
    "DDL_CREATE_LINEAGE_REL",
    "DDL_CREATE_SAME_ENTITY_REL",
    "DDL_CREATE_CLUSTER_NODE",
    "DDL_CREATE_IN_CLUSTER_REL",
    "DDL_CREATE_JOIN_PATH_NODE",
    "DDL_CREATE_METRIC_TEMPLATE_NODE",
    "DDL_CREATE_METRIC_DEPENDS_REL",
    # Semantic-layer extension point: reserved tables, empty until populated.
    "DDL_CREATE_SEMANTIC_METRIC_NODE",
    "DDL_CREATE_SEMANTIC_DIMENSION_NODE",
    "DDL_CREATE_SEMANTIC_BUSINESS_RULE_NODE",
    "DDL_CREATE_SEMANTIC_METRIC_DEPENDS_REL",
    "DDL_CREATE_SEMANTIC_DIMENSION_LEVEL_REL",
    "DDL_CREATE_SEMANTIC_RULE_APPLIES_TO_REL",
]

# Table node — one row per physical table in an indexed database.
DDL_CREATE_TABLE_NODE = """
CREATE NODE TABLE IF NOT EXISTS SchemaTable(
    node_id STRING,
    connection_name STRING,
    database STRING,
    schema_name STRING,
    table_name STRING,
    row_count INT64,
    row_count_source STRING,
    comment STRING,
    description STRING,
    tags STRING[],
    entity_type STRING,
    table_type STRING,
    seq_scan_count INT64,
    idx_scan_count INT64,
    insert_count INT64,
    update_count INT64,
    delete_count INT64,
    is_partitioned BOOL,
    partition_key STRING,
    grants_json STRING,
    access_read_count INT64,
    access_write_count INT64,
    days_since_last_access INT64,
    potentially_unused BOOL,
    table_bytes INT64,
    clustering_key STRING,
    role STRING,
    role_confidence DOUBLE,
    classification_signals STRING,
    has_external_consumers BOOL,
    test_count INT64,
    staleness_status STRING,
    staleness_as_of STRING,
    PRIMARY KEY (node_id)
)
"""

# Physical column (for staleness, impact, and FK join tracing).
DDL_CREATE_COLUMN_NODE = """
CREATE NODE TABLE IF NOT EXISTS SchemaColumn(
    node_id STRING,
    connection_name STRING,
    database STRING,
    schema_name STRING,
    table_name STRING,
    column_name STRING,
    description STRING,
    data_type STRING,
    nullable BOOL,
    is_primary_key BOOL,
    is_foreign_key BOOL,
    comment STRING,
    default_value STRING,
    is_indexed BOOL,
    check_constraints_json STRING,
    ordinal_position INT64,
    most_common_values_json STRING,
    histogram_bounds_json STRING,
    stats_correlation DOUBLE,
    column_cardinality INT64,
    index_type STRING,
    index_is_unique BOOL,
    parent_column_id STRING,
    is_array BOOL,
    PRIMARY KEY (node_id)
)
"""

DDL_CREATE_HAS_COLUMN_REL = """
CREATE REL TABLE IF NOT EXISTS HAS_COLUMN(
    FROM SchemaTable TO SchemaColumn
)
"""

# Nested column (e.g. BigQuery STRUCT sub-field) linked from parent SchemaColumn.
DDL_CREATE_HAS_SUBCOLUMN_REL = """
CREATE REL TABLE IF NOT EXISTS HAS_SUBCOLUMN(
    FROM SchemaColumn TO SchemaColumn
)
"""

# Business entity node — groups one or more physical tables (LLM + classifier).
DDL_CREATE_ENTITY_NODE = """
CREATE NODE TABLE IF NOT EXISTS Entity(
    node_id STRING,
    connection_name STRING,
    database STRING,
    name STRING,
    description STRING,
    PRIMARY KEY (node_id)
)
"""

# Entity → physical table projection.
DDL_CREATE_REPRESENTS_REL = """
CREATE REL TABLE IF NOT EXISTS REPRESENTS(
    FROM Entity TO SchemaTable
)
"""

# Directed foreign-key reference between two SchemaTable nodes.
DDL_CREATE_FK_REL = """
CREATE REL TABLE IF NOT EXISTS FK_REFERENCES(
    FROM SchemaTable TO SchemaTable,
    edge_id STRING,
    source_column STRING,
    target_column STRING,
    constraint_name STRING
)
"""

# Heuristic / LLM / statistically scored implicit join (non-declared FK).
DDL_CREATE_INFERRED_JOIN_REL = """
CREATE REL TABLE IF NOT EXISTS INFERRED_JOIN(
    FROM SchemaTable TO SchemaTable,
    edge_id STRING,
    source_column STRING,
    target_column STRING,
    source STRING,
    confidence DOUBLE,
    reasoning STRING
)
"""

# Table-level data lineage (views, triggers, Snowflake tasks/streams) — structural metadata only.
DDL_CREATE_LINEAGE_REL = """
CREATE REL TABLE IF NOT EXISTS LINEAGE(
    FROM SchemaTable TO SchemaTable,
    edge_id STRING,
    source STRING,
    lineage_type STRING,
    confidence DOUBLE
)
"""

# Cross-database / cross-connection business-entity equivalence (human-confirmed or suggested).
DDL_CREATE_SAME_ENTITY_REL = """
CREATE REL TABLE IF NOT EXISTS SAME_ENTITY(
    FROM Entity TO Entity,
    edge_id STRING,
    status STRING,
    score DOUBLE,
    join_columns STRING,
    reasoning STRING,
    created_at STRING,
    confirmed_at STRING,
    confirmed_by STRING
)
"""

# Domain cluster (Leiden + optional LLM label).
DDL_CREATE_CLUSTER_NODE = """
CREATE NODE TABLE IF NOT EXISTS Cluster(
    node_id STRING,
    database_key STRING,
    label STRING,
    description STRING,
    cohesion_score DOUBLE,
    table_count INT64,
    stale BOOL,
    schema_pattern STRING,
    PRIMARY KEY (node_id)
)
"""

DDL_CREATE_IN_CLUSTER_REL = """
CREATE REL TABLE IF NOT EXISTS IN_CLUSTER(
    FROM SchemaTable TO Cluster
)
"""

# Precomputed join path (steps as JSON until column nodes exist in the graph).
DDL_CREATE_JOIN_PATH_NODE = """
CREATE NODE TABLE IF NOT EXISTS JoinPath(
    node_id STRING,
    database_key STRING,
    from_table_id STRING,
    to_table_id STRING,
    depth INT64,
    confidence DOUBLE,
    ambiguous BOOL,
    steps_json STRING,
    semantic_label STRING,
    stale BOOL,
    PRIMARY KEY (node_id)
)
"""

# Pre-validated SQL metric pattern (built during indexing after classification).
DDL_CREATE_METRIC_TEMPLATE_NODE = """
CREATE NODE TABLE IF NOT EXISTS MetricTemplate(
    node_id STRING,
    connection_name STRING,
    database STRING,
    dialect STRING,
    name STRING,
    display_name STRING,
    description STRING,
    sql_template STRING,
    tables_used_json STRING,
    validated BOOL,
    validation_errors_json STRING,
    generated_at_iso STRING,
    stale BOOL,
    PRIMARY KEY (node_id)
)
"""

DDL_CREATE_METRIC_DEPENDS_REL = """
CREATE REL TABLE IF NOT EXISTS METRIC_DEPENDS(
    FROM MetricTemplate TO SchemaTable
)
"""

# ── Semantic-layer extension point: reserved node and edge tables ─────────────
#
# These tables are created by ``KuzuStore.ensure_schema()`` so downstream
# semantic-layer implementations can populate them without schema migrations.
# Empty by default.

DDL_CREATE_SEMANTIC_METRIC_NODE = """
CREATE NODE TABLE IF NOT EXISTS Metric(
    node_id STRING,
    connection_name STRING,
    name STRING,
    display_name STRING,
    description STRING,
    sql_template STRING,
    PRIMARY KEY (node_id)
)
"""

DDL_CREATE_SEMANTIC_DIMENSION_NODE = """
CREATE NODE TABLE IF NOT EXISTS Dimension(
    node_id STRING,
    connection_name STRING,
    name STRING,
    display_name STRING,
    description STRING,
    dimension_type STRING,
    PRIMARY KEY (node_id)
)
"""

DDL_CREATE_SEMANTIC_BUSINESS_RULE_NODE = """
CREATE NODE TABLE IF NOT EXISTS BusinessRule(
    node_id STRING,
    connection_name STRING,
    name STRING,
    description STRING,
    rule_type STRING,
    expression STRING,
    PRIMARY KEY (node_id)
)
"""

DDL_CREATE_SEMANTIC_METRIC_DEPENDS_REL = """
CREATE REL TABLE IF NOT EXISTS METRIC_DEPENDS_ON(
    FROM Metric TO SchemaTable
)
"""

DDL_CREATE_SEMANTIC_DIMENSION_LEVEL_REL = """
CREATE REL TABLE IF NOT EXISTS DIMENSION_LEVEL(
    FROM Dimension TO SchemaColumn
)
"""

DDL_CREATE_SEMANTIC_RULE_APPLIES_TO_REL = """
CREATE REL TABLE IF NOT EXISTS RULE_APPLIES_TO(
    FROM BusinessRule TO SchemaTable
)
"""

# ── Catalog summary (for cypher tool description and schema tool) ──────────────
#
# Single source of truth for the node labels and edge types exposed to MCP
# clients. Kept compact so it can be inlined into the cypher tool description
# without bloating the wire payload.

CATALOG_NODE_LABELS: tuple[tuple[str, str], ...] = (
    ("SchemaTable", "physical table; key fields: schema_name, table_name, row_count, table_type, role"),
    ("SchemaColumn", "physical column; key fields: column_name, data_type, nullable, is_primary_key, is_foreign_key"),
    ("Entity", "business entity grouping one or more tables (name, description)"),
    ("Cluster", "Leiden community of related tables (label, table_count, cohesion_score)"),
    ("JoinPath", "precomputed join path between two tables (depth, confidence, steps_json)"),
    ("MetricTemplate", "validated SQL metric pattern (name, sql_template, validated, tables_used_json)"),
    ("Metric", "semantic metric (extension point; empty by default)"),
    ("Dimension", "semantic dimension (extension point; empty by default)"),
    ("BusinessRule", "semantic business rule (extension point; empty by default)"),
)

CATALOG_EDGE_TYPES: tuple[tuple[str, str, str, str], ...] = (
    ("HAS_COLUMN", "SchemaTable", "SchemaColumn", "table → its columns"),
    ("HAS_SUBCOLUMN", "SchemaColumn", "SchemaColumn", "nested column (e.g. BigQuery STRUCT field)"),
    ("REPRESENTS", "Entity", "SchemaTable", "entity → underlying physical tables"),
    ("FK_REFERENCES", "SchemaTable", "SchemaTable", "declared foreign key (source_column, target_column)"),
    ("INFERRED_JOIN", "SchemaTable", "SchemaTable", "implicit join (source, confidence, source_column, target_column)"),
    ("LINEAGE", "SchemaTable", "SchemaTable", "view/trigger/task lineage (lineage_type, confidence)"),
    ("SAME_ENTITY", "Entity", "Entity", "cross-database entity equivalence (status, score)"),
    ("IN_CLUSTER", "SchemaTable", "Cluster", "table → cluster membership"),
    ("METRIC_DEPENDS", "MetricTemplate", "SchemaTable", "metric template → tables it reads"),
    ("METRIC_DEPENDS_ON", "Metric", "SchemaTable", "metric → tables (extension point)"),
    ("DIMENSION_LEVEL", "Dimension", "SchemaColumn", "dimension → columns (extension point)"),
    ("RULE_APPLIES_TO", "BusinessRule", "SchemaTable", "rule → tables (extension point)"),
)


def format_catalog_summary() -> str:
    """Return a compact, human-readable summary of the graph's node/edge catalog.

    Used to inline the schema into the ``cypher`` tool description so MCP
    clients can write valid queries without a separate round-trip.
    """
    node_lines = [f"  ({label}) — {desc}" for label, desc in CATALOG_NODE_LABELS]
    edge_lines = [
        f"  -[{name}]->  {src} → {dst} — {desc}"
        for name, src, dst, desc in CATALOG_EDGE_TYPES
    ]
    return (
        "Node labels:\n"
        + "\n".join(node_lines)
        + "\n\nEdge types:\n"
        + "\n".join(edge_lines)
        + "\n\nTip: when ranking by SchemaTable.row_count, exclude views and "
        "unknowns — they may be NULL or -1 (timed-out view counts):\n"
        "  MATCH (t:SchemaTable) WHERE t.table_type = 'table' AND t.row_count > 0 "
        "RETURN t ORDER BY t.row_count DESC LIMIT 25"
    )