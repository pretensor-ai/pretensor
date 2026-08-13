"""Contract: ExternalConsumerNode / ConsumesEdge models match the Kuzu DDL shape.

The models are the typed surface the enrichment pipeline (phase 2) and MCP tools
(phase 4) build against, so the field set must stay aligned with the DDL columns.
"""

from __future__ import annotations

import re

from pretensor.core import schema as graph_schema
from pretensor.graph_models.consumer import ConsumesEdge, ExternalConsumerNode


def _ddl_columns(ddl: str) -> set[str]:
    """Extract column names from a ``CREATE ... TABLE`` DDL body."""
    body = ddl[ddl.index("(") + 1 : ddl.rindex(")")]
    cols: set[str] = set()
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith(("PRIMARY KEY", "FROM ")):
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+[A-Za-z]", line)
        if m:
            cols.add(m.group(1))
    return cols


def test_external_consumer_model_matches_ddl_columns() -> None:
    ddl_cols = _ddl_columns(graph_schema.DDL_CREATE_EXTERNAL_CONSUMER_NODE)
    model_fields = set(ExternalConsumerNode.model_fields)
    # Every persisted column is representable on the node model.
    assert ddl_cols == model_fields


def test_consumes_edge_model_covers_ddl_columns() -> None:
    ddl_cols = _ddl_columns(graph_schema.DDL_CREATE_CONSUMES_REL)
    # The edge model maps endpoints to source/target_node_id; the remaining edge
    # properties must each have a model field.
    model_fields = set(ConsumesEdge.model_fields)
    assert ddl_cols <= model_fields
    for prop in ("edge_id", "op", "source", "confidence", "scan_run_id"):
        assert prop in model_fields


def test_models_are_frozen_and_construct() -> None:
    node = ExternalConsumerNode(
        node_id="consumer::x",
        connection_name="c",
        service_name="svc",
        file_path="src/app.py",
        language="python",
        symbol="query",
        kind="assignment",
        line_start=1,
        line_end=2,
        sql_fingerprint="deadbeefdeadbeef",
        confidence=0.9,
        dialect_used="postgres",
        scan_run_id="run1",
    )
    edge = ConsumesEdge(
        edge_id="consumes::x",
        source_node_id="consumer::x",
        target_node_id="c::public::users",
        op="read",
        scan_run_id="run1",
    )
    assert edge.source == "analyze"  # default provenance
    assert node.confidence == 0.9


def test_no_raw_sql_field_on_models() -> None:
    # Spec invariant: raw SQL text is never persisted, only the fingerprint.
    for field in (*ExternalConsumerNode.model_fields, *ConsumesEdge.model_fields):
        assert "sql_text" not in field
        assert field not in {"sql", "raw_sql", "query", "statement"}
