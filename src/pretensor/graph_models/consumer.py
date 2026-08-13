"""Graph models for external code consumers of physical tables.

``ExternalConsumerNode`` is a code location (service + repo-relative file + a
normalized SQL fingerprint) that issues a SQL statement; ``ConsumesEdge`` links
it to the ``SchemaTable`` it reads or writes. Raw SQL text is never carried on
either model — only the ``sql_fingerprint`` and resolved endpoints.
"""

from __future__ import annotations

from pydantic import Field

from pretensor.graph_models.base import GraphModel

__all__ = ["ExternalConsumerNode", "ConsumesEdge"]


class ExternalConsumerNode(GraphModel):
    """A code location that consumes one or more physical tables via SQL."""

    node_id: str = Field(
        description="Stable primary key in the Kuzu ExternalConsumer node table."
    )
    connection_name: str = Field(
        description="Pretensor connection whose tables this consumer resolves against."
    )
    service_name: str = Field(
        description="Consumer-service label (defaults to the scanned repo basename)."
    )
    file_path: str = Field(
        description="Repo-relative path of the source file (never absolute)."
    )
    language: str = Field(description="Source language, e.g. 'python'.")
    symbol: str = Field(
        description="Enclosing identifier: assigned variable or called method."
    )
    kind: str = Field(
        description="Extraction context: 'assignment', 'call', 'fstring', or 'sql_file'."
    )
    line_start: int
    line_end: int
    sql_fingerprint: str = Field(
        description="sha256[:16] of the normalized SQL. NOT the raw SQL text."
    )
    confidence: float = 1.0
    dialect_used: str = Field(
        default="",
        description="sqlglot dialect that parsed the statement (empty if none).",
    )
    scan_run_id: str = Field(
        description="Idempotency key for the scan that wrote this node."
    )


class ConsumesEdge(GraphModel):
    """Directed edge from an ExternalConsumer to a table it reads or writes."""

    edge_id: str = Field(
        description="Stable primary key in the Kuzu CONSUMES rel table."
    )
    source_node_id: str = Field(description="ExternalConsumer node id (FROM).")
    target_node_id: str = Field(description="SchemaTable node id (TO).")
    op: str = Field(description="Access kind: 'read' or 'write'.")
    source: str = Field(
        default="analyze",
        description="Provenance label for the edge (code-scanner = 'analyze').",
    )
    confidence: float = 1.0
    scan_run_id: str = Field(
        description="Idempotency key for the scan that wrote this edge."
    )
