"""Tests for portable graph JSON export payload construction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pretensor.core.portable_export import export_graph_payload


def test_export_graph_payload_filters_to_connection_scope() -> None:
    """Only nodes/edges for the requested connection/database are exported."""
    store = MagicMock()
    store.query_all_rows.side_effect = [
        [("FK_REFERENCES", "REL"), ("SchemaTable", "NODE")],
        [
            ("node_id", "STRING"),
            ("connection_name", "STRING"),
            ("database", "STRING"),
            ("table_name", "STRING"),
        ],
        [("t1", "myconn", "mydb", "orders"), ("t2", "myconn", "mydb", "customers")],
        [
            ("edge_id", "STRING"),
            ("source", "STRING"),
            ("source_column", "STRING"),
            ("target_column", "STRING"),
        ],
        [
            ("t1", "t2", "e1", "heuristic", "customer_id", "id"),
            ("t2", "t3", "e2", "heuristic", "tenant_id", "id"),
        ],
    ]

    payload = export_graph_payload(
        store,
        connection_name="myconn",
        database_name="mydb",
        graph_path=Path("/tmp/graph.kuzu"),
    )

    assert payload["connection_name"] == "myconn"
    assert payload["database"] == "mydb"
    assert payload["stats"]["node_types"] == 1
    assert payload["stats"]["edge_types"] == 1
    assert payload["stats"]["node_count"] == 2
    assert payload["stats"]["edge_count"] == 1

    node_rows = payload["node_types"][0]["rows"]
    assert node_rows == [
        {
            "node_id": "t1",
            "connection_name": "myconn",
            "database": "mydb",
            "table_name": "orders",
        },
        {
            "node_id": "t2",
            "connection_name": "myconn",
            "database": "mydb",
            "table_name": "customers",
        },
    ]

    edge_rows = payload["edge_types"][0]["rows"]
    assert edge_rows == [
        {
            "__from_node_id": "t1",
            "__to_node_id": "t2",
            "edge_id": "e1",
            "source": "heuristic",
            "source_column": "customer_id",
            "target_column": "id",
        }
    ]


def test_export_node_query_only_binds_referenced_scope_param() -> None:
    """Regression test for the export scope-param binding crash.

    Node tables that carry only one of ``connection_name``/``database``/
    ``database_key`` (or none of them, e.g. semantic-layer tables) must not
    be queried with the *other* param still bound — Kuzu's prepared
    statements reject params that aren't referenced in the query text.
    """
    store = MagicMock()
    store.query_all_rows.side_effect = [
        [("ClusterLike", "NODE"), ("MetricLike", "NODE"), ("Unscoped", "NODE")],
        [("node_id", "STRING"), ("database_key", "STRING"), ("label", "STRING")],
        [("c1", "dbkey1", "Cluster One")],
        [("node_id", "STRING"), ("connection_name", "STRING"), ("name", "STRING")],
        [("m1", "conn1", "Metric One")],
        [("node_id", "STRING"), ("value", "STRING")],
        [("u1", "hello")],
    ]

    export_graph_payload(
        store,
        connection_name="myconn",
        database_name="mydb",
        graph_path=Path("/tmp/graph.kuzu"),
    )

    node_query_calls = [
        call
        for call in store.query_all_rows.call_args_list
        if "MATCH (n:" in call.args[0]
    ]
    assert len(node_query_calls) == 3

    cluster_call, metric_call, unscoped_call = node_query_calls
    assert cluster_call.args[1] == {"database_name": "mydb"}
    assert metric_call.args[1] == {"connection_name": "myconn"}
    assert unscoped_call.args[1] == {}
