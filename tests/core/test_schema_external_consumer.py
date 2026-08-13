"""ExternalConsumer node + CONSUMES rel are declared and advertised in the catalog."""

from __future__ import annotations

from pretensor.core import schema as graph_schema


def test_ddl_constants_exported() -> None:
    assert "DDL_CREATE_EXTERNAL_CONSUMER_NODE" in graph_schema.__all__
    assert "DDL_CREATE_CONSUMES_REL" in graph_schema.__all__
    assert "ExternalConsumer" in graph_schema.DDL_CREATE_EXTERNAL_CONSUMER_NODE
    assert "CONSUMES" in graph_schema.DDL_CREATE_CONSUMES_REL


def test_consumes_rel_endpoints() -> None:
    # The edge must run FROM ExternalConsumer TO SchemaTable.
    assert (
        "FROM ExternalConsumer TO SchemaTable" in graph_schema.DDL_CREATE_CONSUMES_REL
    )


def test_external_consumer_advertised_as_node_label() -> None:
    labels = {name for name, _desc in graph_schema.CATALOG_NODE_LABELS}
    assert "ExternalConsumer" in labels


def test_consumes_advertised_as_edge_type() -> None:
    edges = {
        name: (src, dst) for name, src, dst, _desc in graph_schema.CATALOG_EDGE_TYPES
    }
    assert "CONSUMES" in edges
    assert edges["CONSUMES"] == ("ExternalConsumer", "SchemaTable")


def test_catalog_summary_mentions_new_types() -> None:
    summary = graph_schema.format_catalog_summary()
    assert "ExternalConsumer" in summary
    assert "CONSUMES" in summary
