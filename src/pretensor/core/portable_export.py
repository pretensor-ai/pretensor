"""Portable JSON export helpers for Kuzu graphs."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import TypeAlias, TypedDict

from pretensor.core.store import KuzuStore

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
EdgeRow: TypeAlias = dict[str, JsonValue]

_EDGE_ENDPOINT_COLUMNS = frozenset({"from", "to", "src", "dst"})


class NodeTypeExport(TypedDict):
    """Portable representation of one node label."""

    type: str
    properties: list[str]
    rows: list[dict[str, JsonValue]]


class EdgeTypeExport(TypedDict):
    """Portable representation of one edge label."""

    type: str
    properties: list[str]
    rows: list[EdgeRow]


class PortableGraphExport(TypedDict):
    """Top-level export payload shape."""

    connection_name: str
    database: str
    graph_path: str
    exported_at: str
    node_types: list[NodeTypeExport]
    edge_types: list[EdgeTypeExport]
    stats: dict[str, int]


class _NodeExportResult(TypedDict):
    payload: NodeTypeExport
    node_ids: set[str]


def export_graph_payload(
    store: KuzuStore,
    *,
    connection_name: str,
    database_name: str,
    graph_path: Path,
) -> PortableGraphExport:
    """Materialize a portable JSON payload for all node and edge labels.

    Args:
        store: Open graph store.
        connection_name: Logical registry connection name being exported.
        database_name: Database key from the registry entry.
        graph_path: Path to the Kuzu graph file.

    Returns:
        Portable export payload containing all label rows and properties.
    """
    rows = store.query_all_rows("CALL show_tables() RETURN name, type ORDER BY name")
    node_types: list[NodeTypeExport] = []
    edge_types: list[EdgeTypeExport] = []
    included_node_ids: set[str] = set()
    node_count = 0
    edge_count = 0
    node_table_names: list[str] = []
    edge_table_names: list[str] = []
    for table_name_raw, table_type_raw in rows:
        table_name = str(table_name_raw)
        table_type = str(table_type_raw).upper()
        if "REL" in table_type:
            edge_table_names.append(table_name)
        else:
            node_table_names.append(table_name)

    for table_name in node_table_names:
        columns = _table_columns(store, table_name)
        node_result = _export_node_type(
            store,
            table_name=table_name,
            columns=columns,
            connection_name=connection_name,
            database_name=database_name,
        )
        node_types.append(node_result["payload"])
        included_node_ids.update(node_result["node_ids"])
        node_count += len(node_result["payload"]["rows"])

    for table_name in edge_table_names:
        columns = _table_columns(store, table_name)
        edge_payload = _export_edge_type(
            store,
            table_name=table_name,
            columns=columns,
            included_node_ids=included_node_ids,
        )
        edge_types.append(edge_payload)
        edge_count += len(edge_payload["rows"])
    return {
        "connection_name": connection_name,
        "database": database_name,
        "graph_path": str(graph_path),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "node_types": node_types,
        "edge_types": edge_types,
        "stats": {
            "node_types": len(node_types),
            "edge_types": len(edge_types),
            "node_count": node_count,
            "edge_count": edge_count,
        },
    }


def _table_columns(store: KuzuStore, table_name: str) -> list[str]:
    quoted = table_name.replace("'", "''")
    rows = store.query_all_rows(f"CALL table_info('{quoted}') RETURN name, type")
    return [str(row[0]) for row in rows if row and row[0] is not None]


def _export_node_type(
    store: KuzuStore,
    *,
    table_name: str,
    columns: list[str],
    connection_name: str,
    database_name: str,
) -> _NodeExportResult:
    projections = ", ".join(
        f"n.{_quote_identifier(column)} AS {_quote_identifier(column)}"
        for column in columns
    )
    order_clause = (
        f" ORDER BY n.{_quote_identifier('node_id')}" if "node_id" in columns else ""
    )
    scope_predicate = _scope_predicate(columns, "n")
    where_clause = f" WHERE {scope_predicate}" if scope_predicate else ""
    cypher = (
        f"MATCH (n:{_quote_identifier(table_name)})"
        f"{where_clause} "
        f"RETURN {projections}{order_clause}"
    )
    rows = store.query_all_rows(
        cypher,
        {"connection_name": connection_name, "database_name": database_name},
    )
    materialized_rows = [_materialize_row(columns, row) for row in rows]
    node_ids = {
        str(row_map["node_id"])
        for row_map in materialized_rows
        if "node_id" in row_map and row_map["node_id"] is not None
    }
    return {
        "payload": {
            "type": table_name,
            "properties": columns,
            "rows": materialized_rows,
        },
        "node_ids": node_ids,
    }


def _export_edge_type(
    store: KuzuStore,
    *,
    table_name: str,
    columns: list[str],
    included_node_ids: set[str],
) -> EdgeTypeExport:
    property_columns = [
        column for column in columns if column.lower() not in _EDGE_ENDPOINT_COLUMNS
    ]
    projections = [
        f"src.{_quote_identifier('node_id')} AS {_quote_identifier('__from_node_id')}",
        f"dst.{_quote_identifier('node_id')} AS {_quote_identifier('__to_node_id')}",
    ]
    projections.extend(
        f"r.{_quote_identifier(column)} AS {_quote_identifier(column)}"
        for column in property_columns
    )
    field_names = ["__from_node_id", "__to_node_id", *property_columns]
    cypher = (
        f"MATCH (src)-[r:{_quote_identifier(table_name)}]->(dst) "
        f"RETURN {', '.join(projections)} "
        f"ORDER BY {_quote_identifier('__from_node_id')}, {_quote_identifier('__to_node_id')}"
    )
    rows = store.query_all_rows(cypher)
    filtered_rows = [
        row
        for row in rows
        if str(row[0]) in included_node_ids and str(row[1]) in included_node_ids
    ]
    return {
        "type": table_name,
        "properties": property_columns,
        "rows": [_materialize_edge_row(field_names, row) for row in filtered_rows],
    }


def _scope_predicate(columns: list[str], alias: str) -> str | None:
    if "connection_name" in columns:
        return f"{alias}.{_quote_identifier('connection_name')} = $connection_name"
    if "database" in columns:
        return f"{alias}.{_quote_identifier('database')} = $database_name"
    if "database_key" in columns:
        return f"{alias}.{_quote_identifier('database_key')} = $database_name"
    return None


def _materialize_row(
    field_names: list[str], row: tuple[object, ...]
) -> dict[str, JsonValue]:
    return {
        field_names[idx]: _to_json_value(value) for idx, value in enumerate(row[: len(field_names)])
    }


def _materialize_edge_row(field_names: list[str], row: tuple[object, ...]) -> EdgeRow:
    row_map = _materialize_row(field_names, row)
    return {
        "__from_node_id": row_map["__from_node_id"],
        "__to_node_id": row_map["__to_node_id"],
        **{
            key: value
            for key, value in row_map.items()
            if key not in {"__from_node_id", "__to_node_id"}
        },
    }


def _to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    return str(value)


def _quote_identifier(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"
