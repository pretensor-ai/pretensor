"""MCP ``schema`` tool: introspect node labels and edge types in the Kuzu graph."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pretensor.core.schema import CATALOG_EDGE_TYPES, CATALOG_NODE_LABELS
from pretensor.core.store import KuzuStore

from ..service_registry import (
    graph_path_for_entry,
    load_registry,
    resolve_registry_entry,
)

logger = logging.getLogger(__name__)

__all__ = ["schema_payload"]

_NODE_LABEL_DESC = {label: desc for label, desc in CATALOG_NODE_LABELS}
_EDGE_INFO = {name: (src, dst, desc) for name, src, dst, desc in CATALOG_EDGE_TYPES}


def _list_catalog_tables(store: KuzuStore) -> list[tuple[str, str]]:
    """Return ``[(name, type), ...]`` from ``CALL show_tables()``.

    ``type`` is ``'NODE'`` or ``'REL'`` per Kuzu.
    """
    raw = store.execute("CALL show_tables() RETURN name, type")
    if isinstance(raw, list):
        return []
    rows = raw.rows_as_dict()
    out: list[tuple[str, str]] = []
    while rows.has_next():
        row: Any = rows.get_next()
        if isinstance(row, dict):
            out.append((str(row.get("name", "")), str(row.get("type", "")).upper()))
    return out


def _table_properties(store: KuzuStore, label: str) -> list[dict[str, str]]:
    """Return ``[{"name": ..., "type": ...}, ...]`` for one node/rel table."""
    try:
        raw = store.execute(f"CALL table_info('{label}') RETURN name, type")
    except Exception:
        return []
    if isinstance(raw, list):
        return []
    rows = raw.rows_as_dict()
    out: list[dict[str, str]] = []
    while rows.has_next():
        row: Any = rows.get_next()
        if isinstance(row, dict):
            out.append(
                {
                    "name": str(row.get("name", "")),
                    "type": str(row.get("type", "")),
                }
            )
    return out


def schema_payload(
    graph_dir: Path,
    *,
    database: str,
    label: str | None = None,
) -> dict[str, Any]:
    """Return the node and edge catalog for the indexed graph of ``database``.

    Args:
        graph_dir: Graph workspace directory (registry + Kuzu files).
        database: Connection name or logical database name.
        label: If set, return only the matching node label or edge type.

    Returns:
        ``{"nodes": [...], "edges": [...]}`` on success, or ``{"error": "..."}``.
    """
    db = (database or "").strip()
    if not db:
        return {"error": "Missing `database`"}

    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, db)
    if entry is None:
        return {"error": f"Unknown database: {db!r}"}

    graph_path = graph_path_for_entry(entry)
    if not graph_path.exists():
        return {"error": f"Graph file not found for database: {db!r}"}

    store = KuzuStore(graph_path)
    try:
        store.ensure_schema()
        catalog = _list_catalog_tables(store)
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for name, kind in catalog:
            if label and name != label:
                continue
            properties = _table_properties(store, name)
            if kind == "NODE":
                nodes.append(
                    {
                        "label": name,
                        "description": _NODE_LABEL_DESC.get(name, ""),
                        "properties": properties,
                    }
                )
            elif kind == "REL":
                src, dst, desc = _EDGE_INFO.get(name, ("", "", ""))
                edges.append(
                    {
                        "type": name,
                        "from": src,
                        "to": dst,
                        "description": desc,
                        "properties": properties,
                    }
                )
    finally:
        store.close()

    if label and not nodes and not edges:
        return {"error": f"Unknown label: {label!r}"}
    return {"nodes": nodes, "edges": edges}
