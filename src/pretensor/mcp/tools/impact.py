"""MCP ``impact`` tool payload."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from pretensor.intelligence.shadow_alias import get_shadow_alias_node_ids
from pretensor.visibility.filter import VisibilityFilter

from ..payload_types import ImpactItemPayload
from ..service_context import get_effective_visibility_filter
from ..service_registry import (
    graph_path_for_entry,
    load_registry,
    open_store_for_entry,
    resolve_registry_entry,
)
from .context import qualified_from_node_id, resolve_table_node_id


def impact_payload(
    graph_dir: Path,
    *,
    table: str,
    database: str,
    column: str | None = None,
    max_depth: int = 3,
    visibility_filter: VisibilityFilter | None = None,
) -> dict[str, Any]:
    """Downstream tables grouped by hop depth (FK + inferred edges)."""
    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, database)
    if entry is None:
        return {
            "error": "Unknown database connection or name; pass `database`.",
        }
    gp = graph_path_for_entry(entry)
    if not gp.exists():
        return {"error": f"Graph file missing: {gp}"}

    db_key = str(entry.database)
    vf = visibility_filter or get_effective_visibility_filter()
    cn = str(entry.connection_name)
    store = open_store_for_entry(entry)
    try:
        start_id, err = resolve_table_node_id(
            store, table, db_key, visibility_filter=vf
        )
        if err is not None:
            try:
                return json.loads(err)
            except json.JSONDecodeError:
                return {"error": err}
        if start_id is None:
            return {"error": "Could not resolve starting table; check the table name and database."}

        fk_rows = store.query_all_rows(
            """
            MATCH (a:SchemaTable)-[r:FK_REFERENCES]->(b:SchemaTable)
            WHERE a.database = $db AND b.database = $db
            RETURN a.node_id, b.node_id, r.source_column, r.target_column, 1.0 AS conf
            """,
            {"db": db_key},
        )
        inf_rows = store.query_all_rows(
            """
            MATCH (a:SchemaTable)-[r:INFERRED_JOIN]->(b:SchemaTable)
            WHERE a.database = $db AND b.database = $db
            RETURN a.node_id, b.node_id, r.source_column, r.target_column, r.confidence
            """,
            {"db": db_key},
        )
        shadow_ids = get_shadow_alias_node_ids(store, db_key)

        rev: dict[str, list[tuple[str, str, str, float]]] = {}
        for row in fk_rows + inf_rows:
            child, parent, sc, tc, conf = row
            if child is None or parent is None:
                continue
            sa, sb = str(child), str(parent)
            if sa in shadow_ids or sb in shadow_ids:
                continue
            if vf is not None:
                cmeta = store.query_all_rows(
                    """
                    MATCH (t:SchemaTable {node_id: $id})
                    RETURN t.schema_name, t.table_name
                    """,
                    {"id": sa},
                )
                if cmeta:
                    sn, tn = cmeta[0]
                    if not vf.is_table_visible(cn, str(sn or ""), str(tn or "")):
                        continue
            w = float(conf) if conf is not None else 1.0
            rev.setdefault(sb, []).append((sa, str(sc or ""), str(tc or ""), w))

        q: deque[tuple[str, int, float, str, str]] = deque()
        for dependent, scol, _tcol, w in rev.get(start_id, []):
            if column is not None and scol != column:
                continue
            dep_short = qualified_from_node_id(store, dependent).split(".")[-1]
            via = f"{dep_short}.{scol}"
            q.append((dependent, 1, w, via, start_id))

        seen: dict[str, tuple[int, float, str]] = {}
        while q:
            cur, depth, path_min, via, _parent = q.popleft()
            if cur in seen:
                continue
            seen[cur] = (depth, path_min, via)
            if depth >= max_depth:
                continue
            for dependent, scol, _tcol, w in rev.get(cur, []):
                if dependent in seen:
                    continue
                new_d = depth + 1
                new_min = min(path_min, w)
                dep_short = qualified_from_node_id(store, dependent).split(".")[-1]
                new_via = f"{via} → {dep_short}.{scol}"
                q.append((dependent, new_d, new_min, new_via, cur))

        direct: list[ImpactItemPayload] = []
        two_hop: list[ImpactItemPayload] = []
        three_hop: list[ImpactItemPayload] = []
        low_confidence: list[ImpactItemPayload] = []
        for node_id, (depth, conf, via) in seen.items():
            name = qualified_from_node_id(store, node_id)
            item: ImpactItemPayload = {
                "type": "table",
                "name": name,
                "via": via,
                "confidence": conf,
            }
            if conf < 0.5:
                item["hop"] = depth
                low_confidence.append(item)
                continue
            if depth == 1:
                direct.append(item)
            elif depth == 2:
                two_hop.append(item)
            elif depth == 3:
                three_hop.append(item)

        total = len(seen)
        return {
            "target": {"table": table, "column": column},
            "database": database,
            "impact": {
                "direct": direct,
                "two_hop": two_hop,
                "three_hop": three_hop,
                "low_confidence": low_confidence,
            },
            "total_affected": total,
        }
    finally:
        store.close()


__all__ = ["impact_payload"]
