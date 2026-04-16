"""Kuzu helpers for visibility-scoped node sets."""

from __future__ import annotations

from pretensor.core.store import KuzuStore
from pretensor.visibility.filter import VisibilityFilter

__all__ = ["visible_schema_table_node_ids"]


def visible_schema_table_node_ids(
    store: KuzuStore,
    connection_name: str,
    visibility_filter: VisibilityFilter | None,
) -> set[str] | None:
    """Return visible ``SchemaTable`` node ids, or None when filtering is disabled."""
    if visibility_filter is None:
        return None
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {connection_name: $cn})
        RETURN t.node_id, t.schema_name, t.table_name
        """,
        {"cn": connection_name},
    )
    out: set[str] = set()
    for nid, sn, tn in rows:
        if visibility_filter.is_table_visible(
            connection_name, str(sn or ""), str(tn or "")
        ):
            out.add(str(nid))
    return out
