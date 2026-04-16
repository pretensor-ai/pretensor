"""Write ``Entity`` nodes, ``REPRESENTS`` edges, and ``entity_type`` on tables."""

from __future__ import annotations

from pretensor.connectors.models import SchemaSnapshot, Table
from pretensor.core.ids import entity_node_id, table_node_id
from pretensor.core.store import KuzuStore
from pretensor.entities.llm_extract import ExtractedEntity
from pretensor.graph_models.entity import EntityNode

__all__ = ["EntityBuilder"]


def _resolve_table(snapshot: SchemaSnapshot, ref: str) -> Table | None:
    """Match ``schema.name`` or bare ``name`` (first match)."""
    ref = ref.strip()
    for t in snapshot.tables:
        full = f"{t.schema_name}.{t.name}"
        if ref == full or ref == t.name:
            return t
    return None


class EntityBuilder:
    """Persists extracted business entities into Kuzu."""

    def build(
        self,
        entities: list[ExtractedEntity],
        store: KuzuStore,
        snapshot: SchemaSnapshot,
    ) -> None:
        """Upsert entities and link them to existing ``SchemaTable`` nodes.

        Args:
            entities: Parsed LLM output (grouped tables per business entity).
            store: Open Kuzu store (schema must exist; table nodes should exist).
            snapshot: Used to resolve table references and stable node ids.
        """
        conn = snapshot.connection_name
        db = snapshot.database
        known = {table_node_id(conn, t.schema_name, t.name) for t in snapshot.tables}

        for ext in entities:
            eid = entity_node_id(conn, ext.name)
            store.upsert_entity(
                EntityNode(
                    node_id=eid,
                    connection_name=conn,
                    database=db,
                    name=ext.name,
                    description=ext.description,
                )
            )
            for ref in ext.tables:
                table = _resolve_table(snapshot, ref)
                if table is None:
                    continue
                tid = table_node_id(conn, table.schema_name, table.name)
                if tid not in known:
                    continue
                store.upsert_represents(eid, tid)
                store.set_table_entity_type(tid, ext.name)
