"""Registry, keystore, and Kuzu open helpers for MCP tools."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, NamedTuple

from pretensor.core.dsn_crypto import DSNEncryptor
from pretensor.core.registry import GraphRegistry, RegistryEntry
from pretensor.core.store import KuzuStore
from pretensor.visibility.filter import VisibilityFilter

CapabilityState = Literal["not_attempted", "empty", "present"]


class GraphCounts(NamedTuple):
    """Per-database counts and capability flags derived from the graph in one pass."""

    table_count: int
    column_count: int
    row_count: int
    schemas: list[str]
    has_dbt_manifest: CapabilityState
    has_llm_enrichment: CapabilityState
    has_external_consumers: CapabilityState

logger = logging.getLogger(__name__)


def load_registry(graph_dir: Path) -> GraphRegistry:
    path = graph_dir / "registry.json"
    if not path.exists():
        return GraphRegistry(path)
    try:
        return GraphRegistry(path).load()
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Registry file %s is corrupt: %s", path, exc)
        return GraphRegistry(path)
    except Exception as exc:
        logger.warning("Failed to load registry from %s: %s", path, exc)
        return GraphRegistry(path)


def resolve_registry_entry(
    registry: GraphRegistry, db: str | None
) -> RegistryEntry | None:
    """Resolve ``db`` by ``connection_name`` or logical ``database``; None picks a unique entry."""
    entries = registry.list_entries()
    if not entries:
        return None
    if db is None:
        if len(entries) == 1:
            return entries[0]
        return None
    for entry in entries:
        if entry.connection_name == db or entry.database == db:
            return entry
    return None


def keystore_path(graph_dir: Path) -> Path:
    return graph_dir / "keystore"


def encryptor_for_dir(graph_dir: Path) -> DSNEncryptor:
    return DSNEncryptor(keystore_path(graph_dir))


def entry_plaintext_dsn(entry: RegistryEntry, graph_dir: Path) -> str:
    """Decrypt registry DSN when stored encrypted."""
    return entry.plaintext_dsn(
        encryptor_for_dir(graph_dir) if entry.dsn_encrypted else None
    )


def graph_path_for_entry(entry: RegistryEntry) -> Path:
    return Path(entry.unified_graph_path or entry.graph_path)


def open_store_for_entry(entry: RegistryEntry) -> KuzuStore:
    gp = graph_path_for_entry(entry)
    try:
        store = KuzuStore(gp)
        store.ensure_schema()
        return store
    except Exception as exc:
        raise RuntimeError(
            f"Cannot open graph file for connection {entry.connection_name!r} "
            f"({gp}): {exc}. The file may be corrupt — re-run `pretensor index`."
        ) from exc


def _column_count_for_connection(
    store: KuzuStore,
    *,
    connection_name: str | None = None,
) -> int:
    """Return the number of ``SchemaColumn`` nodes, optionally scoped to a connection."""
    if connection_name is None:
        r = store.query_all_rows(
            "MATCH (c:SchemaColumn) RETURN count(*)"
        )
    else:
        r = store.query_all_rows(
            "MATCH (c:SchemaColumn) WHERE c.connection_name = $cn RETURN count(*)",
            {"cn": connection_name},
        )
    if not r:
        return 0
    val = r[0][0]
    return int(val) if val is not None else 0


_DBT_MODEL_DEPENDENCY_LINEAGE_TYPE = "model_dependency"


def _has_dbt_manifest(store: KuzuStore, *, connection_name: str | None) -> bool:
    """True when any ``LINEAGE`` edge with ``lineage_type='model_dependency'`` exists.

    dbt's ``manifest.json`` is not stored as a graph node; its ``parent_map`` is
    projected as ``LINEAGE`` edges tagged with this ``lineage_type`` (see
    ``src/pretensor/enrichment/dbt/lineage.py``).
    """
    if connection_name is None:
        rows = store.query_all_rows(
            """
            MATCH ()-[r:LINEAGE]->()
            WHERE r.lineage_type = $lt
            RETURN 1 LIMIT 1
            """,
            {"lt": _DBT_MODEL_DEPENDENCY_LINEAGE_TYPE},
        )
    else:
        rows = store.query_all_rows(
            """
            MATCH (s:SchemaTable)-[r:LINEAGE]->()
            WHERE s.connection_name = $cn AND r.lineage_type = $lt
            RETURN 1 LIMIT 1
            """,
            {"cn": connection_name, "lt": _DBT_MODEL_DEPENDENCY_LINEAGE_TYPE},
        )
    return len(rows) > 0


def _has_llm_enrichment(store: KuzuStore, *, connection_name: str | None) -> bool:
    """True when any ``Entity`` node exists (LLM enrichment output)."""
    if connection_name is None:
        rows = store.query_all_rows("MATCH (e:Entity) RETURN 1 LIMIT 1")
    else:
        rows = store.query_all_rows(
            "MATCH (e:Entity) WHERE e.connection_name = $cn RETURN 1 LIMIT 1",
            {"cn": connection_name},
        )
    return len(rows) > 0


def counts_for_graph(
    store: KuzuStore,
    *,
    connection_name: str | None = None,
    visibility_filter: VisibilityFilter | None = None,
    entry: RegistryEntry | None = None,
) -> GraphCounts:
    """Return per-database counts and capability flags in a single store pass.

    Optionally scoped to one connection. Visibility filtering applies to table
    counts, ``schemas``, and ``has_external_consumers`` (hidden tables do not
    contribute); it does not affect the dbt/Entity probes, which are graph-wide.
    """
    params = {"cn": connection_name} if connection_name is not None else {}
    cn_clause = "WHERE t.connection_name = $cn" if connection_name is not None else ""
    rows = store.query_all_rows(
        f"""
        MATCH (t:SchemaTable)
        {cn_clause}
        RETURN t.connection_name, t.schema_name, t.table_name, t.row_count,
               t.has_external_consumers
        """,
        params,
    )
    total = 0
    row_sum = 0
    schemas: set[str] = set()
    has_external = False
    for cn, sn, tn, rc, hec in rows:
        cns = str(cn or "")
        sns = str(sn or "")
        tns = str(tn or "")
        if visibility_filter is not None and not visibility_filter.is_table_visible(
            cns, sns, tns
        ):
            continue
        total += 1
        if rc is not None:
            row_sum += int(rc)
        if sns:
            schemas.add(sns)
        if bool(hec):
            has_external = True
    cc = _column_count_for_connection(store, connection_name=connection_name)
    dbt_present = _has_dbt_manifest(store, connection_name=connection_name)
    llm_present = _has_llm_enrichment(store, connection_name=connection_name)
    # Trinary states use entry provenance when available: "not_attempted" means
    # the user never ran that enrichment pass (so callers shouldn't infer absence
    # of signal). An indexer that did attempt it returns "empty" vs "present".
    dbt_attempted = entry is not None and entry.dbt_manifest_path is not None
    if not dbt_attempted and entry is not None:
        has_dbt: CapabilityState = "not_attempted"
    else:
        has_dbt = "present" if dbt_present else ("empty" if dbt_attempted else "not_attempted")
    llm_attempted = entry is not None and entry.llm_enrichment_ran
    if entry is None:
        has_llm: CapabilityState = "present" if llm_present else "empty"
    else:
        has_llm = "present" if llm_present else ("empty" if llm_attempted else "not_attempted")
    # External-consumer scanner is not shipped yet; unify the "never checked"
    # default with the dbt/LLM flags so callers don't have to special-case it.
    has_ext: CapabilityState = "present" if has_external else "not_attempted"
    return GraphCounts(
        table_count=total,
        column_count=cc,
        row_count=row_sum,
        schemas=sorted(schemas),
        has_dbt_manifest=has_dbt,
        has_llm_enrichment=has_llm,
        has_external_consumers=has_ext,
    )


__all__ = [
    "GraphCounts",
    "counts_for_graph",
    "encryptor_for_dir",
    "entry_plaintext_dsn",
    "graph_path_for_entry",
    "keystore_path",
    "load_registry",
    "open_store_for_entry",
    "resolve_registry_entry",
]
