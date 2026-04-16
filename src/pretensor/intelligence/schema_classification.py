"""Persist table roles and cluster-level schema patterns after clustering."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Literal, Protocol, runtime_checkable

from pretensor.core.store import KuzuStore
from pretensor.entities.classifier import (
    TableClassification,
    TableClassifier,
    TableClassifierInput,
)
from pretensor.intelligence.clustering import Cluster

logger = logging.getLogger(__name__)

__all__ = [
    "LlmTableClassificationClient",
    "classify_database_tables",
    "classify_database_tables_async",
    "compute_cluster_schema_patterns",
    "load_fk_reference_pairs",
]

SchemaPattern = Literal["star", "snowflake", "constellation", "erd", "unknown"]

_CLASSIFIER = TableClassifier()

# Precedence for ``schema_pattern`` when multiple signals apply:
# 1. constellation — multiple facts share at least one dimension (direct FK or via bridges)
# 2. snowflake — star-like hub plus at least one dimension-to-dimension FK in the cluster
# 3. star — one (or primary) fact with 3+ distinct dimensions via outgoing FKs
# 4. erd — connected tables with FKs but not a clear star/snowflake/constellation
# 5. role-only fallback — used when the cluster has no FK edges (introspection gap)


@runtime_checkable
class LlmTableClassificationClient(Protocol):
    """Async JSON batch classifier for low-confidence tables."""

    async def classify_tables_json(self, user_prompt: str) -> str:
        """Return JSON array aligned with batch order: {role, confidence, signals?}."""
        ...


def load_fk_reference_pairs(
    store: KuzuStore, database_key: str
) -> list[tuple[str, str]]:
    """Return all ``(src_table_id, dst_table_id)`` for ``FK_REFERENCES`` in one database."""
    rows = store.query_all_rows(
        """
        MATCH (a:SchemaTable {database: $db})-[r:FK_REFERENCES]->(b:SchemaTable {database: $db})
        RETURN a.node_id, b.node_id
        """,
        {"db": database_key},
    )
    return [(str(a), str(b)) for a, b in rows]


def _fk_degree_rows(store: KuzuStore, database_key: str) -> list[tuple[str, int, int]]:
    return store.query_all_rows(
        """
        MATCH (t:SchemaTable {database: $db})
        OPTIONAL MATCH (t)-[:FK_REFERENCES]->(o:SchemaTable)
        OPTIONAL MATCH (i:SchemaTable)-[:FK_REFERENCES]->(t)
        RETURN t.node_id, count(DISTINCT o.node_id), count(DISTINCT i.node_id)
        """,
        {"db": database_key},
    )


def _table_context_rows(store: KuzuStore, database_key: str) -> list[tuple]:
    return store.query_all_rows(
        """
        MATCH (t:SchemaTable {database: $db})
        RETURN t.node_id, t.table_name, t.schema_name, t.row_count,
               t.seq_scan_count, t.idx_scan_count, t.insert_count, t.update_count
        """,
        {"db": database_key},
    )


def _columns_by_table(store: KuzuStore, database_key: str) -> dict[str, list[str]]:
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {database: $db})-[:HAS_COLUMN]->(c:SchemaColumn)
        RETURN t.node_id, c.column_name, c.ordinal_position
        ORDER BY t.node_id, c.ordinal_position, c.column_name
        """,
        {"db": database_key},
    )
    out: dict[str, list[str]] = {}
    for tid, cname, _ord in rows:
        tid_s = str(tid)
        out.setdefault(tid_s, []).append(str(cname))
    return out


def _initial_classification(
    inp: TableClassifierInput,
) -> TableClassification:
    return _CLASSIFIER.classify(inp)


def _fk_pairs_for_cluster(
    table_ids: frozenset[str], all_pairs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    return [p for p in all_pairs if p[0] in table_ids and p[1] in table_ids]


def _dimensions_reachable_from(
    start: str,
    table_ids: frozenset[str],
    adj_out: dict[str, list[str]],
    dims: set[str],
) -> set[str]:
    """Directed reachability from ``start`` within ``table_ids``; return dimension nodes hit."""
    found: set[str] = set()
    q: deque[str] = deque([start])
    seen = {start}
    while q:
        u = q.popleft()
        if u in dims:
            found.add(u)
        for v in adj_out.get(u, ()):
            if v not in table_ids or v in seen:
                continue
            seen.add(v)
            q.append(v)
    return found


def _facts_share_dimension_via_paths(
    facts: set[str],
    table_ids: frozenset[str],
    fk_pairs: list[tuple[str, str]],
    dims: set[str],
) -> bool:
    """True if some pair of facts reaches a common dimension (paths may cross bridges)."""
    if len(facts) < 2:
        return False
    adj_out: dict[str, list[str]] = {}
    for a, b in fk_pairs:
        adj_out.setdefault(a, []).append(b)
    dim_sets: dict[str, set[str]] = {}
    for f in facts:
        dim_sets[f] = _dimensions_reachable_from(f, table_ids, adj_out, dims)
    fact_list = list(facts)
    for i in range(len(fact_list)):
        for j in range(i + 1, len(fact_list)):
            if dim_sets[fact_list[i]] & dim_sets[fact_list[j]]:
                return True
    return False


def _shared_dimension_direct(
    facts: set[str], dims: set[str], fk_pairs: list[tuple[str, str]]
) -> dict[str, set[str]]:
    """fact -> set of dimensions referenced by a direct FK from the fact."""
    out: dict[str, set[str]] = {f: set() for f in facts}
    for a, b in fk_pairs:
        if a in facts and b in dims:
            out.setdefault(a, set()).add(b)
    return out


def _schema_pattern_role_only(
    tset: frozenset[str],
    roles: dict[str, str],
    bridges: set[str],
) -> SchemaPattern:
    """Heuristic when the cluster has no FK edges (or none inside the cluster)."""
    facts = {tid for tid in tset if roles.get(tid) == "fact"}
    dims = {tid for tid in tset if roles.get(tid) == "dimension"}
    if len(facts) >= 2 and len(dims) >= 2:
        return "constellation"
    if facts and len(dims) >= 3:
        return "star"
    if bridges and (facts or dims):
        return "erd"
    if len(tset) >= 2:
        return "erd"
    return "unknown"


def compute_cluster_schema_patterns(
    clusters: list[Cluster],
    role_by_table: dict[str, TableClassification],
    fk_pairs: list[tuple[str, str]],
) -> dict[int, SchemaPattern]:
    """Map cluster index to ``schema_pattern`` (aligned with ``cluster::{idx}`` ids).

    Args:
        clusters: Leiden clusters.
        role_by_table: Latest classification per table node id.
        fk_pairs: All ``FK_REFERENCES`` (src, dst) for the logical database.
    """
    patterns: dict[int, SchemaPattern] = {}
    for idx, cluster in enumerate(clusters):
        tset = frozenset(cluster.table_ids)
        if not tset:
            patterns[idx] = "unknown"
            continue
        roles = {
            tid: role_by_table.get(tid, TableClassification("unknown", 0.0, ())).role
            for tid in tset
        }
        facts = {tid for tid, r in roles.items() if r == "fact"}
        dims = {tid for tid, r in roles.items() if r == "dimension"}
        bridges = {tid for tid, r in roles.items() if r == "bridge"}
        cluster_pairs = _fk_pairs_for_cluster(tset, fk_pairs)

        if not cluster_pairs:
            patterns[idx] = _schema_pattern_role_only(tset, roles, bridges)
            continue

        dim_dim = sum(1 for a, b in cluster_pairs if a in dims and b in dims)
        fact_to_dim = _shared_dimension_direct(facts, dims, cluster_pairs)
        facts_per_dim: dict[str, set[str]] = {}
        for f, dset in fact_to_dim.items():
            for d in dset:
                facts_per_dim.setdefault(d, set()).add(f)

        shared_direct = any(len(fs) >= 2 for fs in facts_per_dim.values())
        multi_hop_share = _facts_share_dimension_via_paths(
            facts, tset, cluster_pairs, dims
        )
        constellation = len(facts) >= 2 and (shared_direct or multi_hop_share)

        max_dim_from_one_fact = max((len(s) for s in fact_to_dim.values()), default=0)
        star_like = bool(facts and dims and max_dim_from_one_fact >= 3)

        pattern: SchemaPattern = "unknown"
        if constellation:
            pattern = "constellation"
        elif star_like:
            pattern = "snowflake" if dim_dim >= 1 else "star"
        elif len(tset) >= 2 and cluster_pairs:
            pattern = "erd"
        elif bridges and cluster_pairs:
            pattern = "erd"

        patterns[idx] = pattern
    return patterns


def classify_database_tables(
    store: KuzuStore,
    database_key: str,
) -> dict[str, TableClassification]:
    """Write classifier fields and return heuristic results."""
    store.ensure_schema()
    fk_rows = _fk_degree_rows(store, database_key)
    fk_out = {str(r[0]): int(r[1] or 0) for r in fk_rows}
    fk_in = {str(r[0]): int(r[2] or 0) for r in fk_rows}
    col_map = _columns_by_table(store, database_key)
    ctx_rows = _table_context_rows(store, database_key)
    results: dict[str, TableClassification] = {}

    for row in ctx_rows:
        tid = str(row[0])
        tname = str(row[1])
        sname = str(row[2])
        rcount = int(row[3]) if row[3] is not None else None
        seq_sc = int(row[4]) if row[4] is not None else None
        idx_sc = int(row[5]) if row[5] is not None else None
        ins = int(row[6]) if row[6] is not None else None
        upd = int(row[7]) if row[7] is not None else None
        cols = col_map.get(tid, [])
        inp = TableClassifierInput(
            name=tname,
            schema_name=sname,
            columns=cols,
            row_count=rcount,
            fk_out_degree=fk_out.get(tid, 0),
            fk_in_degree=fk_in.get(tid, 0),
            seq_scan_count=seq_sc,
            idx_scan_count=idx_sc,
            insert_count=ins,
            update_count=upd,
        )
        cls = _initial_classification(inp)
        results[tid] = cls
        store.set_table_classification(
            tid,
            role=cls.role,
            role_confidence=cls.confidence,
            classification_signals_json=json.dumps(
                list(cls.signals), ensure_ascii=False
            ),
        )

    return results


async def classify_database_tables_async(
    store: KuzuStore,
    database_key: str,
    *,
    llm_client: LlmTableClassificationClient | None = None,
) -> dict[str, TableClassification]:
    """Run heuristic classification. ``llm_client`` is accepted but ignored."""
    return classify_database_tables(store, database_key)
