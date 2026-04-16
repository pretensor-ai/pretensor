"""Heuristic labels for domain clusters and persistence to Kuzu."""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pretensor.core.store import KuzuStore
from pretensor.intelligence.clustering import Cluster
from pretensor.search.index import _identifier_tokens

# Shared with MCP context resolution when a table has multiple IN_CLUSTER edges.
HEURISTIC_CLUSTER_DESCRIPTION = (
    "Grouped by graph connectivity (heuristic label; no LLM)."
)

__all__ = [
    "ClusterLabeler",
    "HEURISTIC_CLUSTER_DESCRIPTION",
    "LabeledCluster",
    "LlmClusterLabelingClient",
]

logger = logging.getLogger(__name__)


@dataclass
class LabeledCluster:
    """Cluster with persisted metadata."""

    cluster_id: str
    table_ids: list[str]
    label: str
    description: str
    cohesion_score: float


@runtime_checkable
class LlmClusterLabelingClient(Protocol):
    """Minimal async client for batched JSON labels."""

    async def label_clusters_json(self, user_prompt: str) -> str:
        """Return JSON array of objects with keys label, description (per cluster batch)."""
        ...


class ClusterLabeler:
    """Assign domain labels and write ``Cluster`` + ``IN_CLUSTER`` to Kuzu."""

    def __init__(
        self,
        store: KuzuStore,
        llm_client: LlmClusterLabelingClient | None = None,
    ) -> None:
        self._store = store
        self._llm = llm_client

    async def label_and_persist(
        self,
        clusters: list[Cluster],
        database_key: str,
        *,
        cluster_schema_patterns: Mapping[int, str] | None = None,
    ) -> list[LabeledCluster]:
        """Label clusters via heuristic and upsert into Kuzu."""
        out: list[LabeledCluster] = []
        batch_size = 5
        for batch_start in range(0, len(clusters), batch_size):
            batch = clusters[batch_start : batch_start + batch_size]
            labels_descs = [_heuristic_label(self._store, c) for c in batch]
            logger.info(
                "Cluster labeling (heuristic): %d clusters in this batch",
                len(batch),
            )

            for offset, cluster in enumerate(batch):
                idx = batch_start + offset
                cluster_id = f"{database_key}::cluster::{idx}"
                label, desc = (
                    labels_descs[offset]
                    if offset < len(labels_descs)
                    else (
                        f"Cluster {idx + 1}",
                        "Heuristic domain grouping from schema graph connectivity.",
                    )
                )
                schema_pattern = "unknown"
                if cluster_schema_patterns is not None:
                    schema_pattern = cluster_schema_patterns.get(idx, "unknown")
                self._store.upsert_cluster(
                    node_id=cluster_id,
                    database_key=database_key,
                    label=label,
                    description=desc,
                    cohesion_score=float(cluster.cohesion_score),
                    table_count=len(cluster.table_ids),
                    schema_pattern=schema_pattern,
                )
                for tid in cluster.table_ids:
                    self._store.upsert_in_cluster(tid, cluster_id)
                cluster.label = label
                out.append(
                    LabeledCluster(
                        cluster_id=cluster_id,
                        table_ids=list(cluster.table_ids),
                        label=label,
                        description=desc,
                        cohesion_score=cluster.cohesion_score,
                    )
                )
        return out


def _cluster_context(store: KuzuStore, cluster: Cluster) -> str:
    lines: list[str] = []
    for tid in cluster.table_ids:
        rows = store.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $id})
            RETURN t.schema_name, t.table_name, t.comment
            """,
            {"id": tid},
        )
        if not rows:
            lines.append(f"- {tid}: (missing node)")
            continue
        sn, tn, comment = rows[0]
        qual = f"{sn}.{tn}"
        desc = str(comment or "").strip() or "(no description)"
        lines.append(f"- {qual}: {desc}")
    return "\n".join(lines)


_NOUN_STOPWORDS: frozenset[str] = frozenset(
    {
        "tbl",
        "table",
        "tables",
        "fk",
        "id",
        "ids",
        "ref",
        "refs",
        "rel",
        "rels",
        "map",
        "maps",
        "link",
        "links",
        "detail",
        "details",
        "log",
        "logs",
    }
)

# Suffix-based deny: catches hub/audit/categorisation tokens that consistently
# beat the real semantic head on weighted votes (e.g. ``Businessentity`` eating
# ``Person`` on the AW person cluster, ``Orderhistory`` eating ``Order``).
_NOUN_SUFFIX_DENY: tuple[str, ...] = ("entity", "history", "type")


def _cluster_table_rows(
    store: KuzuStore, table_ids: list[str]
) -> list[tuple[str, str, str, list[str], list[str], int, str | None]]:
    """Return (node_id, schema_name, table_name, out_neighbors, in_neighbors, row_count, role) per table."""
    out: list[tuple[str, str, str, list[str], list[str], int, str | None]] = []
    for tid in table_ids:
        rows = store.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $id})
            OPTIONAL MATCH (t)-[:FK_REFERENCES]->(o:SchemaTable)
            OPTIONAL MATCH (i:SchemaTable)-[:FK_REFERENCES]->(t)
            RETURN t.schema_name, t.table_name, t.row_count, t.role,
                   collect(DISTINCT o.node_id), collect(DISTINCT i.node_id)
            """,
            {"id": tid},
        )
        if not rows:
            continue
        sn, tn, rc, role, outs, ins = rows[0]
        role_str = str(role) if role else None
        out.append(
            (
                str(tid),
                str(sn or ""),
                str(tn or ""),
                [str(x) for x in (outs or []) if x],
                [str(x) for x in (ins or []) if x],
                int(rc) if rc is not None else 0,
                role_str,
            )
        )
    return out


_ROLE_WEIGHT: dict[str, float] = {
    "dimension": 1.5,
    "fact": 1.25,
    "bridge": 0.4,
    "link": 0.4,
}


def _role_weight(role: str | None) -> float:
    """Multiplier for a table's contribution to the cluster-head vote.

    Dimensions and facts carry the semantic head noun; bridge/link tables
    (e.g. ``person.businessentity``) connect many neighbours but their names
    are plumbing, so they should not dominate the label. Unknown roles get
    neutral weight 1.0 for back-compat with older graphs.
    """
    if role is None:
        return 1.0
    return _ROLE_WEIGHT.get(role.lower(), 1.0)


def _strip_suffix(table_name: str) -> str:
    """Strip trailing _fk / _id (case-insensitive) while preserving original casing."""
    for suffix in ("_fk", "_id"):
        if table_name.lower().endswith(suffix):
            return table_name[: -len(suffix)]
    return table_name


def _frequent_noun(
    table_entries: list[tuple[str, str, float]], schema_stopwords: set[str]
) -> str | None:
    """Pick the highest-weighted domain-noun stem across table names.

    Each entry is ``(schema, table_name, weight)``. Weight reflects per-table
    importance (in-cluster FK degree scaled by log row_count, multiplied by a
    role weight — dimensions/facts dominate, bridges/links contribute less).
    Ties on weight are broken by raw occurrence count, then alphabetically —
    but alphabetical tie-break is a last resort, not the primary signal, so
    heads like ``Addresstype`` only win when no more central noun stands out.
    """
    weighted: dict[str, float] = {}
    occurrences: Counter[str] = Counter()
    tiebreak: dict[str, str] = {}
    for _schema, table_name, weight in table_entries:
        seen: set[str] = set()
        stripped = _strip_suffix(table_name)
        for token in _identifier_tokens(stripped):
            if not token or token.isdigit() or len(token) < 2:
                continue
            if token in _NOUN_STOPWORDS or token in schema_stopwords:
                continue
            stem = token[:-1] if len(token) > 3 and token.endswith("s") else token
            if stem in seen:
                continue
            if any(stem.endswith(s) for s in _NOUN_SUFFIX_DENY):
                continue
            seen.add(stem)
            weighted[stem] = weighted.get(stem, 0.0) + weight
            occurrences[stem] += 1
            # Preserve a canonical casing for title-casing: prefer longer original.
            prior = tiebreak.get(stem)
            if prior is None or len(token) > len(prior):
                tiebreak[stem] = token
    if not weighted:
        return None
    best = max(weighted.values())
    candidates = [s for s, w in weighted.items() if w == best]
    # Secondary: occurrence count; tertiary: alphabetical.
    candidates.sort(key=lambda s: (-occurrences[s], s))
    return tiebreak.get(candidates[0], candidates[0])


def _heuristic_label(store: KuzuStore, cluster: Cluster) -> tuple[str, str]:
    entries = _cluster_table_rows(store, list(cluster.table_ids))
    if not entries:
        return "Schema domain", HEURISTIC_CLUSTER_DESCRIPTION

    table_count = len(entries)
    cluster_set = {nid for nid, _, _, _, _, _, _ in entries}
    # Anchor: highest *role-weighted* in-cluster FK degree. Bridges/links
    # contribute at 0.4× so a 2-degree dimension (``person``) beats a
    # 3-degree bridge (``businessentity``) on the AW person cluster. Ties
    # fall through to row_count then name.
    best: tuple[float, int, str, str, str] | None = None
    # key shape: (-effective_degree, -row_count, schema, table, node_id)
    for nid, sn, tn, outs, ins, rc, role in entries:
        in_degree = sum(1 for o in outs if o in cluster_set)
        in_degree += sum(1 for i in ins if i in cluster_set)
        eff = in_degree * _role_weight(role)
        key = (-eff, -rc, sn.lower(), tn.lower(), nid)
        if best is None or key < best:
            best = key
    assert best is not None
    anchor_nid = best[4]
    rep_schema = next(
        sn for nid, sn, _tn, _o, _i, _rc, _role in entries if nid == anchor_nid
    )
    anchor_table = next(
        tn for nid, _sn, tn, _o, _i, _rc, _role in entries if nid == anchor_nid
    )
    if not rep_schema or not anchor_table:
        return "Schema domain", HEURISTIC_CLUSTER_DESCRIPTION

    # Schema names are stopworded so a common prefix (``analytics``, ``hr``)
    # doesn't eat the vote — but when a schema happens to share a name with
    # a table in the cluster (e.g. AW's ``person.person``), the noun *is*
    # the domain head and must survive. Skip those.
    table_names_lower = {
        tn.lower() for _nid, _sn, tn, _o, _i, _rc, _role in entries if tn
    }
    schema_stopwords = {
        sn.lower()
        for _nid, sn, _tn, _o, _i, _rc, _role in entries
        if sn and sn.lower() not in table_names_lower
    }
    # Weight each table by (in-cluster FK degree + 1) * log10(row_count + 10),
    # then multiply by a role weight. Dimensions/facts dominate the vote;
    # bridges/links contribute at 0.4× so their plumbing names (e.g.
    # ``businessentity``) don't eat the real head noun. The +1 / +10 floors
    # keep empty or leaf tables in the vote with non-zero weight.
    weighted_entries: list[tuple[str, str, float]] = []
    for _nid, sn, tn, outs, ins, rc, role in entries:
        deg = sum(1 for o in outs if o in cluster_set) + sum(
            1 for i in ins if i in cluster_set
        )
        weight = (deg + 1) * math.log10(max(rc, 0) + 10) * _role_weight(role)
        weighted_entries.append((sn, tn, weight))
    noun = _frequent_noun(weighted_entries, schema_stopwords)
    head = (
        noun.replace("_", " ").title() if noun else anchor_table.replace("_", " ")
    )
    suffix = f" cluster ({table_count} tables)" if table_count > 1 else ""
    label = f"{rep_schema}.{anchor_table} · {head}{suffix}"
    return label, HEURISTIC_CLUSTER_DESCRIPTION
