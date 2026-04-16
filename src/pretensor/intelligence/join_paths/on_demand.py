"""Graph adjacency, Dijkstra + Yen's K-shortest path scoring."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pretensor.core.store import KuzuStore

EdgeKind = Literal["fk", "inferred"]

_FK_CONFIDENCE = 1.0
_MAX_PATHS_PER_PAIR = 12
# Default Yen's K. The MCP traverse tool exposes this as ``top_k``.
_DEFAULT_TOP_K = 3
# Default cap on inferred-join hops per path; enforced inside Dijkstra so the
# search itself prunes weak chains rather than over-enumerating then filtering.
_DEFAULT_MAX_INFERRED_HOPS = 2


def edge_cost(edge: AdjEdge) -> float:
    """Lower is better. Hop baseline + inferred penalty + low-confidence penalty.

    A single inferred hop must cost more than an extra FK hop, else short
    inferred paths hide longer authoritative chains (e.g. pagila
    ``film → inventory → customer`` via an inferred ``store_id`` shortcut
    winning over the 3-hop all-FK ``film → inventory → rental → customer``).
    With ``inferred_penalty=1.5``, a 2-hop-with-1-inferred path costs
    ``1.0 + 2.5 = 3.5``, losing to the 3-hop FK chain at ``3.0`` by a
    comfortable margin. Inferred paths only win when the FK alternative
    is ≥4 hops or absent.
    """
    base = 1.0
    inferred_penalty = 1.5 if edge.kind == "inferred" else 0.0
    low_conf_penalty = max(0.0, 0.7 - edge.confidence)
    return base + inferred_penalty + low_conf_penalty


@dataclass(frozen=True, slots=True)
class JoinStep:
    """One hop along a join path (table → table via columns)."""

    from_schema: str
    from_table: str
    to_schema: str
    to_table: str
    from_column: str
    to_column: str
    edge_type: EdgeKind
    confidence: float


@dataclass(frozen=True, slots=True)
class StoredJoinPath:
    """Materialized path ready for MCP JSON."""

    path_id: str
    from_table_id: str
    to_table_id: str
    depth: int
    confidence: float
    ambiguous: bool
    steps: tuple[JoinStep, ...]
    semantic_label: str
    stale: bool = False
    # Dijkstra cost (lower = better). Persisted on-disk paths predating Yen
    # leave this at 0.0; in-memory paths from ``best_paths`` carry the real cost.
    cost: float = 0.0


@dataclass(frozen=True, slots=True)
class AdjEdge:
    to_id: str
    source_column: str
    target_column: str
    kind: EdgeKind
    confidence: float


def table_meta(store: KuzuStore, database_key: str) -> dict[str, tuple[str, str]]:
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable)
        WHERE t.database = $db
        RETURN t.node_id, t.schema_name, t.table_name
        """,
        {"db": database_key},
    )
    return {str(r[0]): (str(r[1]), str(r[2])) for r in rows if r[0] is not None}


def table_to_cluster(store: KuzuStore, database_key: str) -> dict[str, str]:
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable)-[:IN_CLUSTER]->(c:Cluster)
        WHERE c.database_key = $db
        RETURN t.node_id, c.node_id
        """,
        {"db": database_key},
    )
    return {str(r[0]): str(r[1]) for r in rows if r[0] is not None and r[1] is not None}


def build_adjacency(store: KuzuStore, database_key: str) -> dict[str, list[AdjEdge]]:
    adj: dict[str, list[AdjEdge]] = {}
    fk_rows = store.query_all_rows(
        """
        MATCH (a:SchemaTable)-[r:FK_REFERENCES]->(b:SchemaTable)
        WHERE a.database = $db AND b.database = $db
        RETURN a.node_id, b.node_id, r.source_column, r.target_column
        """,
        {"db": database_key},
    )
    for a, b, sc, tc in fk_rows:
        if a is None or b is None:
            continue
        sa, sb = str(a), str(b)
        s_col = str(sc) if sc is not None else ""
        t_col = str(tc) if tc is not None else ""
        adj.setdefault(sa, []).append(AdjEdge(sb, s_col, t_col, "fk", _FK_CONFIDENCE))
        adj.setdefault(sb, []).append(AdjEdge(sa, t_col, s_col, "fk", _FK_CONFIDENCE))

    inf_rows = store.query_all_rows(
        """
        MATCH (a:SchemaTable)-[r:INFERRED_JOIN]->(b:SchemaTable)
        WHERE a.database = $db AND b.database = $db
        RETURN a.node_id, b.node_id, r.source_column, r.target_column, r.confidence
        """,
        {"db": database_key},
    )
    for row in inf_rows:
        a, b, sc, tc, conf = row
        if a is None or b is None:
            continue
        sa, sb = str(a), str(b)
        w = float(conf) if conf is not None else 0.5
        s_col = str(sc) if sc is not None else ""
        t_col = str(tc) if tc is not None else ""
        adj.setdefault(sa, []).append(AdjEdge(sb, s_col, t_col, "inferred", w))
        adj.setdefault(sb, []).append(AdjEdge(sa, t_col, s_col, "inferred", w))
    return adj


def best_path(
    adj: dict[str, list[AdjEdge]],
    table_meta: dict[str, tuple[str, str]],
    start: str,
    goal: str,
    max_depth: int,
) -> StoredJoinPath | None:
    """Single best path. Asks Yen for top-2 to detect cost ties for ``ambiguous``."""
    paths = best_paths(adj, table_meta, start, goal, max_depth, top_k=2)
    if not paths:
        return None
    winner = paths[0]
    if len(paths) > 1 and abs(paths[0].cost - paths[1].cost) < 1e-9:
        # Tied-cost peer exists — surface ambiguity so callers re-enumerate.
        from dataclasses import replace
        winner = replace(winner, ambiguous=True)
    else:
        from dataclasses import replace
        winner = replace(winner, ambiguous=False)
    return winner


def best_paths(
    adj: dict[str, list[AdjEdge]],
    table_meta: dict[str, tuple[str, str]],
    start: str,
    goal: str,
    max_depth: int,
    *,
    top_k: int = _DEFAULT_TOP_K,
    edge_kinds: tuple[EdgeKind, ...] | None = None,
    max_inferred_hops: int = _DEFAULT_MAX_INFERRED_HOPS,
) -> list[StoredJoinPath]:
    """Return up to ``top_k`` shortest paths under the cost model in :func:`edge_cost`.

    Uses Yen's K-shortest paths over Dijkstra. ``edge_kinds`` restricts the set
    of allowed edge kinds (``("fk",)`` excludes inferred joins entirely).
    ``max_inferred_hops`` caps inferred-join hops per path.
    """
    if start == goal or top_k < 1:
        return []
    ranked = yen_k_shortest(
        adj,
        start,
        goal,
        max_depth,
        k=top_k,
        edge_kinds=edge_kinds,
        max_inferred_hops=max_inferred_hops,
    )
    if not ranked:
        return []
    out: list[StoredJoinPath] = []
    # Belt-and-suspenders dedup: yen_k's signature dedup is keyed on the edge
    # tuple, but two distinct edge sequences can still project to an identical
    # ``path_id`` (the sorted-JSON hash of the projected steps). Drop collisions
    # at the output boundary so MCP clients never see duplicate ``steps``.
    seen_path_ids: set[str] = set()
    for cost, edges in ranked:
        steps: list[JoinStep] = []
        cur = start
        conf = 1.0
        for e in edges:
            fs, ft = table_meta[cur]
            ts, tt = table_meta[e.to_id]
            steps.append(
                JoinStep(
                    from_schema=fs,
                    from_table=ft,
                    to_schema=ts,
                    to_table=tt,
                    from_column=e.source_column,
                    to_column=e.target_column,
                    edge_type=e.kind,
                    confidence=e.confidence,
                )
            )
            conf *= e.confidence
            cur = e.to_id
        steps_t = tuple(steps)
        length = len(edges)
        payload = steps_to_json_payload(steps_t)
        raw = json.dumps(payload, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        path_id = f"{start}::{goal}::{digest}"
        if path_id in seen_path_ids:
            continue
        seen_path_ids.add(path_id)
        label = f"{steps[0].from_table} → {steps[-1].to_table} ({length} hop)"
        out.append(
            StoredJoinPath(
                path_id=path_id,
                from_table_id=start,
                to_table_id=goal,
                depth=length,
                confidence=conf,
                cost=cost,
                ambiguous=len(ranked) > 1,
                steps=steps_t,
                semantic_label=label,
                stale=False,
            )
        )
    return out


def dijkstra_path(
    adj: dict[str, list[AdjEdge]],
    start: str,
    goal: str,
    max_depth: int,
    *,
    edge_kinds: tuple[EdgeKind, ...] | None = None,
    max_inferred_hops: int = _DEFAULT_MAX_INFERRED_HOPS,
    blocked_nodes: frozenset[str] = frozenset(),
    blocked_edges: frozenset[tuple[str, str, str, str]] = frozenset(),
) -> tuple[float, list[AdjEdge]] | None:
    """Single-source shortest path using :func:`edge_cost`.

    ``blocked_edges`` entries are ``(from_id, to_id, from_col, to_col)`` tuples
    so Yen's spur step can prohibit specific edges without disabling the
    underlying node entirely.
    """
    if start == goal:
        return (0.0, [])
    if start in blocked_nodes:
        return None
    counter = 0
    # heap entry: (cum_cost, tie, node, edges, inferred_hops)
    heap: list[tuple[float, int, str, list[AdjEdge], int]] = [
        (0.0, counter, start, [], 0)
    ]
    best_cost: dict[str, float] = {start: 0.0}
    while heap:
        cost, _, cur, edges, inf_hops = heapq.heappop(heap)
        if cur == goal:
            return (cost, edges)
        if cost > best_cost.get(cur, cost):
            continue
        if len(edges) >= max_depth:
            continue
        for e in adj.get(cur, []):
            if edge_kinds is not None and e.kind not in edge_kinds:
                continue
            if e.to_id in blocked_nodes:
                continue
            edge_key = (cur, e.to_id, e.source_column, e.target_column)
            if edge_key in blocked_edges:
                continue
            new_inf = inf_hops + (1 if e.kind == "inferred" else 0)
            if new_inf > max_inferred_hops:
                continue
            # Avoid loops: skip if we'd revisit a node already on this path.
            if any(prev.to_id == e.to_id for prev in edges) or e.to_id == start:
                continue
            new_cost = cost + edge_cost(e)
            if new_cost >= best_cost.get(e.to_id, float("inf")):
                # Yen needs to allow equal-cost spurs through the same node, so
                # only prune *strictly* worse routes; track the cheapest seen.
                continue
            best_cost[e.to_id] = new_cost
            counter += 1
            heapq.heappush(
                heap,
                (new_cost, counter, e.to_id, edges + [e], new_inf),
            )
    return None


def yen_k_shortest(
    adj: dict[str, list[AdjEdge]],
    start: str,
    goal: str,
    max_depth: int,
    k: int,
    *,
    edge_kinds: tuple[EdgeKind, ...] | None = None,
    max_inferred_hops: int = _DEFAULT_MAX_INFERRED_HOPS,
) -> list[tuple[float, list[AdjEdge]]]:
    """Yen's K-shortest-paths over :func:`dijkstra_path`."""
    if k < 1:
        return []
    first = dijkstra_path(
        adj,
        start,
        goal,
        max_depth,
        edge_kinds=edge_kinds,
        max_inferred_hops=max_inferred_hops,
    )
    if first is None:
        return []
    accepted: list[tuple[float, list[AdjEdge]]] = [first]
    # Min-heap of candidate spur paths: (cost, tie, edges)
    candidates: list[tuple[float, int, list[AdjEdge]]] = []
    seen_signatures: set[tuple[tuple[str, str, str, str, str], ...]] = {
        _path_signature(start, first[1])
    }
    counter = 0
    while len(accepted) < k:
        prev_cost, prev_edges = accepted[-1]
        # Walk each prefix of the previous path, treating the next edge as a
        # spur point. We block (a) the edge taken at the spur from any prior
        # accepted path that shares this prefix, and (b) the prefix nodes
        # themselves (so the spur stays simple).
        for i in range(len(prev_edges)):
            spur_node = start if i == 0 else prev_edges[i - 1].to_id
            root_edges = prev_edges[:i]
            # Block every prefix node *before* the spur — keeping the spur node
            # itself reachable so the new Dijkstra can leave from it.
            root_nodes: set[str] = set()
            if i > 0:
                root_nodes.add(start)
                for re in root_edges[:-1]:
                    root_nodes.add(re.to_id)
            blocked_edges: set[tuple[str, str, str, str]] = set()
            for _, edges in accepted:
                if len(edges) > i and edges[:i] == root_edges:
                    e = edges[i]
                    blocked_edges.add(
                        (spur_node, e.to_id, e.source_column, e.target_column)
                    )
            spur_path = dijkstra_path(
                adj,
                spur_node,
                goal,
                max_depth - i,
                edge_kinds=edge_kinds,
                max_inferred_hops=max_inferred_hops - sum(
                    1 for e in root_edges if e.kind == "inferred"
                ),
                blocked_nodes=frozenset(root_nodes),
                blocked_edges=frozenset(blocked_edges),
            )
            if spur_path is None:
                continue
            spur_cost, spur_edges = spur_path
            total_edges = root_edges + spur_edges
            sig = _path_signature(start, total_edges)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            root_cost = sum(edge_cost(e) for e in root_edges)
            counter += 1
            heapq.heappush(
                candidates,
                (root_cost + spur_cost, counter, total_edges),
            )
        if not candidates:
            break
        cost, _, edges = heapq.heappop(candidates)
        accepted.append((cost, edges))
    return accepted


def _path_signature(
    start: str, edges: Iterable[AdjEdge]
) -> tuple[tuple[str, str, str, str, str], ...]:
    """Identity tuple for Yen dedup. Includes ``kind`` so parallel FK + inferred
    edges on the same columns aren't conflated (otherwise yen_k would shadow the
    inferred alternative and callers would never see it)."""
    sig: list[tuple[str, str, str, str, str]] = []
    cur = start
    for e in edges:
        sig.append((cur, e.to_id, e.source_column, e.target_column, e.kind))
        cur = e.to_id
    return tuple(sig)


def steps_to_json_payload(steps: tuple[JoinStep, ...]) -> list[dict[str, Any]]:
    return [
        {
            "from_schema": s.from_schema,
            "from_table": s.from_table,
            "to_schema": s.to_schema,
            "to_table": s.to_table,
            "from_column": s.from_column,
            "to_column": s.to_column,
            "edge_type": s.edge_type,
            "confidence": s.confidence,
        }
        for s in steps
    ]


def reverse_stored_path(path: StoredJoinPath) -> StoredJoinPath:
    rev_steps: list[JoinStep] = []
    for s in reversed(path.steps):
        rev_steps.append(
            JoinStep(
                from_schema=s.to_schema,
                from_table=s.to_table,
                to_schema=s.from_schema,
                to_table=s.from_table,
                from_column=s.to_column,
                to_column=s.from_column,
                edge_type=s.edge_type,
                confidence=s.confidence,
            )
        )
    rt = tuple(rev_steps)
    payload = steps_to_json_payload(rt)
    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    path_id = f"{path.to_table_id}::{path.from_table_id}::{digest}"
    label = f"{rev_steps[0].from_table} → {rev_steps[-1].to_table} ({path.depth} hop)"
    return StoredJoinPath(
        path_id=path_id,
        from_table_id=path.to_table_id,
        to_table_id=path.from_table_id,
        depth=path.depth,
        confidence=path.confidence,
        ambiguous=path.ambiguous,
        steps=rt,
        semantic_label=label,
        stale=path.stale,
        cost=path.cost,
    )


def parse_steps_json(raw: str) -> tuple[JoinStep, ...]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(data, list):
        return ()
    steps: list[JoinStep] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        steps.append(
            JoinStep(
                from_schema=str(item.get("from_schema", "")),
                from_table=str(item.get("from_table", "")),
                to_schema=str(item.get("to_schema", "")),
                to_table=str(item.get("to_table", "")),
                from_column=str(item.get("from_column", "")),
                to_column=str(item.get("to_column", "")),
                edge_type=edge_kind(item.get("edge_type")),
                confidence=float(item.get("confidence", 1.0)),
            )
        )
    return tuple(steps)


def edge_kind(raw: Any) -> EdgeKind:
    if raw == "inferred":
        return "inferred"
    return "fk"


__all__ = [
    "AdjEdge",
    "EdgeKind",
    "JoinStep",
    "StoredJoinPath",
    "best_path",
    "best_paths",
    "build_adjacency",
    "dijkstra_path",
    "edge_cost",
    "parse_steps_json",
    "reverse_stored_path",
    "steps_to_json_payload",
    "table_meta",
    "table_to_cluster",
    "yen_k_shortest",
]
