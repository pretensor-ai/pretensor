"""MCP ``traverse`` tool: join paths and cross-database paths."""

from __future__ import annotations

import heapq
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pretensor.core.registry import RegistryEntry
from pretensor.core.store import KuzuStore
from pretensor.intelligence.join_paths import JoinPathEngine, StoredJoinPath
from pretensor.intelligence.shadow_alias import get_shadow_alias_node_ids
from pretensor.visibility.filter import VisibilityFilter
from pretensor.visibility.kuzu_helpers import visible_schema_table_node_ids

from ..payload_types import TraversePathPayload, TraverseStepPayload
from ..service_context import get_effective_visibility_filter
from ..service_registry import (
    graph_path_for_entry,
    load_registry,
    open_store_for_entry,
    resolve_registry_entry,
)
from .context import qualified_from_node_id, resolve_table_node_id

logger = logging.getLogger(__name__)


def sql_hint_from_steps(steps: list[TraverseStepPayload]) -> str:
    """Build SQL JOIN hints, grouping composite FK columns into one JOIN clause."""
    # Group consecutive steps that share the same tables and constraint_name
    # into a single JOIN with multiple ON conditions.
    groups: list[list[TraverseStepPayload]] = []
    for s in steps:
        cn = s.get("constraint_name")
        if (
            cn
            and groups
            and groups[-1][0].get("constraint_name") == cn
            and groups[-1][0].get("from_table") == s.get("from_table")
            and groups[-1][0].get("to_table") == s.get("to_table")
        ):
            groups[-1].append(s)
        else:
            groups.append([s])

    lines: list[str] = []
    for group in groups:
        first = group[0]
        left = first.get("from_table", "")
        right = first.get("to_table", "")
        left_short = left.split(".")[-1]
        right_short = right.split(".")[-1]
        conditions = [
            f"{left_short}.{g.get('from_column', '')} = {right_short}.{g.get('to_column', '')}"
            for g in group
        ]
        lines.append(f"JOIN {right_short} ON {' AND '.join(conditions)}")
    return "\n".join(lines)


def stored_path_to_payload(path: StoredJoinPath) -> TraversePathPayload:
    steps_out: list[TraverseStepPayload] = []
    for s in path.steps:
        steps_out.append(
            {
                "from_table": f"{s.from_schema}.{s.from_table}",
                "to_table": f"{s.to_schema}.{s.to_table}",
                "from_column": s.from_column,
                "to_column": s.to_column,
                "edge_type": "fk" if s.edge_type == "fk" else "inferred",
            }
        )
    out: TraversePathPayload = {
        "confidence": path.confidence,
        "ambiguous": path.ambiguous,
        "semantic_label": path.semantic_label,
        "steps": steps_out,
        "sql_hint": sql_hint_from_steps(steps_out),
    }
    if path.stale:
        out["stale"] = True
        out["stale_warning"] = (
            "This join path may be outdated due to recent schema changes. "
            "Verify the path before using it in SQL, or run "
            "`pretensor reindex <dsn> --recompute-intelligence`."
        )
    return out


def _qualified_visible(
    vf: VisibilityFilter | None,
    connection_name: str,
    qualified: str,
) -> bool:
    if vf is None:
        return True
    q = qualified.strip()
    if "." not in q:
        return vf.is_table_visible(connection_name, "", q)
    sn, _, tn = q.partition(".")
    return vf.is_table_visible(connection_name, sn, tn)


def traverse_steps_respect_visibility(
    steps: Sequence[Mapping[str, Any]],
    *,
    visibility_filter: VisibilityFilter | None,
    default_connection: str,
) -> bool:
    """True when every hop endpoint is visible (cross-db steps may set from_db/to_db).

    FK steps (``edge_type == "fk"``) are exempt: FK declarations are DDL-level
    join structure, not row data, so surfacing a path through a hidden FK
    waypoint does not leak data. Only inferred edges remain gated.
    """
    if visibility_filter is None:
        return True
    for s in steps:
        if str(s.get("edge_type", "")).strip() == "fk":
            continue
        ft = str(s.get("from_table", "")).strip()
        tt = str(s.get("to_table", "")).strip()
        fdb = str(s.get("from_db", "") or default_connection).strip()
        tdb = str(s.get("to_db", "") or default_connection).strip()
        if not _qualified_visible(visibility_filter, fdb, ft):
            return False
        if not _qualified_visible(visibility_filter, tdb, tt):
            return False
    return True


def dijkstra_join_path(
    store: KuzuStore,
    *,
    start_id: str,
    end_id: str,
    connection_name: str,
    max_depth: int,
    allowed_table_node_ids: set[str] | None = None,
    shadow_alias_ids: frozenset[str] | None = None,
    edge_kinds: tuple[str, ...] | None = None,
    max_inferred_hops: int = 2,
) -> tuple[list[tuple[str, str, str, str, str, str | None]], float] | None:
    """Return the lowest-cost edge list under :func:`edge_cost`.

    See :func:`pretensor.intelligence.join_paths.on_demand.edge_cost` for the
    canonical cost formula (base hop + inferred penalty + low-confidence
    penalty). Returns ``(edges, min_confidence)`` or *None*.

    Invariant: FK edges are never gated by ``allowed_table_node_ids``; only
    inferred edges are. Rationale: FK declarations are DDL-level join
    structure, not row-level data exposure, so pre-pruning them would drop
    authoritative join chains whenever a waypoint table is hidden. Do not
    re-introduce a gate on FK edges here — :func:`traverse_payload` relies
    on this invariant for the fk-only probe that promotes a FK chain over a
    stored inferred winner when the precompute pass wrote a short inferred
    shortcut (e.g. pagila ``film → customer`` across cluster boundaries).
    """
    from pretensor.intelligence.join_paths.on_demand import AdjEdge, edge_cost
    fk_rows = store.query_all_rows(
        """
        MATCH (a:SchemaTable)-[r:FK_REFERENCES]->(b:SchemaTable)
        WHERE a.connection_name = $cn AND b.connection_name = $cn
        RETURN a.node_id, b.node_id, r.source_column, r.target_column, r.constraint_name
        """,
        {"cn": connection_name},
    )
    inf_rows = store.query_all_rows(
        """
        MATCH (a:SchemaTable)-[r:INFERRED_JOIN]->(b:SchemaTable)
        WHERE a.connection_name = $cn AND b.connection_name = $cn
        RETURN a.node_id, b.node_id, r.source_column, r.target_column, r.confidence
        """,
        {"cn": connection_name},
    )
    _shadows = shadow_alias_ids or frozenset()
    # Adjacency mirrors on_demand.build_adjacency: both FK and inferred edges
    # are added **bidirectionally**. Join paths are symmetric (A joins B by
    # the FK iff B joins A by it), so pagila ``film → customer`` needs to
    # walk the inventory FK in reverse — only the directional adjacency
    # would strand the start node when it is the FK parent. Constraint
    # names are preserved on both directions for SQL hint rendering.
    adj: dict[str, list[tuple[AdjEdge, str | None]]] = {}
    for a, b, sc, tc, cn in fk_rows:
        sa, sb = str(a), str(b)
        if sa in _shadows or sb in _shadows:
            continue
        cname = str(cn) if cn is not None else None
        s_col = str(sc)
        t_col = str(tc)
        adj.setdefault(sa, []).append(
            (AdjEdge(sb, s_col, t_col, "fk", 1.0), cname)
        )
        adj.setdefault(sb, []).append(
            (AdjEdge(sa, t_col, s_col, "fk", 1.0), cname)
        )
    for a, b, sc, tc, conf in inf_rows:
        sa, sb = str(a), str(b)
        if sa in _shadows or sb in _shadows:
            continue
        w = float(conf) if conf is not None else 0.5
        s_col = str(sc)
        t_col = str(tc)
        adj.setdefault(sa, []).append(
            (AdjEdge(sb, s_col, t_col, "inferred", w), None)
        )
        adj.setdefault(sb, []).append(
            (AdjEdge(sa, t_col, s_col, "inferred", w), None)
        )

    counter = 0
    heap: list[
        tuple[
            float,
            int,
            str,
            list[tuple[str, str, str, str, str, str | None]],
            float,
            int,
        ]
    ] = []
    heapq.heappush(heap, (0.0, counter, start_id, [], 1.0, 0))
    visited: set[str] = set()
    while heap:
        cost, _, cur, path_edges, min_conf, inf_hops = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == end_id:
            return path_edges, min_conf
        if len(path_edges) >= max_depth:
            continue
        for ae, cname in adj.get(cur, []):
            if edge_kinds is not None and ae.kind not in edge_kinds:
                continue
            if (
                allowed_table_node_ids is not None
                and ae.kind != "fk"
                and ae.to_id not in allowed_table_node_ids
            ):
                continue
            if ae.to_id in visited:
                continue
            new_inf = inf_hops + (1 if ae.kind == "inferred" else 0)
            if new_inf > max_inferred_hops:
                continue
            edge = (cur, ae.to_id, ae.source_column, ae.target_column, ae.kind, cname)
            counter += 1
            heapq.heappush(
                heap,
                (
                    cost + edge_cost(ae),
                    counter,
                    ae.to_id,
                    path_edges + [edge],
                    min(min_conf, ae.confidence),
                    new_inf,
                ),
            )
    return None


def step_dict_from_edge(
    store: KuzuStore,
    edge: tuple[str, str, str, str, str, str | None],
    connection_label: str,
) -> TraverseStepPayload:
    fn, tn, scol, tcol, kind, cname = edge
    qf = qualified_from_node_id(store, fn)
    qt = qualified_from_node_id(store, tn)
    sc = str(scol)
    tc = str(tcol)
    step: TraverseStepPayload = {
        "from_table": qf,
        "to_table": qt,
        "from_column": sc,
        "to_column": tc,
        "from_db": connection_label,
        "to_db": connection_label,
        "via": f"{sc} = {tc}",
        "edge_type": "fk" if kind == "fk" else "inferred",
    }
    if cname is not None:
        step["constraint_name"] = cname
    return step


def traverse_cross_database(
    store: KuzuStore,
    *,
    from_table: str,
    to_table: str,
    from_database_key: str,
    to_database_key: str,
    from_connection: str,
    to_connection: str,
    max_depth: int,
    visibility_filter: VisibilityFilter | None = None,
) -> dict[str, Any]:
    """Path via confirmed ``SAME_ENTITY`` between two connections in one Kuzu file."""
    from_id, err = resolve_table_node_id(
        store,
        from_table,
        from_database_key,
        visibility_filter=visibility_filter,
    )
    if err is not None:
        try:
            return json.loads(err)
        except json.JSONDecodeError:
            return {"error": err}
    to_id, err2 = resolve_table_node_id(
        store,
        to_table,
        to_database_key,
        visibility_filter=visibility_filter,
    )
    if err2 is not None:
        try:
            return json.loads(err2)
        except json.JSONDecodeError:
            return {"error": err2}
    if from_id is None or to_id is None:
        return {"error": "Could not resolve one or both table references."}

    bridge_rows = store.query_all_rows(
        """
        MATCH (e1:Entity)-[r:SAME_ENTITY]->(e2:Entity)
        WHERE r.status = $st
        MATCH (e1)-[:REPRESENTS]->(ta:SchemaTable)
        MATCH (e2)-[:REPRESENTS]->(tb:SchemaTable)
        WHERE ta.connection_name = $fca AND tb.connection_name = $tca
        RETURN ta.node_id, tb.node_id, r.join_columns, r.score
        """,
        {"st": "confirmed", "fca": from_connection, "tca": to_connection},
    )
    bridge_rows += store.query_all_rows(
        """
        MATCH (e1:Entity)-[r:SAME_ENTITY]->(e2:Entity)
        WHERE r.status = $st
        MATCH (e1)-[:REPRESENTS]->(ta:SchemaTable)
        MATCH (e2)-[:REPRESENTS]->(tb:SchemaTable)
        WHERE ta.connection_name = $tca AND tb.connection_name = $fca
        RETURN tb.node_id, ta.node_id, r.join_columns, r.score
        """,
        {"st": "confirmed", "fca": from_connection, "tca": to_connection},
    )
    allowed_from = visible_schema_table_node_ids(
        store, from_connection, visibility_filter
    )
    allowed_to = visible_schema_table_node_ids(
        store, to_connection, visibility_filter
    )
    shadow_from = get_shadow_alias_node_ids(store, from_database_key)
    shadow_to = get_shadow_alias_node_ids(store, to_database_key)

    best: tuple[float, list[TraverseStepPayload]] | None = None
    cross_note = (
        "Cross-database paths require application-side joins or a federation layer; "
        "SQL cannot span these connections in a single query."
    )
    for ta_id, tb_id, jcols, sc in bridge_rows:
        ta_s, tb_s = str(ta_id), str(tb_id)
        left = dijkstra_join_path(
            store,
            start_id=from_id,
            end_id=ta_s,
            connection_name=from_connection,
            max_depth=max_depth,
            allowed_table_node_ids=allowed_from,
            shadow_alias_ids=shadow_from,
        )
        right = dijkstra_join_path(
            store,
            start_id=tb_s,
            end_id=to_id,
            connection_name=to_connection,
            max_depth=max_depth,
            allowed_table_node_ids=allowed_to,
            shadow_alias_ids=shadow_to,
        )
        if left is None or right is None:
            continue
        left_edges, left_conf = left
        right_edges, right_conf = right
        bridge_conf = float(sc) if sc is not None else 1.0
        conf = min(left_conf, right_conf, bridge_conf)

        steps_out: list[TraverseStepPayload] = [
            step_dict_from_edge(store, e, from_connection) for e in left_edges
        ]
        bridge_step: TraverseStepPayload = {
            "from_table": qualified_from_node_id(store, ta_s),
            "to_table": qualified_from_node_id(store, tb_s),
            "from_column": "",
            "to_column": "",
            "from_db": from_connection,
            "to_db": to_connection,
            "via": "SAME_ENTITY (confirmed)",
            "edge_type": "entity_link",
            "join_columns": str(jcols) if jcols is not None else None,
        }
        steps_out.append(bridge_step)
        steps_out.extend(
            step_dict_from_edge(store, e, to_connection) for e in right_edges
        )
        if best is None or conf > best[0]:
            best = (conf, steps_out)

    if best is None:
        return {
            "error": (
                "No cross-database path found via confirmed SAME_ENTITY links. "
                "Confirm entity links with `pretensor confirm`."
            ),
            "from": from_table,
            "to": to_table,
            "from_database": from_connection,
            "to_database": to_connection,
            "cross_database": True,
        }
    conf, steps_out = best
    if visibility_filter is not None and not traverse_steps_respect_visibility(
        steps_out,
        visibility_filter=visibility_filter,
        default_connection=from_connection,
    ):
        return {
            "error": (
                "No cross-database path found that respects visibility rules "
                "(a required hop crosses a hidden table)."
            ),
            "from": from_table,
            "to": to_table,
            "from_database": from_connection,
            "to_database": to_connection,
            "cross_database": True,
        }

    return {
        "from": from_table,
        "to": to_table,
        "from_database": from_connection,
        "to_database": to_connection,
        "cross_database": True,
        "cross_db_note": cross_note,
        "paths": [
            {
                "steps": steps_out,
                "confidence": conf,
                "requires_application_join": True,
                "semantic_label": "cross_database",
                "ambiguous": False,
                "sql_hint": "",
            }
        ],
        "used_precomputed": False,
        "warning": None,
    }


def traverse_payload(
    graph_dir: Path,
    *,
    from_table: str,
    to_table: str,
    database: str,
    to_database: str | None = None,
    max_depth: int = 4,
    top_k: int = 3,
    edge_types: tuple[str, ...] | None = None,
    max_inferred_hops: int = 2,
    visibility_filter: VisibilityFilter | None = None,
) -> dict[str, Any]:
    """Join paths between two tables (precomputed or on-demand Yen's K-shortest)."""
    edge_kinds: tuple[Any, ...] | None = None
    if edge_types:
        valid = {"fk", "inferred"}
        cleaned = tuple(t for t in edge_types if t in valid)
        if cleaned:
            edge_kinds = cleaned
    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, database)
    if entry is None:
        return {
            "error": "Unknown database connection or name; pass `database` "
            "(connection name or logical database)."
        }
    gp = graph_path_for_entry(entry)
    if not gp.exists():
        return {"error": f"Graph file missing: {gp}"}

    db_key = str(entry.database)
    vf = visibility_filter or get_effective_visibility_filter()
    store = open_store_for_entry(entry)
    try:
        allowed = visible_schema_table_node_ids(store, entry.connection_name, vf)
        to_entry_target: RegistryEntry | None = None
        if to_database is not None and str(to_database).strip():
            to_entry_target = resolve_registry_entry(reg, str(to_database).strip())
            if to_entry_target is None:
                return {
                    "error": f"Unknown to_database connection {to_database!r}.",
                }

        if to_entry_target is None:
            entries = reg.list_entries()
            if len(entries) > 1:
                for cand in entries:
                    if cand.connection_name == entry.connection_name:
                        continue
                    if graph_path_for_entry(cand) != gp:
                        continue
                    res = traverse_cross_database(
                        store,
                        from_table=from_table,
                        to_table=to_table,
                        from_database_key=db_key,
                        to_database_key=str(cand.database),
                        from_connection=entry.connection_name,
                        to_connection=cand.connection_name,
                        max_depth=max_depth,
                        visibility_filter=vf,
                    )
                    if "error" not in res:
                        return res

        elif (
            str(to_entry_target.database) != db_key
            or to_entry_target.connection_name != entry.connection_name
        ):
            return traverse_cross_database(
                store,
                from_table=from_table,
                to_table=to_table,
                from_database_key=db_key,
                to_database_key=str(to_entry_target.database),
                from_connection=entry.connection_name,
                to_connection=to_entry_target.connection_name,
                max_depth=max_depth,
                visibility_filter=vf,
            )

        from_id, err = resolve_table_node_id(
            store, from_table, db_key, visibility_filter=vf
        )
        if err is not None:
            try:
                return json.loads(err)
            except json.JSONDecodeError:
                return {"error": err}
        to_id, err2 = resolve_table_node_id(
            store, to_table, db_key, visibility_filter=vf
        )
        if err2 is not None:
            try:
                return json.loads(err2)
            except json.JSONDecodeError:
                return {"error": err2}
        if from_id is None or to_id is None:
            return {"error": "Could not resolve one or both table references."}

        engine = JoinPathEngine(store)
        stored = engine.load_stored_paths(db_key, from_id, to_id)
        used_fallback = False
        paths_payload: list[TraversePathPayload] = []
        visible_stored: list[StoredJoinPath] = []
        for p in stored:
            pay = stored_path_to_payload(p)
            steps = pay.get("steps") or []
            if traverse_steps_respect_visibility(
                steps,
                visibility_filter=vf,
                default_connection=entry.connection_name,
            ):
                paths_payload.append(pay)
                visible_stored.append(p)

        shadow_ids = get_shadow_alias_node_ids(store, db_key)

        # Safety net for stale indexes written before the precompute cross-cluster
        # depth was lifted. If every stored path for this pair goes through an
        # inferred edge, probe for a pure-FK chain within ``max_depth``; when one
        # exists, promote it to the top slot so the authoritative DDL-declared
        # join beats a short inferred shortcut. Fresh indexes (post this PR)
        # already persist the FK chain, so the probe is a no-op.
        if paths_payload and all(
            any(
                str(s.get("edge_type", "")) == "inferred"
                for s in (p.get("steps") or [])
            )
            for p in paths_payload
        ):
            fk_probe = dijkstra_join_path(
                store,
                start_id=from_id,
                end_id=to_id,
                connection_name=entry.connection_name,
                max_depth=max_depth,
                allowed_table_node_ids=allowed,
                shadow_alias_ids=shadow_ids,
                edge_kinds=("fk",),
                max_inferred_hops=0,
            )
            if fk_probe is not None:
                fk_edges, fk_min_conf = fk_probe
                fk_steps_out: list[TraverseStepPayload] = [
                    step_dict_from_edge(store, e, entry.connection_name)
                    for e in fk_edges
                ]
                if traverse_steps_respect_visibility(
                    fk_steps_out,
                    visibility_filter=vf,
                    default_connection=entry.connection_name,
                ):
                    fk_payload: TraversePathPayload = {
                        "confidence": fk_min_conf,
                        "ambiguous": False,
                        "semantic_label": "fk_chain_promoted",
                        "steps": fk_steps_out,
                        "sql_hint": sql_hint_from_steps(fk_steps_out),
                    }
                    paths_payload = [fk_payload, *paths_payload]
                    used_fallback = True
                    logger.info(
                        "traverse: fk_chain_promoted over stored inferred path "
                        "for %s -> %s (%s); reindex to refresh precomputed winners",
                        from_table,
                        to_table,
                        database,
                    )

        # Precompute only persists the single winning path per pair; when it is
        # flagged ambiguous we re-enumerate tied peers so callers see every
        # candidate they may need to pick between.
        if paths_payload and any(p.ambiguous for p in visible_stored):
            tied = engine.find_paths_on_demand(
                db_key,
                from_id,
                to_id,
                max_depth,
                top_k=top_k,
                edge_kinds=edge_kinds,
                max_inferred_hops=max_inferred_hops,
            )
            tied_payloads: list[TraversePathPayload] = []
            for tp in tied:
                tpay = stored_path_to_payload(tp)
                tsteps = tpay.get("steps") or []
                if traverse_steps_respect_visibility(
                    tsteps,
                    visibility_filter=vf,
                    default_connection=entry.connection_name,
                ):
                    tied_payloads.append(tpay)
            if len(tied_payloads) > 1:
                paths_payload = tied_payloads

        if not paths_payload:
            bfs = dijkstra_join_path(
                store,
                start_id=from_id,
                end_id=to_id,
                connection_name=entry.connection_name,
                max_depth=max_depth,
                allowed_table_node_ids=allowed,
                shadow_alias_ids=shadow_ids,
                edge_kinds=edge_kinds,
                max_inferred_hops=max_inferred_hops,
            )
            if bfs is not None:
                edges, min_conf = bfs
                steps_out: list[TraverseStepPayload] = [
                    step_dict_from_edge(store, e, entry.connection_name) for e in edges
                ]
                fallback_path: TraversePathPayload = {
                    "confidence": min_conf,
                    "ambiguous": False,
                    "semantic_label": "visibility_filtered",
                    "steps": steps_out,
                    "sql_hint": sql_hint_from_steps(steps_out),
                }
                paths_payload = [fallback_path]
                used_fallback = True
                logger.warning(
                    "traverse: visibility-filtered on-demand path for %s -> %s (%s)",
                    from_table,
                    to_table,
                    database,
                )

        if not paths_payload:
            on_demand_paths = engine.find_paths_on_demand(
                db_key,
                from_id,
                to_id,
                max_depth,
                top_k=top_k,
                edge_kinds=edge_kinds,
                max_inferred_hops=max_inferred_hops,
            )
            collected: list[TraversePathPayload] = []
            for od in on_demand_paths:
                pay2 = stored_path_to_payload(od)
                steps2 = pay2.get("steps") or []
                if traverse_steps_respect_visibility(
                    steps2,
                    visibility_filter=vf,
                    default_connection=entry.connection_name,
                ):
                    collected.append(pay2)
            if collected:
                paths_payload = collected
                used_fallback = True
                logger.warning(
                    "traverse: on-demand path for %s -> %s (%s)",
                    from_table,
                    to_table,
                    database,
                )

        if not paths_payload:
            return {
                "error": (
                    f"No join path found between {from_table!r} and {to_table!r} "
                    f"within {max_depth} hops that respects visibility rules. "
                    "Try increasing max_depth or adjust visibility.yml."
                ),
                "from": from_table,
                "to": to_table,
                "database": database,
            }
        return {
            "from": from_table,
            "to": to_table,
            "database": database,
            "paths": paths_payload,
            "used_precomputed": not used_fallback,
            "warning": (
                "Path computed on demand; re-run index to refresh precomputed join paths."
                if used_fallback
                else None
            ),
        }
    finally:
        store.close()


__all__ = ["traverse_payload"]
