"""MCP ``context`` tool and table resolution helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pretensor.config import GraphConfig
from pretensor.core.store import KuzuStore
from pretensor.intelligence.cluster_labeler import HEURISTIC_CLUSTER_DESCRIPTION
from pretensor.intelligence.shadow_alias import get_shadow_alias_node_ids
from pretensor.visibility.filter import VisibilityFilter

from ..payload_types import (
    ClusterInfo,
    ColumnInfo,
    ContextPayload,
    LineageRef,
    RelationshipInfo,
    snippet,
    stale_threshold_days,
    staleness_days,
)
from ..service_context import (
    get_effective_graph_config,
    get_effective_visibility_filter,
)
from ..service_registry import (
    graph_path_for_entry,
    load_registry,
    open_store_for_entry,
    resolve_registry_entry,
)

_SCHEMA_PATTERN_LABELS: dict[str, str] = {
    "star": "Star schema",
    "snowflake": "Snowflake schema",
    "constellation": "Constellation schema",
    "erd": "ER-style schema",
    "unknown": "Unknown schema pattern",
}


def _schema_pattern_display(raw: str) -> str:
    key = raw.strip().lower()
    if key in _SCHEMA_PATTERN_LABELS:
        return _SCHEMA_PATTERN_LABELS[key]
    return raw.replace("_", " ").title()


def _tags_from_kuzu_array(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        seen: set[str] = set()
        out: list[str] = []
        for x in raw:
            s = str(x).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out
    return []


def find_table_rows(
    store: KuzuStore,
    table: str,
    *,
    visibility_filter: VisibilityFilter | None = None,
) -> list[tuple[Any, ...]]:
    """Return SchemaTable rows matching bare name or ``schema.table``."""
    table = table.strip()
    if not table:
        return []
    _table_return = """
            RETURN t.node_id, t.schema_name, t.table_name, t.comment,
                   t.description, COALESCE(t.tags, CAST([] AS STRING[])) AS tags,
                   t.row_count,
                   t.entity_type, t.connection_name, t.database, t.table_type,
                   t.seq_scan_count, t.idx_scan_count, t.insert_count, t.update_count,
                   t.delete_count, t.is_partitioned, t.partition_key, t.grants_json,
                   t.access_read_count, t.access_write_count, t.days_since_last_access,
                   t.potentially_unused, t.table_bytes, t.clustering_key,
                   t.role, t.role_confidence, t.classification_signals,
                   t.has_external_consumers, t.test_count,
                   t.staleness_status, t.staleness_as_of
    """
    if "." in table:
        schema_name, _, table_name = table.partition(".")
        params = {"schema": schema_name, "tbl": table_name}
        rows_dot = store.query_all_rows(
            f"""
            MATCH (t:SchemaTable)
            WHERE t.schema_name = $schema AND t.table_name = $tbl
            {_table_return}
            """,
            params,
        )
        if rows_dot:
            return _filter_visible_table_rows(rows_dot, visibility_filter)
        # Case-insensitive fallback (e.g. Snowflake stores identifiers uppercase)
        rows_dot = store.query_all_rows(
            f"""
            MATCH (t:SchemaTable)
            WHERE lower(t.schema_name) = lower($schema)
              AND lower(t.table_name) = lower($tbl)
            {_table_return}
            """,
            params,
        )
        return _filter_visible_table_rows(rows_dot, visibility_filter)
    rows = store.query_all_rows(
        f"""
        MATCH (t:SchemaTable)
        WHERE t.table_name = $tbl
        {_table_return}
        """,
        {"tbl": table},
    )
    if rows:
        return _filter_visible_table_rows(rows, visibility_filter)
    # Case-insensitive fallback
    rows = store.query_all_rows(
        f"""
        MATCH (t:SchemaTable)
        WHERE lower(t.table_name) = lower($tbl)
        {_table_return}
        """,
        {"tbl": table},
    )
    return _filter_visible_table_rows(rows, visibility_filter)


def _filter_visible_table_rows(
    rows: list[tuple[Any, ...]],
    visibility_filter: VisibilityFilter | None,
) -> list[tuple[Any, ...]]:
    if visibility_filter is None:
        return rows
    out: list[tuple[Any, ...]] = []
    for m in rows:
        connection_name = str(m[8])
        schema_name = str(m[1])
        table_name = str(m[2])
        if visibility_filter.is_table_visible(connection_name, schema_name, table_name):
            out.append(m)
    return out


def suggest_table_candidates(
    store: KuzuStore,
    table: str,
) -> list[str]:
    """Return up to 5 qualified table names whose name matches *table* case-insensitively."""
    bare = table.split(".")[-1].strip()
    if not bare:
        return []
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable)
        WHERE lower(t.table_name) = lower($term)
        RETURN t.schema_name, t.table_name
        LIMIT 5
        """,
        {"term": bare},
    )
    return [f"{r[0]}.{r[1]}" for r in rows]


def columns_for_table_node_id(
    store: KuzuStore,
    table_node_id: str,
    *,
    visibility_filter: VisibilityFilter | None = None,
) -> list[ColumnInfo]:
    """Return column metadata linked from a ``SchemaTable`` node."""
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {node_id: $tid})-[:HAS_COLUMN]->(c:SchemaColumn)
        RETURN c.column_name, c.data_type, c.nullable, c.is_primary_key, c.is_foreign_key,
               c.comment, c.description, c.is_indexed, c.check_constraints_json,
               c.most_common_values_json, c.histogram_bounds_json, c.stats_correlation
        ORDER BY c.ordinal_position, c.column_name
        """,
        {"tid": table_node_id},
    )
    tbl_meta: tuple[str, str, str] | None = None
    if visibility_filter is not None:
        trows = store.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $tid})
            RETURN t.connection_name, t.schema_name, t.table_name
            """,
            {"tid": table_node_id},
        )
        if trows:
            cn, sn, tn = trows[0]
            tbl_meta = (str(cn), str(sn), str(tn))
    out: list[ColumnInfo] = []
    for row in rows:
        (
            col_name,
            dtype,
            nullable,
            is_pk,
            is_fk,
            cmt,
            col_desc_raw,
            indexed,
            checks_json,
            mcv_json,
            hb_json,
            scorr,
        ) = row
        cname = str(col_name)
        if visibility_filter is not None and tbl_meta is not None:
            cn, sn, tn = tbl_meta
            if not visibility_filter.visible_columns(cn, sn, tn, [cname]):
                continue
        checks: list[str] = []
        if checks_json:
            try:
                parsed = json.loads(str(checks_json))
                if isinstance(parsed, list):
                    checks = [str(x) for x in parsed]
            except json.JSONDecodeError:
                checks = []
        col_desc = str(col_desc_raw).strip() if col_desc_raw else ""
        cmt_s = str(cmt).strip() if cmt else ""
        item: ColumnInfo = {
            "column_name": cname,
            "name": cname,
            "data_type": str(dtype or ""),
            "nullable": bool(nullable) if nullable is not None else True,
            "is_primary_key": bool(is_pk) if is_pk is not None else False,
            "is_foreign_key": bool(is_fk) if is_fk is not None else False,
            "description": col_desc if col_desc else cmt_s,
        }
        if indexed is not None:
            item["is_indexed"] = bool(indexed)
        if checks:
            item["check_constraints"] = checks
        if mcv_json:
            try:
                mcv_parsed = json.loads(str(mcv_json))
                if isinstance(mcv_parsed, list):
                    item["most_common_values"] = [str(x) for x in mcv_parsed]
            except json.JSONDecodeError:
                pass
        if hb_json:
            try:
                hb_parsed = json.loads(str(hb_json))
                if isinstance(hb_parsed, list):
                    item["histogram_bounds"] = [str(x) for x in hb_parsed]
            except json.JSONDecodeError:
                pass
        if scorr is not None:
            item["stats_correlation"] = float(scorr)
        out.append(item)
    return out


def relationships_for_table(
    store: KuzuStore,
    node_id: str,
    *,
    visibility_filter: VisibilityFilter | None = None,
    shadow_alias_ids: frozenset[str] | None = None,
) -> list[RelationshipInfo]:
    src_cn = ""
    if visibility_filter is not None:
        meta = store.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $id})
            RETURN t.connection_name
            """,
            {"id": node_id},
        )
        if meta:
            src_cn = str(meta[0][0] or "")
    _shadows = shadow_alias_ids or frozenset()
    fk_rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {node_id: $id})-[r:FK_REFERENCES]->(t2:SchemaTable)
        RETURN t2.node_id, t2.schema_name, t2.table_name,
               r.source_column, r.target_column, r.constraint_name
        """,
        {"id": node_id},
    )
    inf_rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {node_id: $id})-[r:INFERRED_JOIN]->(t2:SchemaTable)
        RETURN t2.node_id, t2.schema_name, t2.table_name, r.source_column, r.target_column,
               r.source, r.confidence, r.reasoning
        """,
        {"id": node_id},
    )
    out: list[RelationshipInfo] = []
    # Group FK edges by constraint_name to collapse composite FKs into one row.
    fk_groups: dict[tuple[str, str | None], list[tuple[str, str]]] = {}
    fk_group_target: dict[tuple[str, str | None], str] = {}
    for t2_nid, sn, tn, scol, tcol, cname in fk_rows:
        if str(t2_nid) in _shadows:
            continue
        if visibility_filter is not None and not visibility_filter.is_table_visible(
            src_cn, str(sn), str(tn)
        ):
            continue
        target = f"{sn}.{tn}"
        cname_str = str(cname) if cname is not None else None
        key = (target, cname_str)
        fk_groups.setdefault(key, []).append((str(scol), str(tcol)))
        fk_group_target[key] = target
    for (target, cname_str), col_pairs in fk_groups.items():
        # Sort pairs for deterministic output across Kuzu return orders; pairing is preserved.
        ordered_pairs = sorted(col_pairs)
        src_cols = [p[0] for p in ordered_pairs]
        tgt_cols = [p[1] for p in ordered_pairs]
        rel: RelationshipInfo = {
            "target_table": target,
            "rel_type": "FK_REFERENCES",
            "source_column": src_cols[0],
            "target_column": tgt_cols[0],
            "source_columns": src_cols,
            "target_columns": tgt_cols,
            "constraint_name": cname_str,
            "source": None,
            "confidence": None,
            "reasoning": None,
        }
        out.append(rel)
    for row in inf_rows:
        t2_nid, sn, tn, scol, tcol, src, conf, reason = row
        if str(t2_nid) in _shadows:
            continue
        if visibility_filter is not None and not visibility_filter.is_table_visible(
            src_cn, str(sn), str(tn)
        ):
            continue
        s_str = str(scol) if scol is not None else ""
        t_str = str(tcol) if tcol is not None else ""
        inferred_rel: RelationshipInfo = {
            "target_table": f"{sn}.{tn}",
            "rel_type": "INFERRED_JOIN",
            "source_column": s_str,
            "target_column": t_str,
            "source_columns": [s_str] if s_str else [],
            "target_columns": [t_str] if t_str else [],
            "constraint_name": None,
            "source": str(src) if src is not None else None,
            "confidence": float(conf) if conf is not None else None,
            "reasoning": str(reason) if reason is not None else None,
        }
        out.append(inferred_rel)
    return out


def lineage_for_table(
    store: KuzuStore,
    node_id: str,
    *,
    visibility_filter: VisibilityFilter | None = None,
) -> tuple[list[LineageRef], list[LineageRef]]:
    """Inbound and outbound LINEAGE edges (structural table-level flow)."""
    base_cn = ""
    if visibility_filter is not None:
        meta = store.query_all_rows(
            """
            MATCH (t:SchemaTable {node_id: $id})
            RETURN t.connection_name
            """,
            {"id": node_id},
        )
        if meta:
            base_cn = str(meta[0][0] or "")
    in_rows = store.query_all_rows(
        """
        MATCH (src:SchemaTable)-[r:LINEAGE]->(t:SchemaTable {node_id: $id})
        RETURN src.schema_name, src.table_name, r.lineage_type, r.confidence, r.source
        ORDER BY src.schema_name, src.table_name
        """,
        {"id": node_id},
    )
    out_rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {node_id: $id})-[r:LINEAGE]->(dst:SchemaTable)
        RETURN dst.schema_name, dst.table_name, r.lineage_type, r.confidence, r.source
        ORDER BY dst.schema_name, dst.table_name
        """,
        {"id": node_id},
    )
    lineage_in: list[LineageRef] = []
    for sn, tn, ltype, conf, src in in_rows:
        if visibility_filter is not None and not visibility_filter.is_table_visible(
            base_cn, str(sn), str(tn)
        ):
            continue
        item: LineageRef = {
            "table": f"{sn}.{tn}",
            "lineage_type": str(ltype or ""),
            "confidence": float(conf) if conf is not None else 1.0,
            "source": str(src or ""),
        }
        lineage_in.append(item)
    lineage_out: list[LineageRef] = []
    for sn, tn, ltype, conf, src in out_rows:
        if visibility_filter is not None and not visibility_filter.is_table_visible(
            base_cn, str(sn), str(tn)
        ):
            continue
        item2: LineageRef = {
            "table": f"{sn}.{tn}",
            "lineage_type": str(ltype or ""),
            "confidence": float(conf) if conf is not None else 1.0,
            "source": str(src or ""),
        }
        lineage_out.append(item2)
    return lineage_in, lineage_out


def _lineage_markdown_section(
    lineage_in: list[LineageRef], lineage_out: list[LineageRef]
) -> str:
    """Return a markdown section for lineage edges, or empty string when none exist."""
    if not lineage_in and not lineage_out:
        return ""
    lines: list[str] = ["## Lineage", ""]
    if lineage_in:
        lines.append("**Feeds this table**")
        for ref in lineage_in:
            lt = ref.get("lineage_type", "")
            cf = ref.get("confidence")
            tbl = ref.get("table", "")
            conf_s = f" (confidence {cf:.2f})" if isinstance(cf, float) else ""
            lines.append(f"- `{tbl}` — {lt}{conf_s}")
        lines.append("")
    if lineage_out:
        lines.append("**This table feeds**")
        for ref in lineage_out:
            lt = ref.get("lineage_type", "")
            cf = ref.get("confidence")
            tbl = ref.get("table", "")
            conf_s = f" (confidence {cf:.2f})" if isinstance(cf, float) else ""
            lines.append(f"- `{tbl}` — {lt}{conf_s}")
    return "\n".join(lines).strip()


_DEPRECATION_ACCESS_DAYS = 90


def cluster_for_table_node_id(
    store: KuzuStore, table_node_id: str, database_key: str
) -> ClusterInfo | None:
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {node_id: $tid})-[:IN_CLUSTER]->(c:Cluster)
        WHERE c.database_key = $db
        RETURN c.node_id, c.label, c.description, c.cohesion_score, c.schema_pattern, c.stale
        ORDER BY CASE WHEN c.description = $heuristic THEN 1 ELSE 0 END,
                 c.cohesion_score DESC
        LIMIT 1
        """,
        {
            "tid": table_node_id,
            "db": database_key,
            "heuristic": HEURISTIC_CLUSTER_DESCRIPTION,
        },
    )
    if not rows:
        return None
    cid, lab, desc, coh, spattern, stale = rows[0]
    desc_str = str(desc or "")
    if desc_str == HEURISTIC_CLUSTER_DESCRIPTION:
        desc_str = ""
    info: ClusterInfo = {
        "cluster_id": str(cid),
        "label": str(lab or ""),
        "description": desc_str,
        "cohesion_score": float(coh) if coh is not None else 0.0,
    }
    if spattern is not None and str(spattern).strip():
        info["schema_pattern"] = str(spattern)
    if bool(stale):
        info["stale"] = True
        info["stale_warning"] = (
            "This cluster was last computed before recent schema changes. "
            "Re-run `pretensor reindex <dsn> --recompute-intelligence` "
            "for accurate cluster membership."
        )
    return info


def resolve_table_node_id(
    store: KuzuStore,
    table: str,
    database_key: str,
    *,
    visibility_filter: VisibilityFilter | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(node_id, error_message)`` for a bare or qualified table name."""
    matches = find_table_rows(store, table, visibility_filter=visibility_filter)
    if not matches:
        msg = f"No table matched {table!r}"
        candidates = suggest_table_candidates(store, table)
        if candidates:
            msg += f". Did you mean: {', '.join(candidates)}"
        return None, msg
    scoped = [m for m in matches if str(m[7]) == database_key]
    if not scoped:
        scoped = list(matches)
    if len(scoped) > 1:
        opts = [f"{m[1]}.{m[2]}" for m in scoped]
        return None, json.dumps(
            {"error": "Ambiguous table name; use schema.table", "candidates": opts}
        )
    return str(scoped[0][0]), None


def qualified_from_node_id(store: KuzuStore, node_id: str) -> str:
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {node_id: $id})
        RETURN t.schema_name, t.table_name
        """,
        {"id": node_id},
    )
    if not rows:
        return node_id
    sn, tn = rows[0]
    return f"{sn}.{tn}"


def shadow_aliases_for_base(
    store: KuzuStore, node_id: str, shadow_ids: frozenset[str]
) -> list[str]:
    """Return qualified names of shadow alias views that project from this base table."""
    if not shadow_ids:
        return []
    # Shadow aliases have a LINEAGE edge pointing *to* the view *from* the base.
    out_rows = store.query_all_rows(
        """
        MATCH (base:SchemaTable {node_id: $id})-[:LINEAGE]->(v:SchemaTable)
        RETURN v.node_id, v.schema_name, v.table_name
        """,
        {"id": node_id},
    )
    aliases: list[str] = []
    for vid, sn, tn in out_rows:
        if str(vid) in shadow_ids:
            aliases.append(f"{sn}.{tn}")
    return sorted(aliases)


def shadow_of_for_view(
    store: KuzuStore, node_id: str, shadow_ids: frozenset[str]
) -> str | None:
    """If *node_id* is a shadow alias, return the qualified name of its base table."""
    if node_id not in shadow_ids:
        return None
    rows = store.query_all_rows(
        """
        MATCH (src:SchemaTable)-[:LINEAGE]->(v:SchemaTable {node_id: $id})
        RETURN src.schema_name, src.table_name
        LIMIT 1
        """,
        {"id": node_id},
    )
    if not rows:
        return None
    sn, tn = rows[0]
    return f"{sn}.{tn}"


def entity_for_table(store: KuzuStore, node_id: str) -> tuple[str | None, str | None]:
    rows = store.query_all_rows(
        """
        MATCH (e:Entity)-[:REPRESENTS]->(t:SchemaTable {node_id: $id})
        RETURN e.name, e.description
        LIMIT 1
        """,
        {"id": node_id},
    )
    if not rows:
        return None, None
    name, desc = rows[0]
    return (str(name) if name is not None else None, str(desc) if desc else None)


def context_payload(
    graph_dir: Path,
    *,
    table: str,
    db: str | None = None,
    detail: Literal["summary", "standard", "full"] = "standard",
    config: GraphConfig | None = None,
    visibility_filter: VisibilityFilter | None = None,
) -> dict[str, Any]:
    """360° view for one physical table."""
    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, db)
    if entry is None and db is not None:
        return {"error": f"Unknown database connection or name: {db!r}"}
    if entry is None:
        return {
            "error": "Multiple indexed databases — pass `db` (connection name or logical database name)."
        }
    gp = graph_path_for_entry(entry)
    if not gp.exists():
        return {"error": f"Graph file missing for {entry.connection_name!r}: {gp}"}

    vf = visibility_filter or get_effective_visibility_filter()
    store = open_store_for_entry(entry)
    try:
        matches = find_table_rows(store, table, visibility_filter=vf)
        if not matches:
            msg = f"No table matched {table!r} in {entry.connection_name!r}"
            candidates = suggest_table_candidates(store, table)
            if candidates:
                msg += f". Did you mean: {', '.join(candidates)}"
            return {"error": msg}
        if len(matches) > 1:
            opts = [f"{m[1]}.{m[2]}" for m in matches]
            return {
                "error": "Ambiguous table name; use schema.table",
                "candidates": opts,
            }

        (
            node_id,
            schema_name,
            table_name,
            comment,
            table_desc_raw,
            tags_array_raw,
            row_count,
            entity_type,
            connection_name,
            database,
            table_type,
            seq_scan,
            idx_scan,
            ins_ct,
            upd_ct,
            del_ct,
            is_part,
            part_key,
            grants_json,
            acc_read,
            acc_write,
            days_since_acc,
            pot_unused,
            table_bytes,
            cluster_key,
            role_raw,
            role_conf_raw,
            class_sig_json,
            has_ext_raw,
            test_count_raw,
            staleness_status_raw,
            staleness_as_of_raw,
        ) = matches[0]

        qualified = f"{schema_name}.{table_name}"
        table_description = str(table_desc_raw).strip() if table_desc_raw else ""
        tags_list = _tags_from_kuzu_array(tags_array_raw)
        has_ext_bool: bool | None = None
        if has_ext_raw is not None:
            has_ext_bool = bool(has_ext_raw)
        test_count_val: int | None = None
        if test_count_raw is not None:
            try:
                test_count_val = int(test_count_raw)
            except (TypeError, ValueError):
                test_count_val = None
        staleness_status = (
            str(staleness_status_raw).strip() if staleness_status_raw else ""
        )
        staleness_as_of = (
            str(staleness_as_of_raw).strip() if staleness_as_of_raw else ""
        )
        description = (
            table_description if table_description else str(comment or "")
        )
        shadow_ids = get_shadow_alias_node_ids(store, str(database))
        rels = relationships_for_table(
            store, str(node_id), visibility_filter=vf,
            shadow_alias_ids=shadow_ids,
        )
        lineage_in, lineage_out = lineage_for_table(
            store, str(node_id), visibility_filter=vf
        )
        lineage_md = _lineage_markdown_section(lineage_in, lineage_out)
        ent_name, ent_desc = entity_for_table(store, str(node_id))
        cluster_info = cluster_for_table_node_id(store, str(node_id), str(database))
        alias_names = shadow_aliases_for_base(store, str(node_id), shadow_ids)
        shadow_of_name = shadow_of_for_view(store, str(node_id), shadow_ids)
        columns = columns_for_table_node_id(store, str(node_id), visibility_filter=vf)

        role_str = str(role_raw).strip() if role_raw else ""
        role_conf_f: float | None = None
        if role_conf_raw is not None:
            try:
                role_conf_f = float(role_conf_raw)
            except (TypeError, ValueError):
                role_conf_f = None
        class_signals: list[str] = []
        if class_sig_json:
            try:
                parsed = json.loads(str(class_sig_json))
                if isinstance(parsed, list):
                    class_signals = [str(x) for x in parsed if isinstance(x, str)]
            except json.JSONDecodeError:
                class_signals = []
        pattern_label = ""
        if cluster_info:
            raw_pattern = cluster_info.get("schema_pattern")
            if raw_pattern:
                pattern_label = _schema_pattern_display(str(raw_pattern))
        role_pretty = role_str.replace("_", " ").title() if role_str else ""
        class_summary = ""
        if role_pretty:
            if pattern_label:
                class_summary = (
                    f"This is a {role_pretty} table (confidence: {role_conf_f:.2f}) "
                    f"in a {pattern_label} cluster."
                    if role_conf_f is not None
                    else f"This is a {role_pretty} table in a {pattern_label} cluster."
                )
            else:
                class_summary = (
                    f"This is a {role_pretty} table (confidence: {role_conf_f:.2f})."
                    if role_conf_f is not None
                    else f"This is a {role_pretty} table."
                )

        cfg = get_effective_graph_config(config)
        days = staleness_days(entry.last_indexed_at)
        stale = days > stale_threshold_days(cfg)
        warn = (
            f"Graph index is {days} days old (threshold {stale_threshold_days(cfg)} days); "
            "run `pretensor reindex <dsn>` or `pretensor index`."
            if stale
            else None
        )

        dep_sig: str | None = None
        days_acc_int: int | None = (
            int(days_since_acc) if days_since_acc is not None else None
        )
        pot_u = bool(pot_unused) if pot_unused is not None else False
        if (
            pot_u
            and days_acc_int is not None
            and days_acc_int > _DEPRECATION_ACCESS_DAYS
            and not lineage_in
        ):
            dep_sig = (
                f"No inbound lineage and not accessed in {days_acc_int} days "
                "— likely deprecated"
            )

        base: ContextPayload = {
            "connection_name": str(connection_name),
            "database": str(database),
            "schema_name": str(schema_name),
            "table_name": str(table_name),
            "table_type": str(table_type) if table_type else None,
            "qualified_name": qualified,
            "description": description,
            "row_count": int(row_count) if row_count is not None else None,
            "entity_type": str(entity_type) if entity_type else None,
            "entity_name": ent_name,
            "entity_description": ent_desc,
            "role": role_str or None,
            "role_confidence": role_conf_f,
            "cluster": cluster_info,
            "columns": columns,
            "relationships": rels,
            "deprecation_signal": dep_sig,
            "staleness_warning": warn,
            "detail": detail,
        }
        # Only surface lineage when there's something to show. Empty arrays read as
        # "broken feature" on graphs without dbt manifests or view definitions.
        if lineage_in or lineage_out:
            base["lineage_in"] = lineage_in
            base["lineage_out"] = lineage_out
            if lineage_md:
                base["lineage_markdown"] = lineage_md
        if alias_names:
            base["aliases"] = alias_names
        if shadow_of_name is not None:
            base["shadow_of"] = shadow_of_name
        if tags_list:
            base["tags"] = tags_list
        if has_ext_bool is True:
            base["has_external_consumers"] = True
        if test_count_val is not None:
            base["test_count"] = test_count_val
        if staleness_status:
            base["staleness_status"] = staleness_status
        if staleness_as_of:
            base["staleness_as_of"] = staleness_as_of
        if detail == "full":
            base["classification_signals"] = class_signals
            base["classification_summary"] = class_summary
        elif detail == "standard":
            if class_summary:
                base["classification_summary"] = class_summary

        usage_stats: dict[str, int] = {}
        for label, val in (
            ("seq_scan_count", seq_scan),
            ("idx_scan_count", idx_scan),
            ("insert_count", ins_ct),
            ("update_count", upd_ct),
            ("delete_count", del_ct),
        ):
            if val is not None:
                usage_stats[label] = int(val)
        if usage_stats and detail != "summary":
            base["usage_stats"] = usage_stats

        if detail != "summary" and (
            is_part is True or (part_key is not None and str(part_key).strip() != "")
        ):
            base["partition"] = {
                "is_partitioned": bool(is_part) if is_part is not None else True,
                "partition_key": str(part_key) if part_key else None,
            }

        if grants_json and detail != "summary":
            try:
                gparsed = json.loads(str(grants_json))
                if isinstance(gparsed, list):
                    base["grants"] = [dict(x) for x in gparsed if isinstance(x, dict)]
            except json.JSONDecodeError:
                pass

        access_blob: dict[str, object] = {}
        if acc_read is not None:
            access_blob["read_operations_1y"] = int(acc_read)
        if acc_write is not None:
            access_blob["write_operations_1y"] = int(acc_write)
        if days_since_acc is not None:
            access_blob["days_since_last_access"] = int(days_since_acc)
        if pot_unused is not None:
            access_blob["potentially_unused_90d"] = bool(pot_unused)
        if table_bytes is not None:
            access_blob["table_bytes"] = int(table_bytes)
        if cluster_key:
            access_blob["clustering_key"] = str(cluster_key)
        if access_blob and detail != "summary":
            base["access_patterns"] = access_blob

        if detail == "summary":
            sum_blob: dict[str, object] = {
                "qualified_name": base["qualified_name"],
                "description": snippet(description, 120),
                "column_count": len(columns),
                "relationship_count": len(rels),
                "lineage_in_count": len(lineage_in),
                "lineage_out_count": len(lineage_out),
                "entity_name": ent_name,
            }
            if role_str:
                sum_blob["role"] = role_str
            if dep_sig:
                sum_blob["deprecation_signal"] = dep_sig
            return {
                "summary": sum_blob,
                "staleness_warning": warn,
            }
        return dict(base)
    finally:
        store.close()


__all__ = [
    "cluster_for_table_node_id",
    "columns_for_table_node_id",
    "context_payload",
    "entity_for_table",
    "find_table_rows",
    "lineage_for_table",
    "qualified_from_node_id",
    "relationships_for_table",
    "resolve_table_node_id",
    "shadow_aliases_for_base",
    "shadow_of_for_view",
    "suggest_table_candidates",
]
