"""MCP markdown resources (databases, per-db overview, clusters, metrics)."""

from __future__ import annotations

import json
from pathlib import Path

from pretensor.intelligence.cluster_labeler import HEURISTIC_CLUSTER_DESCRIPTION
from pretensor.mcp.payload_types import iso_format, stale_threshold_days, staleness_days
from pretensor.mcp.service_context import (
    get_effective_graph_config,
    get_effective_visibility_filter,
)
from pretensor.mcp.service_registry import (
    counts_for_graph,
    graph_path_for_entry,
    load_registry,
    open_store_for_entry,
    release_store,
    resolve_registry_entry,
)
from pretensor.mcp.tools.list import list_databases_payload
from pretensor.visibility.filter import VisibilityFilter


def clusters_resource_markdown(
    graph_dir: Path,
    name: str,
    *,
    visibility_filter: VisibilityFilter | None = None,
) -> str:
    """Markdown for ``pretensor://db/{name}/clusters``."""
    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, name)
    if entry is None:
        return f"# Unknown database `{name}`\n\nNo matching registry entry."
    gp = graph_path_for_entry(entry)
    if not gp.exists():
        return f"# Clusters — {name}\n\n_Graph file missing._"

    db_key = str(entry.database)
    vf = visibility_filter or get_effective_visibility_filter()
    cn = str(entry.connection_name)
    store = open_store_for_entry(entry)
    try:
        rows = store.query_all_rows(
            """
            MATCH (t:SchemaTable)-[:IN_CLUSTER]->(c:Cluster)
            WHERE c.database_key = $db
            RETURN c.label, c.description, c.cohesion_score,
                   collect(concat(t.schema_name, '.', t.table_name)) AS tables
            ORDER BY c.cohesion_score DESC
            """,
            {"db": db_key},
        )
        lines = [
            f"# Domain Clusters — {entry.connection_name}",
            "",
            f"_Logical database:_ `{db_key}`",
            "",
        ]
        if not rows:
            lines.append("_No clusters indexed yet; re-run `pretensor index`._")
            return "\n".join(lines)

        for lab, desc, coh, tables in rows:
            tlist = [str(x) for x in (tables or []) if x]
            if vf is not None:
                filtered: list[str] = []
                for qual in tlist:
                    if "." not in qual:
                        continue
                    sn, _, tn = qual.partition(".")
                    if vf.is_table_visible(cn, sn, tn):
                        filtered.append(qual)
                tlist = sorted(set(filtered))
            else:
                tlist = sorted(set(tlist))
            cohesion = float(coh) if coh is not None else 0.0
            lines.append(f"## {lab} _(cohesion: {cohesion:.2f})_")
            lines.append("")
            if desc and str(desc) != HEURISTIC_CLUSTER_DESCRIPTION:
                lines.append(str(desc))
                lines.append("")
            show = tlist[:40]
            suffix = f" and {len(tlist) - 40} more" if len(tlist) > 40 else ""
            lines.append("**Tables:** " + ", ".join(show) + suffix)
            lines.append("")
        lines.append(
            f"*Last indexed: {iso_format(entry.last_indexed_at)}. "
            "Use `context` for table detail or `traverse` for join paths.*"
        )
        return "\n".join(lines)
    finally:
        release_store(store)


def databases_resource_markdown(
    graph_dir: Path,
    *,
    visibility_filter: VisibilityFilter | None = None,
) -> str:
    """Markdown for ``pretensor://databases``."""
    data = list_databases_payload(graph_dir, visibility_filter=visibility_filter)
    lines = ["# Indexed databases", ""]
    for item in data["databases"]:
        lines.append(f"## {item['name']}")
        # ``database`` is omitted from the payload when it equals ``name`` (see
        # list_databases_payload); fall back to ``name`` for the markdown row.
        lines.append(f"- **Database:** {item.get('database', item['name'])}")
        lines.append(f"- **Tables:** {item['table_count']}")
        lines.append(f"- **Row count (sum):** {item['row_count']}")
        schemas = item.get("schemas") or []
        lines.append("- **Schemas:** " + (", ".join(schemas) if schemas else "_none_"))
        lines.append(
            f"- **dbt manifest:** {item.get('has_dbt_manifest', 'not_attempted')}"
        )
        lines.append(
            f"- **LLM enrichment:** {item.get('has_llm_enrichment', 'not_attempted')}"
        )
        lines.append(
            f"- **External consumers:** {item.get('has_external_consumers', 'not_attempted')}"
        )
        lines.append(f"- **Last indexed:** {item['last_indexed']}")
        lines.append(f"- **Stale (>7d):** {item['is_stale']}")
        lines.append(f"- **Graph:** `{item['graph_path']}`")
        lines.append("")
    if len(data["databases"]) == 0:
        lines.append("_No entries in registry._")
    return "\n".join(lines)


def db_overview_resource_markdown(
    graph_dir: Path,
    name: str,
    *,
    visibility_filter: VisibilityFilter | None = None,
) -> str:
    """Markdown for ``pretensor://db/{name}/overview``."""
    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, name)
    if entry is None:
        return f"# Unknown database `{name}`\n\nNo matching registry entry."
    gp = graph_path_for_entry(entry)
    entity_count = 0
    table_count = 0
    column_count = 0
    row_sum = 0
    vf = visibility_filter or get_effective_visibility_filter()
    if gp.exists():
        store = open_store_for_entry(entry)
        try:
            counts = counts_for_graph(
                store,
                connection_name=entry.connection_name,
                visibility_filter=vf,
            )
            table_count = counts.table_count
            column_count = counts.column_count
            row_sum = counts.row_count
            er = store.query_all_rows("MATCH (e:Entity) RETURN count(*)")
            if er:
                entity_count = int(er[0][0])
        finally:
            release_store(store)
    days = staleness_days(entry.last_indexed_at)
    threshold = stale_threshold_days(get_effective_graph_config())
    stale = days > threshold
    lines = [
        f"# Overview: {entry.connection_name}",
        "",
        f"- **Logical database:** {entry.database}",
        f"- **Tables:** {table_count}",
        f"- **Columns (graph):** {column_count}",
        f"- **Entities:** {entity_count}",
        f"- **Row count (sum over tables):** {row_sum}",
        f"- **Last indexed:** {iso_format(entry.last_indexed_at)}",
        f"- **Staleness (days):** {days} (warn if >{threshold})",
        f"- **Graph file:** `{entry.graph_path}`",
        "",
    ]
    if stale:
        lines.append(
            f"> **Warning:** Graph index is {days} days old; run "
            "`pretensor reindex <dsn>` or `pretensor index`."
        )
    return "\n".join(lines)


def metrics_resource_markdown(
    graph_dir: Path,
    name: str,
    *,
    visibility_filter: VisibilityFilter | None = None,
) -> str:
    """Markdown for ``pretensor://db/{name}/metrics``."""
    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, name)
    if entry is None:
        return f"# Unknown database `{name}`\n\nNo matching registry entry."
    gp = graph_path_for_entry(entry)
    if not gp.exists():
        return f"# Metrics — {name}\n\n_Graph file missing._"

    db_key = str(entry.database)
    cn = str(entry.connection_name)
    vf = visibility_filter or get_effective_visibility_filter()
    store = open_store_for_entry(entry)
    try:
        rows = store.query_all_rows(
            """
            MATCH (m:MetricTemplate {connection_name: $cn, database: $db})
            OPTIONAL MATCH (m)-[:METRIC_DEPENDS]->(t:SchemaTable)
            WITH m, collect(concat(t.schema_name, '.', t.table_name)) AS dep_tables
            RETURN m.name, m.display_name, m.description, m.sql_template, m.validated,
                   m.validation_errors_json, m.generated_at_iso, m.stale, m.dialect, dep_tables
            ORDER BY m.name
            """,
            {"cn": cn, "db": db_key},
        )
        lines = [
            f"# Metric templates — {entry.connection_name}",
            "",
            f"_Logical database:_ `{db_key}`",
            "",
        ]
        if not rows:
            lines.append(
                "_No `MetricTemplate` nodes yet. The default OSS indexing flow does "
                "not create them automatically; this resource only lists templates "
                "when they are already present in the graph._"
            )
            return "\n".join(lines)

        for (
            mname,
            dname,
            desc,
            sql,
            ok,
            errs_json,
            gen_iso,
            stale,
            dialect,
            dep_tables,
        ) in rows:
            tlist_raw = sorted(
                {str(x) for x in (dep_tables or []) if x is not None and str(x).strip()}
            )
            if vf is not None:
                tlist_vis = []
                for qual in tlist_raw:
                    if "." not in qual:
                        continue
                    sn, _, tn = qual.partition(".")
                    if vf.is_table_visible(cn, sn, tn):
                        tlist_vis.append(qual)
                if not tlist_vis and tlist_raw:
                    continue
                tlist = tlist_vis
            else:
                tlist = tlist_raw
            lines.append(f"## `{mname}`")
            lines.append("")
            if dname:
                lines.append(f"**Display:** {dname}")
                lines.append("")
            if desc:
                lines.append(str(desc))
                lines.append("")
            validated = bool(ok) if ok is not None else False
            lines.append(f"- **Validated:** {validated}")
            if dialect:
                lines.append(f"- **Dialect:** {dialect}")
            if stale:
                lines.append(
                    "- **Stale:** yes — graph changed since this template was generated; "
                    "re-run full `pretensor index` or `--recompute-intelligence` on reindex."
                )
            if gen_iso:
                lines.append(f"- **Generated:** {gen_iso}")
            if tlist:
                lines.append(f"- **Tables:** {', '.join(tlist)}")
            lines.append("")
            lines.append("```sql")
            lines.append(str(sql or "").strip() or "-- (empty)")
            lines.append("```")
            lines.append("")
            if errs_json and str(errs_json).strip() not in ("", "[]"):
                try:
                    parsed = json.loads(str(errs_json))
                    if isinstance(parsed, list) and parsed:
                        lines.append("_Validation messages:_")
                        for item in parsed[:8]:
                            lines.append(f"- {item}")
                        lines.append("")
                except json.JSONDecodeError:
                    pass

        lines.append(
            f"*Last registry index: {iso_format(entry.last_indexed_at)}. "
            "Use `query` to search tables when available.*"
        )
        return "\n".join(lines)
    finally:
        release_store(store)


__all__ = [
    "clusters_resource_markdown",
    "databases_resource_markdown",
    "db_overview_resource_markdown",
    "metrics_resource_markdown",
]
