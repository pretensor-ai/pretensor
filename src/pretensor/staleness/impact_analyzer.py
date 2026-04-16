"""Downstream graph impact for a batch of :class:`SchemaChange` rows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pretensor.connectors.snapshot import ChangeTarget, SchemaChange
from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore

__all__ = ["ImpactAnalyzer", "ImpactReport"]


@dataclass
class ImpactReport:
    """Artifacts that may be stale or invalid after schema changes."""

    stale_join_path_ids: list[str] = field(default_factory=list)
    stale_join_path_labels: list[str] = field(default_factory=list)
    affected_cluster_labels: list[str] = field(default_factory=list)
    broken_metrics: list[str] = field(default_factory=list)
    summary: str = ""

    def to_impact_dict(self) -> dict[str, object]:
        """Shape used by MCP ``detect_changes``."""
        return {
            "stale_join_paths": len(self.stale_join_path_ids),
            "affected_clusters": list(dict.fromkeys(self.affected_cluster_labels)),
            "broken_metrics": list(self.broken_metrics),
        }


class ImpactAnalyzer:
    """Find join paths and clusters affected by schema drift (read-only)."""

    def __init__(self, store: KuzuStore) -> None:
        self._store = store

    def analyze(
        self,
        changes: list[SchemaChange],
        *,
        connection_name: str,
        database_key: str,
    ) -> ImpactReport:
        """Return join paths and clusters touching changed tables/columns."""
        report = ImpactReport()
        if not changes:
            report.summary = "No schema changes."
            return report

        affected_tables: set[tuple[str, str]] = set()
        affected_columns: set[tuple[str, str, str]] = set()
        for ch in changes:
            affected_tables.add((ch.schema_name, ch.table_name))
            if ch.target == ChangeTarget.COLUMN and ch.column_name:
                affected_columns.add((ch.schema_name, ch.table_name, ch.column_name))

        table_ids = {table_node_id(connection_name, s, t) for s, t in affected_tables}

        path_rows = self._store.query_all_rows(
            """
            MATCH (p:JoinPath)
            WHERE p.database_key = $db
            RETURN p.node_id, p.semantic_label, p.from_table_id, p.to_table_id,
                   p.steps_json, p.stale
            """,
            {"db": database_key},
        )
        seen_paths: set[str] = set()
        for row in path_rows:
            pid, label, from_id, to_id, steps_json, _stale = row
            pid_s = str(pid)
            if self._path_is_affected(
                pid_s,
                str(from_id or ""),
                str(to_id or ""),
                str(steps_json or ""),
                table_ids,
                affected_tables,
                affected_columns,
            ):
                if pid_s not in seen_paths:
                    seen_paths.add(pid_s)
                    report.stale_join_path_ids.append(pid_s)
                    report.stale_join_path_labels.append(str(label or pid_s))

        cluster_labels: set[str] = set()
        for schema_name, table_name in affected_tables:
            rows = self._store.query_all_rows(
                """
                MATCH (t:SchemaTable)-[:IN_CLUSTER]->(c:Cluster)
                WHERE t.database = $db AND t.schema_name = $sn AND t.table_name = $tn
                RETURN c.label
                """,
                {"db": database_key, "sn": schema_name, "tn": table_name},
            )
            for (lab,) in rows:
                if lab:
                    cluster_labels.add(str(lab))
        report.affected_cluster_labels = sorted(cluster_labels)

        parts: list[str] = []
        if report.stale_join_path_ids:
            parts.append(
                f"{len(report.stale_join_path_ids)} precomputed join path(s) may be "
                "invalid or stale; verify before using in SQL."
            )
        if report.affected_cluster_labels:
            parts.append(
                f"{len(report.affected_cluster_labels)} cluster(s) may need "
                "re-labeling after membership changes."
            )
        if not parts:
            report.summary = (
                "Schema changes are additive or non-breaking for indexed join paths "
                "and clusters."
            )
        else:
            report.summary = " ".join(parts)
        report.summary += (
            " Run `pretensor reindex <dsn>` to refresh the graph; use "
            "`pretensor index` for a full rebuild if needed."
        )
        return report

    def _path_is_affected(
        self,
        _path_id: str,
        from_id: str,
        to_id: str,
        steps_json: str,
        table_ids: set[str],
        affected_tables: set[tuple[str, str]],
        affected_columns: set[tuple[str, str, str]],
    ) -> bool:
        if from_id in table_ids or to_id in table_ids:
            return True
        try:
            data = json.loads(steps_json)
        except json.JSONDecodeError:
            return False
        if not isinstance(data, list):
            return False
        for step in data:
            if not isinstance(step, dict):
                continue
            fs = str(step.get("from_schema", ""))
            ft = str(step.get("from_table", ""))
            ts_ = str(step.get("to_schema", ""))
            tt = str(step.get("to_table", ""))
            fc = str(step.get("from_column", ""))
            tc = str(step.get("to_column", ""))
            if (fs, ft) in affected_tables or (ts_, tt) in affected_tables:
                return True
            for sn, tn, cn in affected_columns:
                if (fs, ft) == (sn, tn) and (fc == cn or tc == cn):
                    return True
                if (ts_, tt) == (sn, tn) and (tc == cn or fc == cn):
                    return True
        return False
