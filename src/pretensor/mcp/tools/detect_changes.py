"""MCP ``detect_changes`` tool payload."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pretensor.config import GraphConfig
from pretensor.connectors.inspect import inspect
from pretensor.connectors.snapshot import ChangeTarget, SchemaChange, diff_snapshots
from pretensor.introspection.models.dsn import connection_config_from_registry_dsn
from pretensor.staleness.impact_analyzer import ImpactAnalyzer
from pretensor.staleness.snapshot_store import SnapshotStore
from pretensor.visibility.filter import VisibilityFilter

from ..payload_types import iso_format, stale_threshold_days, staleness_days, utc_now
from ..service_context import (
    get_effective_graph_config,
    get_effective_visibility_filter,
)
from ..service_registry import (
    entry_plaintext_dsn,
    graph_path_for_entry,
    load_registry,
    open_store_for_entry,
    resolve_registry_entry,
)

logger = logging.getLogger(__name__)

# Matches the `details` string produced by `_lineage_diffs` in connectors/snapshot.py:
#   "{lineage_type}: {source_schema}.{source_table} → {target_schema}.{target_table}"
# If that format changes, this regex silently stops filtering lineage targets (passthrough).
_LINEAGE_TARGET_RE = re.compile(r"→\s*([^.]+)\.([^\s]+)\s*$")


def _remediation_hint(dialect: str | None) -> str:
    """Human-readable remediation for a connection failure, keyed by dialect.

    The first-impression OSS user runs ``detect_changes`` on a fresh box; when
    creds aren't in the env, the raw exception (``"SNOWFLAKE_ACCOUNT not set"``)
    is opaque. A structured hint points them at ``pretensor connect --dry-run``
    and the env vars the connector actually reads.
    """
    d = (dialect or "").lower()
    if d == "snowflake":
        return (
            "Snowflake connection failed. Verify credentials with "
            "`pretensor connect --dry-run <dsn>`, or set "
            "SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER / SNOWFLAKE_PASSWORD "
            "in the environment."
        )
    if d in {"postgres", "postgresql"}:
        return (
            "Postgres connection failed. Verify with "
            "`pretensor connect --dry-run <dsn>` and confirm the host is reachable."
        )
    return (
        "Connection failed; verify the DSN with "
        "`pretensor connect --dry-run <dsn>`."
    )


def _connection_unavailable_envelope(
    connection_name: str,
    dialect: str | None,
    exc: Exception,
    *,
    last_indexed: Any,
    checked_at: Any,
    message_prefix: str = "",
) -> dict[str, Any]:
    """Structured envelope for connection failures; clients branch on ``category``."""
    raw = str(exc)
    message = f"{message_prefix}{raw}" if message_prefix else raw
    return {
        "database": connection_name,
        "status": "connection_unavailable",
        "category": "connection_unavailable",
        "message": message,
        "hint": _remediation_hint(dialect),
        "last_indexed": iso_format(last_indexed),
        "checked_at": iso_format(checked_at),
    }


def _change_visible(
    ch: SchemaChange,
    *,
    connection_name: str,
    vf: VisibilityFilter,
) -> bool:
    if ch.target in (ChangeTarget.TABLE, ChangeTarget.COLUMN):
        return vf.is_table_visible(connection_name, ch.schema_name, ch.table_name)
    if ch.target == ChangeTarget.LINEAGE:
        if not vf.is_table_visible(connection_name, ch.schema_name, ch.table_name):
            return False
        det = ch.details or ""
        m = _LINEAGE_TARGET_RE.search(det)
        if m:
            return vf.is_table_visible(connection_name, m.group(1), m.group(2))
        return True
    return True


def schema_change_to_dict(ch: SchemaChange) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": ch.change_type.value,
        "target": ch.target.value,
        "schema": ch.schema_name,
        "table": ch.table_name,
    }
    if ch.column_name:
        out["column"] = ch.column_name
    if ch.details:
        out["details"] = ch.details
    return out


def detect_changes_payload(
    graph_dir: Path,
    *,
    database: str,
    config: GraphConfig | None = None,
    visibility_filter: VisibilityFilter | None = None,
) -> dict[str, Any]:
    """Compare live schema to the last saved snapshot (read-only)."""
    cfg = get_effective_graph_config(config)
    vf = visibility_filter or get_effective_visibility_filter()
    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, database)
    if entry is None:
        return {"error": "Unknown database connection or name."}

    snap_store = SnapshotStore(graph_dir)
    old_snapshot = snap_store.load(entry.connection_name)
    if old_snapshot is None:
        return {
            "database": entry.connection_name,
            "status": "no_snapshot",
            "message": "No saved snapshot; run `pretensor index` first.",
        }

    checked_at = utc_now()
    last_indexed = entry.last_indexed_at
    days = staleness_days(last_indexed, checked_at)
    stale_warn = days > stale_threshold_days(cfg)

    try:
        dsn_plain = entry_plaintext_dsn(entry, graph_dir)
        conn_cfg = connection_config_from_registry_dsn(
            dsn_plain, entry.connection_name, entry.dialect
        )
        new_snapshot = inspect(conn_cfg)
    except OSError as exc:
        return _connection_unavailable_envelope(
            entry.connection_name,
            entry.dialect,
            exc,
            last_indexed=last_indexed,
            checked_at=checked_at,
        )
    except Exception as exc:
        logger.warning("detect_changes: introspection failed: %s", exc)
        return _connection_unavailable_envelope(
            entry.connection_name,
            entry.dialect,
            exc,
            last_indexed=last_indexed,
            checked_at=checked_at,
            message_prefix="Introspection failed: ",
        )

    changes = diff_snapshots(old_snapshot, new_snapshot)
    if vf is not None:
        changes = [
            c
            for c in changes
            if _change_visible(
                c, connection_name=entry.connection_name, vf=vf
            )
        ]
    gp = graph_path_for_entry(entry)
    if not gp.exists():
        return {
            "database": entry.connection_name,
            "error": f"Graph file missing: {gp}",
        }

    store = open_store_for_entry(entry)
    try:
        analyzer = ImpactAnalyzer(store)
        impact = analyzer.analyze(
            changes,
            connection_name=entry.connection_name,
            database_key=str(entry.database),
        )
    finally:
        store.close()

    base: dict[str, Any] = {
        "database": entry.connection_name,
        "last_indexed": iso_format(last_indexed),
        "checked_at": iso_format(checked_at),
        "changes": [schema_change_to_dict(c) for c in changes],
        "graph_impact": impact.to_impact_dict(),
        "affected_join_paths": impact.stale_join_path_labels,
        "affected_metrics": impact.broken_metrics,
        "summary": impact.summary,
        "recommendation": (
            "Run `pretensor reindex <dsn> --database "
            f"{entry.connection_name!r}` to update the graph "
            "(add `--recompute-intelligence` if join paths or clusters drifted)."
        ),
    }
    if stale_warn:
        base["stale_warning"] = True
        base["staleness_days"] = days
    if not changes:
        base["up_to_date"] = True
    return base


__all__ = ["detect_changes_payload", "schema_change_to_dict"]
