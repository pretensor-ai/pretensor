"""Load and persist ``JoinPath`` rows in Kuzu."""

from __future__ import annotations

import json

from pretensor.core.store import KuzuStore
from pretensor.intelligence.join_paths.on_demand import (
    StoredJoinPath,
    parse_steps_json,
    steps_to_json_payload,
)


def load_stored_paths(
    store: KuzuStore,
    database_key: str,
    from_table_id: str,
    to_table_id: str,
) -> list[StoredJoinPath]:
    """Return precomputed paths for an ordered table pair."""
    rows = store.query_all_rows(
        """
        MATCH (p:JoinPath)
        WHERE p.database_key = $db
          AND p.from_table_id = $from_id
          AND p.to_table_id = $to_id
        RETURN p.node_id, p.depth, p.confidence, p.ambiguous, p.steps_json,
               p.semantic_label, p.stale
        ORDER BY p.confidence DESC
        """,
        {"db": database_key, "from_id": from_table_id, "to_id": to_table_id},
    )
    out: list[StoredJoinPath] = []
    for row in rows:
        nid, depth, conf, amb, steps_json, sem, stale = row
        steps = parse_steps_json(str(steps_json))
        stale_b = bool(stale) if stale is not None else False
        out.append(
            StoredJoinPath(
                path_id=str(nid),
                from_table_id=from_table_id,
                to_table_id=to_table_id,
                depth=int(depth) if depth is not None else len(steps),
                confidence=float(conf) if conf is not None else 0.0,
                ambiguous=bool(amb),
                steps=steps,
                semantic_label=str(sem or ""),
                stale=stale_b,
            )
        )
    return out


def persist_path(store: KuzuStore, database_key: str, path: StoredJoinPath) -> None:
    payload = steps_to_json_payload(path.steps)
    store.upsert_join_path(
        node_id=path.path_id,
        database_key=database_key,
        from_table_id=path.from_table_id,
        to_table_id=path.to_table_id,
        depth=path.depth,
        confidence=path.confidence,
        ambiguous=path.ambiguous,
        steps_json=json.dumps(payload),
        semantic_label=path.semantic_label,
    )


__all__ = ["load_stored_paths", "persist_path"]
