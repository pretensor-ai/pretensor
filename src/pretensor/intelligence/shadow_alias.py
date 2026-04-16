"""Shadow-alias detection for single-source views.

A *shadow alias* is a view (``table_type = 'view'``) whose sole data source
is one base table — signaled by having exactly one incoming ``LINEAGE`` edge
(or at most ``lineage_in_max_for_alias`` edges, configurable via
:class:`~pretensor.config.GraphConfig`).

These views duplicate information already present on the base table and cause
noise in clustering, INFERRED_JOIN generation, impact analysis, and traversal.
The helpers here let each consumer detect and suppress them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pretensor.config import GraphConfig
    from pretensor.core.store import KuzuStore

__all__ = [
    "get_shadow_alias_node_ids",
    "is_shadow_alias",
]

logger = logging.getLogger(__name__)


def get_shadow_alias_node_ids(
    store: KuzuStore,
    database_key: str,
    *,
    config: GraphConfig | None = None,
) -> frozenset[str]:
    """Return ``node_id`` values for all shadow aliases in *database_key*.

    A table is a shadow alias when:

    1. ``table_type = 'view'``
    2. ``count(incoming LINEAGE edges) <= lineage_in_max_for_alias`` (default 1)
    3. It has at least one incoming LINEAGE edge (pure orphan views are not aliases)

    The result is a :class:`frozenset` suitable for fast ``in`` checks by
    downstream consumers.
    """
    if config is not None and not config.collapse_shadow_aliases:
        return frozenset()

    max_lineage_in = 1 if config is None else config.lineage_in_max_for_alias

    rows = store.query_all_rows(
        """
        MATCH (src:SchemaTable)-[:LINEAGE]->(v:SchemaTable {database: $db})
        WHERE v.table_type = 'view'
        WITH v.node_id AS vid, count(DISTINCT src.node_id) AS lin
        WHERE lin >= 1 AND lin <= $max_lin
        RETURN vid
        """,
        {"db": database_key, "max_lin": max_lineage_in},
    )
    ids = frozenset(str(r[0]) for r in rows)
    if ids:
        logger.debug(
            "Detected %d shadow alias(es) in %s (lineage_in_max=%d)",
            len(ids),
            database_key,
            max_lineage_in,
        )
    return ids


def is_shadow_alias(
    store: KuzuStore,
    node_id: str,
    *,
    config: GraphConfig | None = None,
) -> bool:
    """Check whether a single table node is a shadow alias.

    Prefer :func:`get_shadow_alias_node_ids` when checking many tables at once
    (one query vs. N).
    """
    if config is not None and not config.collapse_shadow_aliases:
        return False

    max_lineage_in = 1 if config is None else config.lineage_in_max_for_alias

    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {node_id: $nid})
        WHERE t.table_type = 'view'
        WITH t
        MATCH (src:SchemaTable)-[:LINEAGE]->(t)
        WITH count(DISTINCT src.node_id) AS lin
        WHERE lin >= 1 AND lin <= $max_lin
        RETURN lin
        """,
        {"nid": node_id, "max_lin": max_lineage_in},
    )
    return len(rows) > 0
