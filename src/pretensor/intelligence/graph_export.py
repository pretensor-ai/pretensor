"""Export ``SchemaTable`` nodes and join edges from Kuzu into an igraph graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import igraph as ig

from pretensor.core.store import KuzuStore
from pretensor.intelligence.shadow_alias import get_shadow_alias_node_ids

if TYPE_CHECKING:
    from pretensor.config import GraphConfig

__all__ = ["GraphExporter"]

_FK_WEIGHT = 1.0


class GraphExporter:
    """Build a weighted undirected igraph graph for one logical database."""

    def __init__(self, store: KuzuStore) -> None:
        self._store = store

    def to_igraph(
        self,
        database_key: str,
        *,
        config: GraphConfig | None = None,
    ) -> ig.Graph:
        """Load tables and FK / inferred edges for ``database_key`` into igraph.

        Vertex attributes: ``node_id`` (Kuzu ``SchemaTable.node_id``), ``name``
        (``schema.table``). Edge attribute: ``weight`` (FK = 1.0, inferred =
        confidence). Undirected; parallel A–B edges are merged with max weight.

        Shadow aliases (single-source views) are excluded from the graph when
        ``config.collapse_shadow_aliases`` is enabled (the default).

        Args:
            database_key: Value of ``SchemaTable.database`` (logical DB name).
            config: Graph config; controls shadow-alias collapsing.

        Returns:
            Possibly empty graph (no vertices if no tables match).
        """
        shadow_ids = get_shadow_alias_node_ids(
            self._store, database_key, config=config
        )

        rows = self._store.query_all_rows(
            """
            MATCH (t:SchemaTable)
            WHERE t.database = $db
            RETURN t.node_id, t.schema_name, t.table_name
            ORDER BY t.node_id
            """,
            {"db": database_key},
        )
        if not rows:
            return ig.Graph()

        # Filter out shadow aliases from the vertex set
        filtered = [
            (str(r[0]), f"{r[1]}.{r[2]}")
            for r in rows
            if str(r[0]) not in shadow_ids
        ]
        if not filtered:
            return ig.Graph()
        node_ids = [nid for nid, _ in filtered]
        names = [name for _, name in filtered]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        pair_weights: dict[tuple[int, int], tuple[float, str]] = {}
        self._accumulate_edges(
            id_to_idx,
            pair_weights,
            """
            MATCH (a:SchemaTable)-[r:FK_REFERENCES]->(b:SchemaTable)
            WHERE a.database = $db AND b.database = $db
            RETURN a.node_id, b.node_id, r.source_column, r.target_column
            """,
            {"db": database_key},
            edge_kind="fk",
            weight_for_row=lambda _row: _FK_WEIGHT,
        )
        self._accumulate_edges(
            id_to_idx,
            pair_weights,
            """
            MATCH (a:SchemaTable)-[r:INFERRED_JOIN]->(b:SchemaTable)
            WHERE a.database = $db AND b.database = $db
            RETURN a.node_id, b.node_id, r.confidence
            """,
            {"db": database_key},
            edge_kind="inferred",
            weight_for_row=lambda row: float(row[2]) if row[2] is not None else 0.5,
        )

        g = ig.Graph(n=len(node_ids), directed=False)
        g.vs["node_id"] = node_ids
        g.vs["name"] = names

        edges: list[tuple[int, int]] = []
        weights: list[float] = []
        for (i, j), (w, _kind) in sorted(pair_weights.items()):
            edges.append((i, j))
            weights.append(w)
        if edges:
            g.add_edges(edges)
            g.es["weight"] = weights
        return g

    def _accumulate_edges(
        self,
        id_to_idx: dict[str, int],
        pair_weights: dict[tuple[int, int], tuple[float, str]],
        cypher: str,
        params: dict[str, Any],
        *,
        edge_kind: str,
        weight_for_row: Any,
    ) -> None:
        for row in self._store.query_all_rows(cypher, params):
            a_raw, b_raw = row[0], row[1]
            if a_raw is None or b_raw is None:
                continue
            a, b = str(a_raw), str(b_raw)
            ia = id_to_idx.get(a)
            ib = id_to_idx.get(b)
            if ia is None or ib is None:
                continue
            i, j = (ia, ib) if ia < ib else (ib, ia)
            w = float(weight_for_row(row))
            prev = pair_weights.get((i, j))
            if prev is None or w > prev[0]:
                pair_weights[(i, j)] = (w, edge_kind)
