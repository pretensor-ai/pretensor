"""Raw Cypher execution wrapper extracted from KuzuStore."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import kuzu

from pretensor.observability import log_timed_operation

# Use the same logger name as store.py so existing log pipelines see identical
# source fields regardless of which layer emits the record.
logger = logging.getLogger("pretensor.core.store")


class QueryRunner:
    """Thin wrapper around a kuzu.Connection that handles all Cypher execution.

    Owns the three execution methods that were previously on KuzuStore:
    ``execute``, ``execute_write``, and ``query_all_rows``.
    """

    def __init__(self, conn: kuzu.Connection, path: Path) -> None:
        self._conn = conn
        # Stored so execute_write can include graph_path in the log record,
        # matching the field emitted by the original KuzuStore.execute_write.
        self._path = path

    def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a write (mutation) Cypher query. Use this for DELETE / SET operations."""
        with log_timed_operation(
            logger,
            event="graph.execute_write",
            level=logging.DEBUG,
            has_params=bool(params),
            query_preview=query.strip().splitlines()[0][:120] if query.strip() else "",
            graph_path=str(self._path),
        ):
            if params:
                self._conn.execute(query, params)
            else:
                self._conn.execute(query)

    def execute(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> kuzu.QueryResult | list[kuzu.QueryResult]:
        """Run a read query (or arbitrary Cypher) with optional parameters."""
        return self._conn.execute(cypher, parameters)

    def query_all_rows(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[tuple[Any, ...]]:
        """Execute Cypher and materialize all result rows as tuples.

        Keeps callers (MCP, search index) off raw ``kuzu.QueryResult`` handles
        while still routing reads through :class:`QueryRunner`.
        """
        result = self.execute(cypher, parameters)
        if isinstance(result, list):
            raise TypeError("Expected a single QueryResult from query_all_rows")
        rows: list[tuple[Any, ...]] = []
        while result.has_next():
            nxt = result.get_next()
            rows.append(tuple(nxt) if not isinstance(nxt, tuple) else nxt)
        return rows
