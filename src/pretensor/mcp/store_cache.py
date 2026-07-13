"""Per-server-lifetime cache of open KuzuStore handles, keyed by resolved graph path.

The MCP server is single-threaded (stdio + one asyncio loop), so a single reused
KuzuStore (one kuzu.Connection) per graph path is safe and avoids re-opening the
database + re-running ensure_schema() on every tool call. Cross-process staleness
caveat: a separate `pretensor reindex` process writing the same file is not
reflected in an already-open cached read handle until the server restarts — this
matches the previous behavior's freshness boundary for the server's lifetime.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pretensor.core.store import KuzuStore

logger = logging.getLogger(__name__)


class StoreCache:
    def __init__(self) -> None:
        self._stores: dict[Path, KuzuStore] = {}

    def get(self, graph_path: Path) -> KuzuStore:
        key = graph_path.resolve()
        store = self._stores.get(key)
        if store is None:
            store = KuzuStore(key)
            store.ensure_schema()
            self._stores[key] = store
        return store

    def owns(self, store: KuzuStore) -> bool:
        return any(s is store for s in self._stores.values())

    def close(self) -> None:
        for path, store in self._stores.items():
            try:
                store.close()
            except Exception:
                logger.warning("Error closing cached store for %s", path, exc_info=True)
        self._stores.clear()
