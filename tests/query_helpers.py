"""Test helpers for Kuzu query results (Pyright narrows ``execute`` return type)."""

from __future__ import annotations

from typing import Any, cast

import kuzu

from pretensor.core.store import KuzuStore


def single_query_result(
    store: KuzuStore,
    cypher: str,
    parameters: dict[str, Any] | None = None,
) -> kuzu.QueryResult:
    """Return a single ``QueryResult`` from :meth:`KuzuStore.execute`."""
    raw = store.execute(cypher, parameters)
    assert not isinstance(raw, list)
    return cast(kuzu.QueryResult, raw)


def first_cell(result: kuzu.QueryResult) -> Any:
    """First value from ``get_next()`` (runtime rows are list-like; stubs disagree)."""
    row = cast(Any, result.get_next())
    return row[0]
