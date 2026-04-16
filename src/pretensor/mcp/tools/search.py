"""MCP ``query`` tool (BM25 search) payload."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from pretensor.observability import log_timed_operation
from pretensor.search.base import BaseSearchIndex
from pretensor.search.index import KeywordSearchIndex
from pretensor.visibility.filter import VisibilityFilter

from ..payload_types import QueryHit, snippet
from ..service_context import (
    get_effective_search_index_cls,
    get_effective_visibility_filter,
)
from ..service_registry import load_registry

logger = logging.getLogger(__name__)


def _hit_visible(hit: QueryHit, vf: VisibilityFilter | None) -> bool:
    if vf is None or hit.get("node_type") != "SchemaTable":
        return True
    name = str(hit.get("name", "")).strip()
    if "." not in name:
        return vf.is_table_visible(str(hit.get("connection_name", "")), "", name)
    sn, _, tn = name.partition(".")
    return vf.is_table_visible(str(hit.get("connection_name", "")), sn, tn)


def _load_search_index(
    graph_dir: Path,
    *,
    search_index_cls: type[BaseSearchIndex],
) -> BaseSearchIndex:
    """Load/build a search index using class-level hooks when available."""
    reg = load_registry(graph_dir)
    search_index_cls_any = search_index_cls
    default_path = getattr(search_index_cls_any, "default_path", None)
    index_path = (
        Path(cast(Path, default_path(graph_dir)))
        if callable(default_path)
        else KeywordSearchIndex.default_path(graph_dir)
    )
    load_or_build = getattr(search_index_cls_any, "load_or_build", None)
    if callable(load_or_build):
        built = load_or_build(reg, index_path)
        if not isinstance(built, BaseSearchIndex):
            raise TypeError(
                f"{search_index_cls.__name__}.load_or_build must return BaseSearchIndex"
            )
        return built
    index = cast(Any, search_index_cls_any)(index_path)
    if not isinstance(index, BaseSearchIndex):
        raise TypeError(
            f"{search_index_cls.__name__} constructor must return BaseSearchIndex"
        )
    index.index_graph(reg)
    return index


def query_payload(
    graph_dir: Path,
    *,
    q: str,
    db: str | None = None,
    limit: int = 10,
    search_index_cls: type[BaseSearchIndex] | None = None,
    visibility_filter: VisibilityFilter | None = None,
) -> dict[str, Any]:
    """BM25 search over indexed metadata."""
    with log_timed_operation(
        logger,
        event="mcp.query_payload",
        query=q,
        db=db,
        limit=limit,
        graph_dir=str(graph_dir),
    ):
        vf = visibility_filter or get_effective_visibility_filter()
        index_cls = get_effective_search_index_cls(search_index_cls)
        idx = _load_search_index(graph_dir, search_index_cls=index_cls)
        raw = idx.search(q, db=db, limit=limit * 4)
        hits: list[QueryHit] = []
        for row in raw:
            if row.node_type != "SchemaTable":
                continue
            item: QueryHit = {
                "node_type": row.node_type,
                "name": row.name,
                "database_name": row.database_name,
                "connection_name": row.connection_name,
                "description": row.description,
                "snippet": snippet(row.description),
                "score": row.score,
            }
            if not _hit_visible(item, vf):
                continue
            hits.append(item)
            if len(hits) >= limit:
                break
        return {"query": q, "db": db, "results": hits}


__all__ = ["query_payload"]
