"""MCP ``query`` tool.

BM25 search over table and entity metadata (SQLite FTS5). When the
``[embeddings]`` extra is installed and at least one indexed table carries a
vector, the results are fused with a cosine top-K pass via Reciprocal Rank
Fusion (RRF) and the envelope gains a ``"rerank": "rrf"`` marker. When no
vectors are present or the embedding client fails at runtime, the envelope is
byte-identical to the pre-fusion response (no ``rerank`` key).

Determinism contract: fusion must never override the BM25 path when
embeddings are absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from pretensor.intelligence.embeddings import (
    EmbeddingClient,
    embeddings_disabled_via_env,
    get_default_embedding_client,
)
from pretensor.mcp.tool_registry import McpTool
from pretensor.observability import log_timed_operation
from pretensor.search.base import BaseSearchIndex
from pretensor.search.index import KeywordSearchIndex
from pretensor.visibility.filter import VisibilityFilter

from ..payload_types import QueryHit, snippet
from ..service_context import (
    get_effective_search_index_cls,
    get_effective_visibility_filter,
)
from ..service_registry import load_registry, resolve_registry_entry
from ._rank import (
    CosineHit,
    any_entry_has_vectors,
    cosine_rank_tables,
    fusion_key,
    rrf_fuse,
)

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


def _split_name(name: str) -> tuple[str, str]:
    """Split ``"schema.table"`` into ``(schema, table)``; tolerate a missing dot."""
    if "." in name:
        schema_name, _, table_name = name.partition(".")
        return schema_name, table_name
    return "", name


def _bm25_key(hit: QueryHit) -> str:
    schema_name, table_name = _split_name(str(hit.get("name", "")))
    return fusion_key(
        connection_name=str(hit.get("connection_name", "")),
        database=str(hit.get("database_name", "")),
        schema_name=schema_name,
        table_name=table_name,
    )


def _hit_from_cosine(hit: CosineHit, *, score: float) -> QueryHit:
    item: QueryHit = {
        "node_type": "SchemaTable",
        "name": (
            f"{hit.schema_name}.{hit.table_name}" if hit.schema_name else hit.table_name
        ),
        "database_name": hit.database,
        "connection_name": hit.connection_name,
        "description": hit.description,
        "snippet": snippet(hit.description),
        "score": score,
    }
    return item


def query_payload(
    graph_dir: Path,
    *,
    q: str,
    db: str | None = None,
    limit: int = 10,
    search_index_cls: type[BaseSearchIndex] | None = None,
    visibility_filter: VisibilityFilter | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> dict[str, Any]:
    """BM25 search over indexed metadata; hybrid RRF rerank when embeddings available."""
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

        # BM25 pool: retain up to limit*4 visible SchemaTable hits so fusion
        # has a reasonable pool to pull from; null path truncates to ``limit``.
        bm25_pool: list[QueryHit] = []
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
            bm25_pool.append(item)

        fused = _maybe_fuse(
            graph_dir,
            q=q,
            db=db,
            limit=limit,
            bm25_pool=bm25_pool,
            visibility_filter=vf,
            embedding_client=embedding_client,
        )
        if fused is not None:
            return fused

        # Null path: BM25-only envelope, byte-identical to pre-fusion behavior.
        return {"query": q, "db": db, "results": bm25_pool[:limit]}


def _maybe_fuse(
    graph_dir: Path,
    *,
    q: str,
    db: str | None,
    limit: int,
    bm25_pool: list[QueryHit],
    visibility_filter: VisibilityFilter | None,
    embedding_client: EmbeddingClient | None,
) -> dict[str, Any] | None:
    """Attempt the hybrid rerank; return ``None`` when the null path applies."""
    # Kill switch wins over everything, including an injected client, so
    # PRETENSOR_EMBEDDINGS_DISABLED=1 forces the BM25-only envelope even
    # when the extra is installed and vectors exist.
    if embeddings_disabled_via_env():
        return None

    reg = load_registry(graph_dir)
    all_entries = reg.list_entries()
    if not all_entries:
        return None
    if db is None:
        entries = all_entries
    else:
        resolved = resolve_registry_entry(reg, db)
        if resolved is None:
            # Unknown db: BM25 already returns empty for this case; fall back
            # to its behavior instead of marking the envelope as reranked.
            return None
        entries = [resolved]

    # Probe for stored vectors BEFORE embedding the query: when no table
    # carries a vector the fused result is discarded anyway, and on a cold
    # process the first embed() call would trigger the ONNX model download
    # + session init for nothing.
    if not any_entry_has_vectors(entries):
        return None

    # ``LocalEmbeddingClient.__init__`` does no I/O — the heavy lifting
    # happens lazily inside ``embed()``. So a fresh-or-cached construction
    # here cannot raise ImportError; the cached default keeps the ONNX
    # session warm across query calls instead of re-initializing per
    # request.
    client = embedding_client or get_default_embedding_client()

    try:
        vectors = client.embed([q])
    except ImportError as exc:
        logger.warning("query: [embeddings] extra not installed: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 — tool must never raise
        logger.warning("query: embedding client failed (%s); BM25 only", exc)
        return None

    if not vectors or not vectors[0]:
        return None
    qvec = vectors[0]

    cosine_hits, any_candidate_had_vector = cosine_rank_tables(
        entries,
        qvec,
        database=db,
        cluster=None,
        visibility_filter=visibility_filter,
        k=max(20, limit * 4),
    )
    if not any_candidate_had_vector:
        # No tables carry vectors → preserve null-path parity.
        return None

    bm25_by_key: dict[str, QueryHit] = {}
    bm25_keys: list[str] = []
    for hit in bm25_pool:
        key = _bm25_key(hit)
        if key in bm25_by_key:
            # Duplicate FQN in the BM25 pool (shouldn't happen, but be safe).
            continue
        bm25_by_key[key] = hit
        bm25_keys.append(key)
    cosine_by_key: dict[str, CosineHit] = {h.key: h for h in cosine_hits}
    cosine_keys = [h.key for h in cosine_hits]

    fused_results: list[QueryHit] = []
    for key, score in rrf_fuse(bm25_keys=bm25_keys, cosine_keys=cosine_keys):
        if key in bm25_by_key:
            # Reuse BM25 hit (keeps its snippet); overwrite score with RRF.
            hit = dict(bm25_by_key[key])
            hit["score"] = score
            fused_results.append(cast(QueryHit, hit))
        elif key in cosine_by_key:
            fused_results.append(_hit_from_cosine(cosine_by_key[key], score=score))
        if len(fused_results) >= limit:
            break

    return {"query": q, "db": db, "results": fused_results, "rerank": "rrf"}


__all__ = ["create_tool", "query_payload"]


def create_tool(graph_dir: Path) -> McpTool:
    from ._timed import timed_tool

    async def _handle(args: dict) -> dict:
        q = str(args.get("q", "")).strip()
        if not q:
            return {"error": "Missing or empty `q`"}
        limit = int(args.get("limit", 10))
        db = args.get("db")
        db_s = str(db) if db is not None else None
        with timed_tool("query", graph_dir, db=db_s, limit=limit):
            return query_payload(graph_dir, q=q, db=db_s, limit=limit)

    return McpTool(
        name="query",
        description="BM25 keyword search over table and entity metadata (FTS5).",
        input_schema={
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "Search query"},
                "db": {
                    "type": ["string", "null"],
                    "description": "Filter by connection_name or logical database name",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["q"],
            "additionalProperties": False,
        },
        handler=_handle,
    )
