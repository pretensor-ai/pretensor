"""MCP ``semantic_search`` tool — cosine ranking over ``SchemaTable.embedding``.

Read-only. Additive signal. Returns a structured ``fallback_bm25`` envelope
when the ``[embeddings]`` extra is absent or no candidate table carries a
vector — per the determinism contract, the tool must never raise
from its happy path and must never create graph edges.

At index time, each table is embedded as ``"{schema}.{table}: cols..."``.
At query time, the natural-language ``query`` string is embedded **raw**
(asymmetric retrieval); callers who want lexical search should use the
BM25 ``query`` tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pretensor.core.registry import RegistryEntry
from pretensor.intelligence.embeddings import (
    EmbeddingClient,
    embeddings_disabled_via_env,
    get_default_embedding_client,
)
from pretensor.mcp.tool_registry import McpTool
from pretensor.observability import log_timed_operation
from pretensor.visibility.filter import VisibilityFilter

from ..payload_types import SemanticHit, snippet
from ..service_context import get_effective_visibility_filter
from ..service_registry import (
    load_registry,
    resolve_registry_entry,
)
from ._rank import CosineHit, any_entry_has_vectors, cosine_rank_tables

logger = logging.getLogger(__name__)

_K_MIN = 1
_K_MAX = 50
_FALLBACK_HINT = "Install `pretensor[embeddings]` and reindex, or use the `query` tool."


def _fallback(
    *,
    query: str,
    database: str | None,
    cluster: str | None,
    hint: str | None = None,
) -> dict[str, Any]:
    """Return the documented ``fallback_bm25`` envelope.

    Extended with ``query`` / ``database`` / ``cluster`` echo so callers can
    log and correlate without branching on envelope shape.  Reserved for
    "embeddings unavailable" (extra absent, disabled, or no vectors) —
    caller mistakes such as an unresolved ``database`` get a plain error
    envelope instead, matching sibling tools.
    """
    return {
        "mode": "fallback_bm25",
        "hint": hint or _FALLBACK_HINT,
        "results": [],
        "query": query,
        "database": database,
        "cluster": cluster,
    }


def _select_entries(
    graph_dir: Path, database: str | None
) -> tuple[list[RegistryEntry], str | None]:
    """Resolve candidate registry entries; return ``(entries, error)``."""
    reg = load_registry(graph_dir)
    entries = reg.list_entries()
    if not entries:
        return [], None  # empty registry → normal fallback, not an error
    if database is None:
        return entries, None
    resolved = resolve_registry_entry(reg, database)
    if resolved is None:
        return [], f"Unknown database: {database!r}"
    return [resolved], None


def _hit_from_cosine(hit: CosineHit) -> SemanticHit:
    result: SemanticHit = {
        "node_type": "SchemaTable",
        "name": (
            f"{hit.schema_name}.{hit.table_name}" if hit.schema_name else hit.table_name
        ),
        "database_name": hit.database,
        "connection_name": hit.connection_name,
        "description": hit.description,
        "snippet": snippet(hit.description),
        "score": hit.score,
        "cluster_id": hit.cluster_id,
    }
    return result


def semantic_search_payload(
    graph_dir: Path,
    *,
    query: str,
    k: int = 10,
    database: str | None = None,
    cluster: str | None = None,
    visibility_filter: VisibilityFilter | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> dict[str, Any]:
    """Top-K ``SchemaTable`` ids by cosine similarity against ``SchemaTable.embedding``.

    Returns the happy-path envelope ``{mode: "semantic", query, database,
    cluster, results: [...]}`` when embeddings are available, otherwise the
    documented ``fallback_bm25`` envelope.  The response never raises; any
    ``ImportError`` from the embedding client (missing ``[embeddings]``
    extra) or unexpected runtime failure funnels to the fallback envelope
    with a single ``WARNING`` logged.
    """
    k_clamped = max(_K_MIN, min(_K_MAX, int(k)))
    with log_timed_operation(
        logger,
        event="mcp.semantic_search_payload",
        query=query,
        database=database,
        cluster=cluster,
        k=k_clamped,
        graph_dir=str(graph_dir),
    ):
        # Kill switch wins over everything, including an injected client,
        # so PRETENSOR_EMBEDDINGS_DISABLED=1 yields the same fallback
        # envelope as a missing extra.
        if embeddings_disabled_via_env():
            return _fallback(
                query=query,
                database=database,
                cluster=cluster,
                hint=(
                    "Embeddings are disabled via PRETENSOR_EMBEDDINGS_DISABLED; "
                    "unset it to enable semantic ranking, or use the `query` tool."
                ),
            )

        vf = visibility_filter or get_effective_visibility_filter()

        entries, error = _select_entries(graph_dir, database)
        if error is not None:
            # Invalid required argument: surface a plain error envelope so
            # callers can correct the database name, matching the contract
            # of sibling tools (`context`, `traverse`). The fallback_bm25
            # envelope is reserved for "embeddings unavailable", not for
            # caller mistakes.
            return {
                "error": error,
                "query": query,
                "database": database,
                "cluster": cluster,
                "hint": (
                    "Resolve the database name with the `list_databases` "
                    "tool, then retry."
                ),
            }
        if not entries:
            return _fallback(query=query, database=database, cluster=cluster)

        # Probe for stored vectors BEFORE embedding the query: with none
        # present the response is the fallback envelope regardless, and on
        # a cold process the first embed() call would trigger the ONNX
        # model download + session init for nothing.
        if not any_entry_has_vectors(entries):
            return _fallback(query=query, database=database, cluster=cluster)

        # Embed the natural-language query as-is. The indexed documents use
        # ``"{schema}.{table}: cols..."``; asymmetric CLS-pooled retrieval.
        # Construction is I/O-free (the ONNX session loads lazily inside
        # ``embed()``); only the embed() call can raise, and it's caught
        # below to route to the never-raises fallback envelope.
        client: EmbeddingClient = embedding_client or get_default_embedding_client()
        try:
            vectors = client.embed([query])
        except ImportError as exc:
            logger.warning("semantic_search: [embeddings] extra not installed: %s", exc)
            return _fallback(query=query, database=database, cluster=cluster)
        except Exception as exc:  # noqa: BLE001 — tool must never raise
            logger.warning(
                "semantic_search: embedding client failed (%s); returning fallback",
                exc,
            )
            return _fallback(query=query, database=database, cluster=cluster)

        if not vectors:
            # NullEmbeddingClient path.
            return _fallback(query=query, database=database, cluster=cluster)

        qvec = vectors[0]

        cosine_hits, any_candidate_had_vector = cosine_rank_tables(
            entries,
            qvec,
            database=database,
            cluster=cluster,
            visibility_filter=vf,
            k=k_clamped,
        )

        if not any_candidate_had_vector:
            return _fallback(query=query, database=database, cluster=cluster)

        results = [_hit_from_cosine(h) for h in cosine_hits]

        return {
            "mode": "semantic",
            "query": query,
            "database": database,
            "cluster": cluster,
            "results": results,
        }


__all__ = ["create_tool", "semantic_search_payload"]


def create_tool(graph_dir: Path) -> McpTool:
    from ._timed import timed_tool

    async def _handle(args: dict) -> dict:
        q = str(args.get("query", "")).strip()
        if not q:
            return {"error": "Missing or empty `query`"}
        k = int(args.get("k", 10))
        database = args.get("database")
        database_s = str(database) if database is not None else None
        cluster = args.get("cluster")
        cluster_s = str(cluster) if cluster is not None else None
        with timed_tool(
            "semantic_search", graph_dir, database=database_s, cluster=cluster_s, k=k
        ):
            return semantic_search_payload(
                graph_dir,
                query=q,
                k=k,
                database=database_s,
                cluster=cluster_s,
            )

    return McpTool(
        name="semantic_search",
        description=(
            "Natural-language semantic search over SchemaTable nodes by cosine similarity "
            "against the persisted embedding column. Requires the `[embeddings]` extra — "
            "falls back to a structured hint pointing at the `query` (BM25) tool when the "
            "extra is absent or no tables carry vectors."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query to embed and match against table vectors",
                },
                "k": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of hits to return (clamped to 1..50)",
                },
                "database": {
                    "type": ["string", "null"],
                    "description": "Restrict scan to one connection_name or logical database",
                },
                "cluster": {
                    "type": ["string", "null"],
                    "description": "Restrict scan to one Cluster id (node_id)",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=_handle,
    )
