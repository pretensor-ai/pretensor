"""Index-time embedding computation for SchemaTable nodes.

Opt-in via ``PretensorConfig.embeddings.index_tables``.  When the toggle is
off (default) nothing runs — zero reads, zero writes — so the null path
stays byte-identical to the pre-embedding pipeline output.

:func:`compute_table_embeddings` is the single computation entry point.
Index and reindex call it right after schema rows are written and *before*
relationship discovery, so the embedding relationship scorer, the
clustering blend, and the classify step's role vote all consume vectors
from the same run — no one-cycle staging lag.  :class:`EmbeddingIndexStep`
remains in the pipeline for callers that invoke
``run_intelligence_layer`` directly; it delegates to the same function and
skips when the context marks vectors as already computed this run.

Every entry point honors the ``PRETENSOR_EMBEDDINGS_DISABLED`` environment
variable: when set to a truthy value (``1``, ``true``, ``yes`` —
case-insensitive), the embedding pass is forced to the null path even when
the ``[embeddings]`` extra is installed and ``index_tables`` is True.  CI
exercises this to assert that "extra installed but user opted out"
produces output byte-identical to "extra absent" — the determinism
contract behind the release gate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pretensor.intelligence.embeddings import (
    embeddings_disabled_via_env,
    format_entity_text,
    get_default_embedding_client,
)

if TYPE_CHECKING:
    from pretensor.config import EmbeddingsConfig
    from pretensor.core.store import KuzuStore
    from pretensor.intelligence.steps import PipelineContext

# ``embeddings_disabled_via_env`` is canonically defined in
# ``embeddings.py``; it is imported above for this module's own use but is
# deliberately NOT re-exported here, so there is one authoritative import
# path for every caller.
__all__ = [
    "EmbeddingIndexStep",
    "compute_table_embeddings",
]

logger = logging.getLogger(__name__)


# Canonical context keys live in ``steps.py`` (also home to PipelineStep
# / PipelineRunner / PipelineContext). Importing them keeps writer and
# reader on one source of truth — drift between sibling string literals
# would silently turn this step into a no-op.
from pretensor.intelligence.steps import (  # noqa: E402  (post-imports to keep the section clear)
    _CTX_DATABASE_KEY,
    _CTX_EMBEDDINGS_CONFIG,
    _CTX_EMBEDDINGS_PRECOMPUTED,
    _CTX_STORE,
)


def compute_table_embeddings(store: KuzuStore, database_key: str) -> None:
    """Compute one embedding per ``SchemaTable`` and persist it to Kuzu.

    Callers gate on ``EmbeddingsConfig.index_tables`` and
    :func:`embeddings_disabled_via_env` — this function assumes the user
    opted in.  When the ``[embeddings]`` extra is absent it logs once at
    ``WARNING`` and returns without raising.  Per-table write failures are
    tolerated: the offending table is left with ``embedding=None`` and the
    rest of the pass continues.
    """
    rows = store.query_all_rows(
        """
        MATCH (t:SchemaTable {database: $db})
        OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:SchemaColumn)
        RETURN t.node_id, t.schema_name, t.table_name, collect(c.column_name)
        """,
        {"db": database_key},
    )
    if not rows:
        return

    client = get_default_embedding_client()

    # Build the per-table input texts and a parallel ``node_id`` list so
    # we can issue ONE batched ``embed()`` call instead of N. ONNX
    # inference is dominated by per-call overhead (model warm-up,
    # tensor allocation), so a single 50-row batch beats 50 single-row
    # calls by a wide margin on every realistic schema size.
    nids: list[str] = []
    texts: list[str] = []
    for row in rows:
        nids.append(str(row[0]))
        qualified_name = f"{row[1]}.{row[2]}"
        column_names = [str(c) for c in (row[3] or []) if c is not None]
        texts.append(format_entity_text(qualified_name, column_names))

    try:
        vectors = client.embed(texts)
    except ImportError as exc:
        logger.warning(
            "embedding_index: skipping, [embeddings] extra not installed: %s",
            exc,
        )
        return
    except Exception as exc:  # noqa: BLE001 — batch failure is tolerated
        logger.warning(
            "embedding_index: batched embed failed (%s); leaving every "
            "table's embedding=None",
            exc,
        )
        return

    if not vectors:
        return
    if len(vectors) != len(texts):
        logger.warning(
            "embedding_index: client returned %d vectors for %d texts; "
            "skipping write to avoid mis-aligned assignment",
            len(vectors),
            len(texts),
        )
        return

    for nid, vec in zip(nids, vectors, strict=True):
        try:
            store.set_table_embedding(nid, vec)
        except ValueError as exc:
            logger.warning("embedding_index: invalid vector for %s: %s", nid, exc)


class EmbeddingIndexStep:
    """Pipeline wrapper around :func:`compute_table_embeddings`.

    Kept for callers that drive ``run_intelligence_layer`` directly (and
    for plugins that anchor on the ``embedding_index`` step name).  The
    index/reindex CLI paths compute embeddings *before* relationship
    discovery and mark the context with ``embeddings_precomputed=True``,
    in which case this step is a no-op.

    Setting ``PRETENSOR_EMBEDDINGS_DISABLED=1`` forces the null path even
    when ``index_tables`` is True; used by CI to enforce null-path parity.
    """

    name = "embedding_index"
    # No dependencies: the computation reads only SchemaTable + HAS_COLUMN
    # rows, which exist before the intelligence pipeline starts.  Running
    # first lets the classify step's role vote consume same-run vectors
    # instead of being one index cycle behind.
    dependencies: list[str] = []

    async def execute(self, ctx: PipelineContext) -> None:
        emb_cfg: EmbeddingsConfig | None = ctx.get(_CTX_EMBEDDINGS_CONFIG)
        if emb_cfg is None or not emb_cfg.index_tables:
            return
        if embeddings_disabled_via_env():
            logger.info("embedding_index: skipping (PRETENSOR_EMBEDDINGS_DISABLED set)")
            return
        if ctx.get(_CTX_EMBEDDINGS_PRECOMPUTED, False):
            logger.debug("embedding_index: vectors already computed this run")
            return

        store: KuzuStore = ctx.get(_CTX_STORE)
        database_key: str = ctx.get(_CTX_DATABASE_KEY)
        compute_table_embeddings(store, database_key)
