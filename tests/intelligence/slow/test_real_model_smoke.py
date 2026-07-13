"""Opt-in real-model smoke test for the index-time embedding pass.

Loads the pinned ``Snowflake/snowflake-arctic-embed-xs`` ONNX from the Hugging
Face Hub and runs ``EmbeddingIndexStep`` end-to-end on a tiny inline fixture.
Mirrors the pattern from the determinism contract's slow smoke for the bare embedding client.

Mark: ``@pytest.mark.slow``.  Skipped by default.  Run with ``pytest -m slow``;
requires the ``[embeddings]`` extra installed and outbound network access to
download the pinned model revision.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Re-use the canonical extras-installed probe from the top-level conftest
# instead of duplicating an identical helper here.
from tests.conftest import embeddings_extra_installed

from pretensor.config import EmbeddingsConfig, PretensorConfig
from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.intelligence.embeddings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    cosine_similarity,
)
from pretensor.intelligence.pipeline import run_intelligence_layer

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not embeddings_extra_installed(),
        reason="requires the [embeddings] extra",
    ),
]


def _table(name: str, columns: list[str]) -> Table:
    return Table(
        name=name,
        schema_name="public",
        columns=[Column(name=c, data_type="text") for c in columns],
        foreign_keys=[],
        comment="",
    )


def test_real_model_populates_vectors_and_self_similarity_is_one(
    tmp_path: Path,
) -> None:
    """End-to-end: real ONNX → every table has a 384-dim vector; cosine(a, a) ≈ 1.0."""
    snap = SchemaSnapshot(
        connection_name="smoke",
        database="smoke",
        schemas=["public"],
        tables=[
            _table("customers", ["id", "first_name", "last_name", "email"]),
            _table("orders", ["id", "customer_id", "total"]),
            _table("products", ["sku", "name", "price"]),
        ],
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    store = KuzuStore(tmp_path / "smoke.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        cfg = PretensorConfig(embeddings=EmbeddingsConfig(index_tables=True))
        asyncio.run(run_intelligence_layer(store, "smoke", config=cfg))

        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {database: $db}) "
            "RETURN t.node_id, t.embedding ORDER BY t.node_id",
            {"db": "smoke"},
        )
        assert rows, "expected smoke tables to be present"
        for nid, vec in rows:
            assert vec is not None, f"expected embedding for {nid}, got None"
            assert len(vec) == EMBEDDING_DIM, (
                f"expected {EMBEDDING_DIM}-dim vector for {nid}, got {len(vec)}"
            )
            sim = cosine_similarity(list(vec), list(vec))
            assert abs(sim - 1.0) < 1e-5, (
                f"expected cosine self-similarity ≈ 1.0 for {nid}, got {sim}"
            )
    finally:
        store.close()


def test_real_model_pinned_revision_is_advertised() -> None:
    """Belt-and-braces: the module advertises the pinned revision documented."""
    assert EMBEDDING_MODEL_ID == "Snowflake/snowflake-arctic-embed-xs"
    assert EMBEDDING_MODEL_REVISION  # pinned, non-empty
    assert len(EMBEDDING_MODEL_REVISION) >= 7  # at least short SHA length
