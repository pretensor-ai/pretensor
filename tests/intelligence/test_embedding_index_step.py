"""Tests for the index-time embedding PipelineStep."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from pretensor.config import EmbeddingsConfig, PretensorConfig
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.intelligence.embeddings import (
    EMBEDDING_DIM,
    LocalEmbeddingClient,
    embeddings_disabled_via_env,
)
from pretensor.intelligence.pipeline import run_intelligence_layer


def _all_table_embeddings(
    store: KuzuStore, database_key: str
) -> list[tuple[str, list[float] | None]]:
    rows = store.query_all_rows(
        "MATCH (t:SchemaTable {database: $db}) "
        "RETURN t.node_id, t.embedding "
        "ORDER BY t.node_id",
        {"db": database_key},
    )
    return [(str(r[0]), r[1]) for r in rows]


def _build_pagila(tmp_path: Path, load_schema: Any) -> KuzuStore:
    snap = load_schema("pagila")
    store = KuzuStore(tmp_path / "pagila.kuzu")
    GraphBuilder().build(snap, store, run_relationship_discovery=False)
    return store


def test_null_path_parity_when_disabled(tmp_path: Path, load_schema: Any) -> None:
    """Default config keeps ``index_tables=False`` → no embeddings written."""
    store = _build_pagila(tmp_path, load_schema)
    try:
        asyncio.run(run_intelligence_layer(store, "pagila", config=PretensorConfig()))
        embeddings = _all_table_embeddings(store, "pagila")
        assert embeddings, "expected pagila tables to be present after indexing"
        for nid, emb in embeddings:
            assert emb is None, (
                f"expected embedding=None for {nid} on null path, got a vector"
            )

        # Sanity: the rest of the intelligence layer still ran.
        cluster_rows = store.query_all_rows(
            "MATCH (c:Cluster) WHERE c.database_key = $db RETURN count(c)",
            {"db": "pagila"},
        )
        assert int(cluster_rows[0][0]) > 0, "clusters should still be produced"
    finally:
        store.close()


def test_on_path_populates_embeddings(
    tmp_path: Path, load_schema: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the flag on and a stub embedder, every table gets a 384-dim vector."""

    def _fake_embed(self: LocalEmbeddingClient, texts: list[str]) -> list[list[float]]:
        return [[0.01] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _fake_embed)

    store = _build_pagila(tmp_path, load_schema)
    try:
        cfg = PretensorConfig(embeddings=EmbeddingsConfig(index_tables=True))
        asyncio.run(run_intelligence_layer(store, "pagila", config=cfg))
        embeddings = _all_table_embeddings(store, "pagila")
        assert embeddings, "expected pagila tables to be present after indexing"
        for nid, emb in embeddings:
            assert emb is not None, f"expected embedding for {nid}, got None"
            assert len(emb) == EMBEDDING_DIM, (
                f"expected {EMBEDDING_DIM}-dim vector for {nid}, got {len(emb)}"
            )
    finally:
        store.close()


def test_batched_embed_failure_leaves_every_table_none(
    tmp_path: Path,
    load_schema: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing batched ``embed`` call logs once and leaves every embedding None.

    The step issues a single batched ``embed()`` call across all tables;
    if it raises, no per-row recovery is attempted (ONNX failures are
    almost always batch-fatal: model crash, OOM, tokenizer error). The
    contract is "all-or-nothing per batch" — verified here.
    """

    def _failing_batch(
        self: LocalEmbeddingClient, texts: list[str]
    ) -> list[list[float]]:
        raise RuntimeError("simulated ONNX failure")

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _failing_batch)

    store = _build_pagila(tmp_path, load_schema)
    try:
        cfg = PretensorConfig(embeddings=EmbeddingsConfig(index_tables=True))
        with caplog.at_level(
            logging.WARNING, logger="pretensor.intelligence.steps_embedding"
        ):
            asyncio.run(run_intelligence_layer(store, "pagila", config=cfg))

        embeddings = _all_table_embeddings(store, "pagila")
        for nid, emb in embeddings:
            assert emb is None, (
                f"expected {nid} to keep embedding=None on batch failure, got a vector"
            )

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == "pretensor.intelligence.steps_embedding"
        ]
        assert len(warnings) == 1, f"expected one WARNING, got {len(warnings)}"
        assert "simulated ONNX failure" in warnings[0].getMessage()
    finally:
        store.close()


def test_embed_invoked_once_per_pipeline_run(
    tmp_path: Path,
    load_schema: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EmbeddingIndexStep`` issues exactly one batched ``embed()`` call.

    Pins the perf optimization that the step does not loop ``embed([text])``
    per table — ONNX inference is dominated by per-call overhead, so any
    regression to per-row calls would be catastrophic on real schemas.
    """
    calls: list[int] = []

    def _counting_embed(
        self: LocalEmbeddingClient, texts: list[str]
    ) -> list[list[float]]:
        calls.append(len(texts))
        return [[0.01] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _counting_embed)

    store = _build_pagila(tmp_path, load_schema)
    try:
        cfg = PretensorConfig(embeddings=EmbeddingsConfig(index_tables=True))
        asyncio.run(run_intelligence_layer(store, "pagila", config=cfg))

        assert len(calls) == 1, (
            f"expected exactly one batched embed() call across all tables, "
            f"got {len(calls)}: batch sizes were {calls}"
        )
        # The pagila fixture has multiple tables — confirm the batch
        # covered all of them rather than degenerating to a 1-row call.
        assert calls[0] >= 5, (
            f"expected the batch to cover all pagila tables, got batch size {calls[0]}"
        )
    finally:
        store.close()


def test_extra_not_installed_fallback(
    tmp_path: Path,
    load_schema: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing ``[embeddings]`` extra → log WARNING once, skip step, no crash."""

    def _missing_deps(
        self: LocalEmbeddingClient, texts: list[str]
    ) -> list[list[float]]:
        raise ImportError(
            "Install embedding dependencies with: pip install 'pretensor[embeddings]'"
        )

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _missing_deps)

    store = _build_pagila(tmp_path, load_schema)
    try:
        cfg = PretensorConfig(embeddings=EmbeddingsConfig(index_tables=True))
        with caplog.at_level(
            logging.WARNING, logger="pretensor.intelligence.steps_embedding"
        ):
            # Must not raise.
            asyncio.run(run_intelligence_layer(store, "pagila", config=cfg))

        embeddings = _all_table_embeddings(store, "pagila")
        assert embeddings
        for nid, emb in embeddings:
            assert emb is None, (
                f"expected embedding=None for {nid} on missing-extra path"
            )

        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == "pretensor.intelligence.steps_embedding"
        ]
        assert len(warnings) == 1, f"expected one WARNING, got {len(warnings)}"
        assert "pretensor[embeddings]" in warnings[0].getMessage()
    finally:
        store.close()


# ── PRETENSOR_EMBEDDINGS_DISABLED env-var override ────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("anything-else", False),
    ],
)
def test_embeddings_disabled_via_env_truthiness(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    """Truthiness mirrors the ``_TRUTHY`` set; everything else is False."""
    monkeypatch.setenv("PRETENSOR_EMBEDDINGS_DISABLED", raw)
    assert embeddings_disabled_via_env() is expected


def test_embeddings_disabled_via_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRETENSOR_EMBEDDINGS_DISABLED", raising=False)
    assert embeddings_disabled_via_env() is False


def test_env_kill_switch_forces_null_path(
    tmp_path: Path,
    load_schema: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PRETENSOR_EMBEDDINGS_DISABLED=1`` + ``index_tables=True`` → no vectors written.

    This is the load-bearing CI assertion behind the determinism contract: a Null-client baseline
    captured with ``index_tables=False`` must equal the run with the extra
    installed and the env var set.  Both paths leave every embedding ``None``.
    """

    def _real_embed_should_not_be_called(
        self: LocalEmbeddingClient, texts: list[str]
    ) -> list[list[float]]:
        raise AssertionError(
            "LocalEmbeddingClient.embed must not be called when "
            "PRETENSOR_EMBEDDINGS_DISABLED is set"
        )

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _real_embed_should_not_be_called)
    monkeypatch.setenv("PRETENSOR_EMBEDDINGS_DISABLED", "1")

    store = _build_pagila(tmp_path, load_schema)
    try:
        cfg = PretensorConfig(embeddings=EmbeddingsConfig(index_tables=True))
        # Must not raise — the step short-circuits before touching the client.
        asyncio.run(run_intelligence_layer(store, "pagila", config=cfg))

        embeddings = _all_table_embeddings(store, "pagila")
        assert embeddings, "expected pagila tables to be present"
        for nid, emb in embeddings:
            assert emb is None, (
                f"expected embedding=None when env kill-switch set, got vector for {nid}"
            )
    finally:
        store.close()


def test_builder_precomputes_embeddings_before_discovery(
    tmp_path: Path,
    load_schema: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GraphBuilder.build`` with ``index_tables=True`` writes vectors during
    the build itself (before relationship discovery), and the in-pipeline
    ``embedding_index`` step does not embed a second time."""
    call_count = {"n": 0}

    def _counting_embed(
        self: LocalEmbeddingClient, texts: list[str]
    ) -> list[list[float]]:
        call_count["n"] += 1
        return [[0.1] * EMBEDDING_DIM for _ in texts]

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _counting_embed)

    snap = load_schema("pagila")
    store = KuzuStore(tmp_path / "pagila.kuzu")
    try:
        cfg = PretensorConfig(embeddings=EmbeddingsConfig(index_tables=True))
        GraphBuilder().build(snap, store, config=cfg)

        embeddings = _all_table_embeddings(store, "pagila")
        assert embeddings and all(emb is not None for _, emb in embeddings), (
            "expected every table to carry a vector after a full build"
        )
        assert call_count["n"] == 1, (
            "expected exactly one batched embed() call for the whole build "
            f"(precompute), got {call_count['n']}"
        )
    finally:
        store.close()


def test_pipeline_step_skips_when_precomputed(
    tmp_path: Path,
    load_schema: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_intelligence_layer(embeddings_precomputed=True)`` must not embed."""

    def _must_not_embed(
        self: LocalEmbeddingClient, texts: list[str]
    ) -> list[list[float]]:
        raise AssertionError("embed() must not run when embeddings_precomputed=True")

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _must_not_embed)

    store = _build_pagila(tmp_path, load_schema)
    try:
        cfg = PretensorConfig(embeddings=EmbeddingsConfig(index_tables=True))
        asyncio.run(
            run_intelligence_layer(
                store, "pagila", config=cfg, embeddings_precomputed=True
            )
        )
    finally:
        store.close()
