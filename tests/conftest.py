"""Pytest fixtures for pretensor."""

from __future__ import annotations

import asyncio
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

# Editable install (`pip install -e .`) adds `pretensor`; for bare pytest, put `src` on path.
_ROOT = Path(__file__).resolve().parents[1]
_src = _ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from pretensor.connectors.models import SchemaSnapshot, Table  # noqa: E402
from pretensor.core.store import KuzuStore  # noqa: E402

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def embeddings_extra_installed() -> bool:
    """Return True iff the [embeddings] extra's runtime deps import cleanly.

    Public so test modules can reuse the canonical probe instead of
    duplicating the import-list. (conftest.py is an unusual home for a
    public symbol, but every test in this tree already discovers
    conftest, and a separate helpers module would just be a thin
    re-export with no other contents.)
    """
    for mod in ("numpy", "onnxruntime", "huggingface_hub", "transformers"):
        try:
            importlib.import_module(mod)
        except ImportError:
            return False
    return True


@pytest.fixture(autouse=True)
def _reset_default_embedding_client():
    """Reset the process-wide cached embedding client between tests.

    Several tests monkeypatch ``LocalEmbeddingClient.embed`` to produce
    deterministic vectors. Without this fixture, a cached client from
    one test would persist into the next, where the new test's
    monkeypatch would still hit the same cached instance — fine in
    practice (monkeypatches are class-level), but resetting between
    tests removes the surprise and matches how
    ``compute_role_centroids``'s per-client cache also gets reset.
    """
    from pretensor.intelligence.embeddings import (
        _reset_default_embedding_client_for_tests,
    )

    _reset_default_embedding_client_for_tests()
    yield
    _reset_default_embedding_client_for_tests()


# ── Snapshot helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def make_snapshot():
    """Return a factory that builds a :class:`SchemaSnapshot` from a table list."""

    def _factory(
        tables: list[Table],
        *,
        connection_name: str = "demo",
        database: str = "db",
        schemas: list[str] | None = None,
    ) -> SchemaSnapshot:
        return SchemaSnapshot(
            connection_name=connection_name,
            database=database,
            schemas=schemas or ["public"],
            tables=tables,
            introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    return _factory


@pytest.fixture
def load_schema():
    """Load a named YAML schema fixture from ``tests/fixtures/schemas/``."""
    schemas_dir = _FIXTURES_DIR / "schemas"

    def _load(name: str) -> SchemaSnapshot:
        return SchemaSnapshot.from_yaml((schemas_dir / f"{name}.yaml").read_text())

    return _load


# ── KuzuStore fixture ─────────────────────────────────────────────────────────


@pytest.fixture
def graph_store(tmp_path: Path):
    """Yield an auto-closing :class:`KuzuStore` with schema ensured."""
    store = KuzuStore(tmp_path / "test.kuzu")
    store.ensure_schema()
    try:
        yield store
    finally:
        store.close()


# ── Embedding-aware fixtures — shared harness for the embedding-extra paths ──


def _build_pagila_indexed(
    tmp_path: Path,
    load_schema_fn: Callable[[str], SchemaSnapshot],
    *,
    name: str,
    config: Any,
) -> KuzuStore:
    """Build a fully-indexed pagila store and run the intelligence layer.

    Internal helper for ``embedded_store`` / ``null_embedded_store``.  Imports
    are lazy so the conftest stays cheap when these fixtures are not requested.
    """
    from pretensor.core.builder import GraphBuilder
    from pretensor.intelligence.pipeline import run_intelligence_layer

    snap = load_schema_fn("pagila")
    store = KuzuStore(tmp_path / f"{name}.kuzu")
    GraphBuilder().build(snap, store, run_relationship_discovery=True)
    asyncio.run(run_intelligence_layer(store, "pagila", config=config))
    return store


@pytest.fixture
def null_embedded_store(tmp_path: Path, load_schema):
    """Pagila store with the embeddings toggle off; every ``SchemaTable.embedding`` is ``None``.

    Use as the ``before`` side of any null-path parity assertion: the snapshot
    of intelligence-layer outputs from this store must stay byte-identical
    across every embedding-related change.
    """
    from pretensor.config import EmbeddingsConfig, PretensorConfig

    cfg = PretensorConfig(embeddings=EmbeddingsConfig())  # explicit defaults
    store = _build_pagila_indexed(tmp_path, load_schema, name="null", config=cfg)
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def embedded_store(tmp_path: Path, load_schema, monkeypatch: pytest.MonkeyPatch):
    """Pagila store with embeddings populated by a deterministic stub embedder.

    Skips when the ``[embeddings]`` extra is not installed.  The stub avoids a
    Hub download and gives reproducible per-text vectors so on-path tests stay
    deterministic without requiring the real ONNX model (the real model is
    exercised by ``tests/intelligence/slow/test_real_model_smoke.py``).
    """
    if not embeddings_extra_installed():
        pytest.skip("requires the [embeddings] extra")

    from pretensor.config import EmbeddingsConfig, PretensorConfig
    from pretensor.intelligence.embeddings import EMBEDDING_DIM, LocalEmbeddingClient

    def _det_embed(self: LocalEmbeddingClient, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            seed = (abs(hash(t)) % 1_000) / 1_000.0
            # Deterministic non-zero unit-ish vector keyed on text hash.
            out.append([seed + (i / EMBEDDING_DIM) for i in range(EMBEDDING_DIM)])
        return out

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _det_embed)

    cfg = PretensorConfig(embeddings=EmbeddingsConfig(index_tables=True))
    store = _build_pagila_indexed(tmp_path, load_schema, name="embedded", config=cfg)
    try:
        yield store
    finally:
        store.close()
