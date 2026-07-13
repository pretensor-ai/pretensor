"""Tests for ``compile_metric`` ``did_you_mean`` suggestions.

Pinned contracts:

1.  Compile success paths produce ``MetricCompileError``-free output and
    no ``did_you_mean`` field is added to the success envelope (the field
    only ever lives on the error class).
2.  Compile failure on an unresolved name (metric, table, column) carries
    a ``did_you_mean: list[str]`` populated with up to 3 close matches.
3.  Levenshtein fallback applies when the ``[embeddings]`` extra is
    unavailable or its embed call fails — the suggestions are still
    relevant.
4.  Type-aware: an unknown metric name only suggests metric names; an
    unknown column name only suggests column names; cross-kind leakage
    is forbidden.
5.  The ``did_you_mean`` text is also embedded in ``str(exc)`` so plain
    string consumers see it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from pretensor.connectors.models import (
    Column,
    ForeignKey,
    SchemaSnapshot,
    Table,
)
from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.introspection.models.semantic import (
    Attribute,
    AttributeRole,
    Domain,
    Entity,
    Metric,
    MetricType,
)
from pretensor.introspection.models.semantic import (
    SemanticLayer as SemanticLayerModel,
)
from pretensor.semantic.compiler import (
    MetricCompileError,
    MetricSqlCompiler,
    _suggest_did_you_mean,
)

# The autouse cache reset for the shared default embedding client lives
# in ``tests/conftest.py``; no module-local fixture needed here.


def _raise_import_error(self, texts: list[str]) -> list[list[float]]:
    """Helper used to force the Levenshtein fallback in compiler tests.

    Replaces the inline ``(_ for _ in ()).throw(ImportError(...))`` lambda
    pattern with a readable named function — same effect, fewer surprises.
    """
    raise ImportError("no extra (test stub)")


def _stub_embedder_factory(
    score_by_token: dict[str, list[float]],
) -> Callable[[object, list[str]], list[list[float]]]:
    """Return a deterministic stub ``embed`` for the cosine-ranked path.

    Maps each input string to the vector keyed by the first matching
    substring in ``score_by_token`` (case-insensitive). Lets a test pin
    "this candidate scores higher than that one" without loading ONNX.
    """
    default = [0.0, 0.0, 0.0, 1.0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            tl = t.lower()
            chosen = default
            for token, vec in score_by_token.items():
                if token.lower() in tl:
                    chosen = vec
                    break
            out.append(list(chosen))
        return out

    return _embed


def _snapshot() -> SchemaSnapshot:
    t_orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="user_id", data_type="int", is_foreign_key=True),
            Column(name="amount", data_type="numeric"),
            Column(name="ordered_at", data_type="timestamp"),
        ],
        foreign_keys=[
            ForeignKey(
                source_schema="public",
                source_table="orders",
                source_column="user_id",
                target_schema="public",
                target_table="users",
                target_column="id",
            )
        ],
    )
    t_users = Table(
        name="users",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
        foreign_keys=[],
    )
    return SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[t_orders, t_users],
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _layer(metrics: list[Metric]) -> SemanticLayerModel:
    entity = Entity(
        name="orders",
        description="Orders",
        source_table="public.orders",
        attributes=[
            Attribute(
                name="id",
                description="pk",
                role=AttributeRole.IDENTIFIER,
                source_column="id",
            ),
            Attribute(
                name="amount",
                description="measure",
                role=AttributeRole.MEASURE,
                source_column="amount",
            ),
        ],
        metrics=metrics,
    )
    return SemanticLayerModel(
        connection_name="demo",
        domains=[Domain(name="sales", description="sales", entities=[entity])],
    )


def _build_store(tmp_path) -> KuzuStore:
    store = KuzuStore(tmp_path / "g.kuzu")
    GraphBuilder().build(_snapshot(), store, run_relationship_discovery=False)
    return store


# ── Levenshtein fallback (pure unit tests) ───────────────────────────────────


def test_suggest_returns_top_k_levenshtein_when_embeddings_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``LocalEmbeddingClient.embed`` raises ImportError, fall through to difflib."""
    from pretensor.intelligence.embeddings import LocalEmbeddingClient

    def _missing(self: LocalEmbeddingClient, texts: list[str]) -> list[list[float]]:
        raise ImportError("no extra")

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _missing)

    out = _suggest_did_you_mean("custmer", ["customer", "customers", "user", "order"])
    assert out, "Levenshtein should still produce hits when embeddings are unavailable"
    assert all(isinstance(s, str) for s in out)
    assert len(out) <= 3
    # The closest match by edit distance is "customer".
    assert "customer" in out


def test_suggest_returns_empty_for_empty_candidate_set() -> None:
    assert _suggest_did_you_mean("anything", []) == []


def test_suggest_skips_huge_candidate_sets() -> None:
    """When the candidate set is unreasonably large, no suggestions are made.

    The compile-error path is rare; spending O(N) on a 50k-column schema
    just to produce a "did you mean" hint isn't worth the latency.
    """
    huge = [f"col_{i}" for i in range(2000)]
    assert _suggest_did_you_mean("col_42", huge) == []


def test_suggest_dedupes_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate candidate names collapse to one entry in the output."""
    from pretensor.intelligence.embeddings import LocalEmbeddingClient

    monkeypatch.setattr(
        LocalEmbeddingClient, "embed", lambda self, texts: []
    )  # force fallback

    out = _suggest_did_you_mean("customer", ["customer", "customer", "user"])
    assert out.count("customer") <= 1


# ── End-to-end via MetricSqlCompiler ─────────────────────────────────────────


def test_unknown_metric_includes_did_you_mean(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for an unknown metric returns suggestions ranked over metric names only."""
    from pretensor.intelligence.embeddings import LocalEmbeddingClient

    # Force Levenshtein fallback so the test is deterministic without ONNX.
    monkeypatch.setattr(LocalEmbeddingClient, "embed", _raise_import_error)

    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="total_revenue",
                    description="sum",
                    type=MetricType.SUM,
                    field="amount",
                ),
                Metric(
                    name="order_count",
                    description="count",
                    type=MetricType.COUNT,
                    field="id",
                ),
            ]
        )
        compiler = MetricSqlCompiler(store, connection_name="demo", database_key="demo")

        with pytest.raises(MetricCompileError) as exc_info:
            compiler.compile(layer, "total_revnue")  # typo

        exc = exc_info.value
        assert isinstance(exc.did_you_mean, list)
        assert "total_revenue" in exc.did_you_mean
        # Type-awareness: only metric names, not table or column names.
        assert "amount" not in exc.did_you_mean
        assert "public.orders" not in exc.did_you_mean
        # Hint is also in the formatted message for plain str(exc) consumers.
        assert "did you mean" in str(exc).lower()
        assert "total_revenue" in str(exc)
    finally:
        store.close()


def test_unknown_column_includes_did_you_mean(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown column on the resolved entity table returns column-only suggestions."""
    from pretensor.intelligence.embeddings import LocalEmbeddingClient

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _raise_import_error)

    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="total",
                    description="sum",
                    type=MetricType.SUM,
                    field="amout",  # typo for "amount"
                )
            ]
        )
        compiler = MetricSqlCompiler(store, connection_name="demo", database_key="demo")

        with pytest.raises(MetricCompileError) as exc_info:
            compiler.compile(layer, "total")

        exc = exc_info.value
        assert "amount" in exc.did_you_mean
        # Type-awareness: column names only — no table or metric names.
        assert "public.orders" not in exc.did_you_mean
        assert "total" not in exc.did_you_mean
    finally:
        store.close()


def test_compile_success_does_not_decorate_envelope(tmp_path) -> None:
    """Successful compile produces no ``did_you_mean`` on the result.

    ``did_you_mean`` lives only on the error class; ``CompiledMetric``
    must stay byte-identical to the pre-embedding shape.
    """
    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="total",
                    description="sum",
                    type=MetricType.SUM,
                    field="amount",
                )
            ]
        )
        compiler = MetricSqlCompiler(store, connection_name="demo", database_key="demo")
        compiled = compiler.compile(layer, "total")
        assert compiled.metric == "total"
        assert not hasattr(compiled, "did_you_mean")
    finally:
        store.close()


def test_metric_compile_error_default_did_you_mean_is_empty() -> None:
    """Backward compat: callers that raise without a hint see ``[]``."""
    err = MetricCompileError("bare message")
    assert err.did_you_mean == []
    assert "did you mean" not in str(err).lower()


# ── Embedding-ranked path (cosine, no ImportError) ───────────────────────────


def test_suggest_uses_embedding_cosine_when_client_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``LocalEmbeddingClient.embed`` returns vectors, ranking is cosine.

    Cross-checks Levenshtein vs. cosine: ``account`` and ``customer`` are
    both lexically far from ``user``, but the stub embedder maps both
    ``user`` and ``account`` to the same vector, so cosine ranks
    ``account`` first — Levenshtein would rank ``user_archive`` (closer
    edit distance) first instead.
    """
    from pretensor.intelligence.embeddings import LocalEmbeddingClient

    monkeypatch.setattr(
        LocalEmbeddingClient,
        "embed",
        _stub_embedder_factory(
            {
                "user": [1.0, 0.0, 0.0, 0.0],
                "account": [1.0, 0.0, 0.0, 0.0],  # cosine 1.0 with "user"
                "customer": [0.0, 1.0, 0.0, 0.0],  # cosine 0.0 with "user"
                "order": [0.0, 0.0, 1.0, 0.0],
            }
        ),
    )

    out = _suggest_did_you_mean("user", ["account", "customer", "order"])
    assert out, "embedding-ranked path should return suggestions"
    assert out[0] == "account", (
        f"cosine ranking should put 'account' first (vec match), got {out!r}"
    )


def test_suggest_caches_embedding_client_across_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated suggestion calls must reuse one ``LocalEmbeddingClient`` instance.

    Constructing a fresh client per call would re-init the multi-MB ONNX
    session and tokenizer state every compile error.  This test counts
    constructor calls to confirm the cache pins one instance.
    """
    from pretensor.intelligence import embeddings as embeddings_mod

    constructed: list[object] = []
    real_init = embeddings_mod.LocalEmbeddingClient.__init__

    def _counting_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        constructed.append(self)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(embeddings_mod.LocalEmbeddingClient, "__init__", _counting_init)
    monkeypatch.setattr(
        embeddings_mod.LocalEmbeddingClient,
        "embed",
        _stub_embedder_factory({"x": [1.0, 0.0]}),
    )

    _suggest_did_you_mean("user", ["account", "order"])
    _suggest_did_you_mean("user", ["account", "order"])
    _suggest_did_you_mean("user", ["account", "order"])

    assert len(constructed) == 1, (
        f"expected exactly one LocalEmbeddingClient across 3 calls, got "
        f"{len(constructed)}"
    )


# ── End-to-end via MetricSqlCompiler — remaining error paths ────────────────


def test_unindexed_source_table_includes_did_you_mean(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_resolve_entity_table`` attaches did_you_mean on unindexed tables."""
    from pretensor.intelligence.embeddings import LocalEmbeddingClient

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _raise_import_error)

    store = _build_store(tmp_path)
    try:
        # Build a layer whose entity points at a non-existent table on a
        # store that already indexes ``public.orders`` and ``public.users``.
        bad_entity = Entity(
            name="orders_typo",
            description="entity referencing a table that doesn't exist",
            source_table="public.ordrs",  # typo for "orders"
            attributes=[
                Attribute(
                    name="amount",
                    description="m",
                    role=AttributeRole.MEASURE,
                    source_column="amount",
                )
            ],
            metrics=[
                Metric(
                    name="total",
                    description="sum",
                    type=MetricType.SUM,
                    field="amount",
                )
            ],
        )
        layer = SemanticLayerModel(
            connection_name="demo",
            domains=[Domain(name="sales", description="s", entities=[bad_entity])],
        )
        compiler = MetricSqlCompiler(store, connection_name="demo", database_key="demo")

        with pytest.raises(MetricCompileError) as exc_info:
            compiler.compile(layer, "total")

        exc = exc_info.value
        assert exc.did_you_mean, "unresolved source_table should populate did_you_mean"
        assert "public.orders" in exc.did_you_mean
        # Type-awareness: only table qualnames, not metric/column names.
        assert "amount" not in exc.did_you_mean
        assert "total" not in exc.did_you_mean
        # Hint surfaces in the formatted message too.
        assert "did you mean" in str(exc).lower()
    finally:
        store.close()


def test_derived_metric_unknown_table_includes_did_you_mean(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_compile_derived`` attaches did_you_mean on unknown table refs in SQL."""
    from pretensor.intelligence.embeddings import LocalEmbeddingClient

    monkeypatch.setattr(LocalEmbeddingClient, "embed", _raise_import_error)

    store = _build_store(tmp_path)
    try:
        layer = _layer(
            metrics=[
                Metric(
                    name="ratio",
                    description="derived metric over a typo'd table",
                    type=MetricType.DERIVED,
                    expression=(
                        # Reference 'public.ordrs' (typo for orders) — the
                        # derived-metric SQL parser should surface the
                        # unknown-table error with did_you_mean populated.
                        "SELECT SUM(amount) / NULLIF(COUNT(*), 0) FROM public.ordrs"
                    ),
                )
            ]
        )
        compiler = MetricSqlCompiler(store, connection_name="demo", database_key="demo")

        with pytest.raises(MetricCompileError) as exc_info:
            compiler.compile(layer, "ratio")

        exc = exc_info.value
        assert exc.did_you_mean, (
            "unknown table reference in derived SQL should populate did_you_mean"
        )
        assert "public.orders" in exc.did_you_mean
        assert "did you mean" in str(exc).lower()
    finally:
        store.close()
