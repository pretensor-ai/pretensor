"""Unit tests for intelligence/pipeline.py — run_intelligence_layer orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pretensor.core.builder import GraphBuilder
from pretensor.core.store import KuzuStore
from pretensor.intelligence.pipeline import build_oss_pipeline, run_intelligence_layer
from pretensor.intelligence.steps import (
    CyclicDependencyError,
    PipelineContext,
    PipelineRunner,
    PipelineStep,
)


def _count(store: KuzuStore, query: str, params: dict | None = None) -> int:
    rows = store.query_all_rows(query, params or {})
    return int(rows[0][0]) if rows else 0


# ---------------------------------------------------------------------------
# Existing integration tests (unchanged semantics)
# ---------------------------------------------------------------------------


def test_empty_store_returns_early_without_error(tmp_path: Path) -> None:
    store = KuzuStore(tmp_path / "empty.kuzu")
    store.ensure_schema()
    try:
        # No tables indexed → should return early with no exception
        asyncio.run(run_intelligence_layer(store, "demo"))
    finally:
        store.close()


def test_full_run_heuristic_only(tmp_path: Path, load_schema) -> None:
    snap = load_schema("tpch")
    store = KuzuStore(tmp_path / "tpch.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        asyncio.run(run_intelligence_layer(store, "tpch"))

        cluster_count = _count(
            store,
            "MATCH (c:Cluster) WHERE c.database_key = $db RETURN count(c)",
            {"db": "tpch"},
        )
        assert cluster_count > 0, "expected at least one cluster after intelligence run"
    finally:
        store.close()


def test_full_run_with_llm_client_ignored(tmp_path: Path, load_schema) -> None:
    """Labeling always uses heuristic path."""
    snap = load_schema("pagila")
    store = KuzuStore(tmp_path / "pagila.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
        asyncio.run(run_intelligence_layer(store, "pagila"))

        cluster_count = _count(
            store,
            "MATCH (c:Cluster) WHERE c.database_key = $db RETURN count(c)",
            {"db": "pagila"},
        )
        assert cluster_count > 0
        rows = store.query_all_rows(
            "MATCH (c:Cluster) WHERE c.database_key = $db RETURN c.label",
            {"db": "pagila"},
        )
        labels = [str(r[0]) for r in rows if r[0]]
        assert len(labels) > 0
    finally:
        store.close()


def test_rerun_clears_prior_artifacts(tmp_path: Path, load_schema) -> None:
    snap = load_schema("analytics_dwh")
    store = KuzuStore(tmp_path / "dwh.kuzu")
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)

        asyncio.run(run_intelligence_layer(store, "analytics"))
        count_first = _count(
            store,
            "MATCH (c:Cluster) WHERE c.database_key = $db RETURN count(c)",
            {"db": "analytics"},
        )

        # Second run should clear and rebuild — not accumulate
        asyncio.run(run_intelligence_layer(store, "analytics"))
        count_second = _count(
            store,
            "MATCH (c:Cluster) WHERE c.database_key = $db RETURN count(c)",
            {"db": "analytics"},
        )

        assert count_first == count_second, "rerun should not duplicate clusters"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# PipelineContext unit tests
# ---------------------------------------------------------------------------


def test_pipeline_context_get_set() -> None:
    ctx = PipelineContext(foo="bar")
    assert ctx.get("foo") == "bar"
    ctx.set("baz", 42)
    assert ctx.get("baz") == 42


def test_pipeline_context_get_default() -> None:
    ctx = PipelineContext()
    assert ctx.get("missing") is None
    assert ctx.get("missing", "fallback") == "fallback"


def test_pipeline_context_contains() -> None:
    ctx = PipelineContext(x=1)
    assert "x" in ctx
    assert "y" not in ctx


# ---------------------------------------------------------------------------
# PipelineRunner unit tests
# ---------------------------------------------------------------------------


class _RecordingStep:
    """Test step that appends its name to a shared log list."""

    def __init__(self, name: str, dependencies: list[str] | None = None) -> None:
        self.name = name
        self.dependencies: list[str] = dependencies or []

    async def execute(self, ctx: PipelineContext) -> None:
        log: list[str] = ctx.get("log", [])
        log.append(self.name)
        ctx.set("log", log)


def test_runner_executes_steps_in_dependency_order() -> None:
    runner = PipelineRunner(
        [
            _RecordingStep("b", ["a"]),
            _RecordingStep("c", ["b"]),
            _RecordingStep("a"),
        ]
    )
    ctx = PipelineContext(log=[])
    asyncio.run(runner.run(ctx))
    assert ctx.get("log") == ["a", "b", "c"]


def test_runner_executes_single_step() -> None:
    runner = PipelineRunner([_RecordingStep("only")])
    ctx = PipelineContext(log=[])
    asyncio.run(runner.run(ctx))
    assert ctx.get("log") == ["only"]


def test_runner_executes_independent_steps_in_registration_order() -> None:
    runner = PipelineRunner(
        [
            _RecordingStep("first"),
            _RecordingStep("second"),
        ]
    )
    ctx = PipelineContext(log=[])
    asyncio.run(runner.run(ctx))
    assert ctx.get("log") == ["first", "second"]


def test_runner_empty_pipeline_is_noop() -> None:
    runner = PipelineRunner()
    ctx = PipelineContext()
    asyncio.run(runner.run(ctx))  # must not raise


def test_runner_raises_on_unknown_dependency() -> None:
    runner = PipelineRunner([_RecordingStep("a", ["nonexistent"])])
    ctx = PipelineContext()
    with pytest.raises(ValueError, match="unknown dependency"):
        asyncio.run(runner.run(ctx))


def test_runner_raises_on_cyclic_dependency() -> None:
    runner = PipelineRunner(
        [
            _RecordingStep("a", ["b"]),
            _RecordingStep("b", ["a"]),
        ]
    )
    ctx = PipelineContext()
    with pytest.raises(CyclicDependencyError):
        asyncio.run(runner.run(ctx))


def test_runner_raises_on_duplicate_step_name() -> None:
    runner = PipelineRunner([_RecordingStep("a")])
    with pytest.raises(ValueError, match="already registered"):
        runner.register(_RecordingStep("a"))


def test_runner_register_appends_step() -> None:
    runner = PipelineRunner([_RecordingStep("a")])
    runner.register(_RecordingStep("b", ["a"]))
    ctx = PipelineContext(log=[])
    asyncio.run(runner.run(ctx))
    assert ctx.get("log") == ["a", "b"]


# ---------------------------------------------------------------------------
# PipelineStep Protocol structural check
# ---------------------------------------------------------------------------


def test_recording_step_satisfies_protocol() -> None:
    step = _RecordingStep("x")
    assert isinstance(step, PipelineStep)


# ---------------------------------------------------------------------------
# build_oss_pipeline tests
# ---------------------------------------------------------------------------


def test_build_oss_pipeline_returns_runner_with_four_steps() -> None:
    runner = build_oss_pipeline()
    # Access internal steps to verify count and names
    step_names = [s.name for s in runner._steps]  # type: ignore[attr-defined]
    assert set(step_names) == {"classify", "cluster", "label", "join_paths"}
    assert len(step_names) == 4


def test_build_oss_pipeline_can_register_extra_step() -> None:
    runner = build_oss_pipeline()
    extra = _RecordingStep("llm_refine", ["label"])
    runner.register(extra)
    step_names = [s.name for s in runner._steps]  # type: ignore[attr-defined]
    assert "llm_refine" in step_names


def test_build_oss_pipeline_does_not_share_state_between_calls() -> None:
    r1 = build_oss_pipeline()
    r2 = build_oss_pipeline()
    r1.register(_RecordingStep("extra", ["metrics"]))
    # r2 must be unaffected
    names_r2 = [s.name for s in r2._steps]  # type: ignore[attr-defined]
    assert "extra" not in names_r2
