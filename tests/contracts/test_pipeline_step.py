"""Contract tests for PipelineStep extension point and the five OSS step implementations.

``PipelineStepContractTest`` verifies that any object satisfying the
:class:`~pretensor.intelligence.steps.PipelineStep` Protocol can be used by
:class:`~pretensor.intelligence.steps.PipelineRunner`.  Cloud implementations
import this class and bind ``make_step`` to their own factory.

Concrete tests for the five OSS steps (classify, cluster, label, join_paths,
metrics) each subclass :class:`PipelineStepContractTest` and exercise their
structural contract.  Full integration (executing the steps against a live
Kuzu store) is covered by ``tests/intelligence/test_pipeline.py``.
"""

from __future__ import annotations

import abc
import asyncio

import pytest

from pretensor.intelligence.pipeline import build_oss_pipeline
from pretensor.intelligence.steps import PipelineContext, PipelineRunner, PipelineStep

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _MinimalStep:
    """Minimal valid PipelineStep for contract verification."""

    name = "minimal"
    dependencies: list[str] = []

    async def execute(self, ctx: PipelineContext) -> None:
        ctx.set("minimal_ran", True)


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------


class PipelineStepContractTest(abc.ABC):
    """Reusable contract suite for :class:`PipelineStep` implementations.

    Subclass and implement :meth:`make_step` to verify any step.

    Example (Cloud)::

        class TestMyCloudStep(PipelineStepContractTest):
            def make_step(self) -> object:
                return MyCloudStep()
    """

    @abc.abstractmethod
    def make_step(self) -> PipelineStep:
        """Return an instance of the step under test."""

    # -- Structural Protocol check --------------------------------------------

    def test_satisfies_pipeline_step_protocol(self) -> None:
        """Step must satisfy the PipelineStep runtime-checkable Protocol."""
        step = self.make_step()
        assert isinstance(step, PipelineStep), (
            f"{type(step).__name__} does not satisfy PipelineStep protocol"
        )

    # -- Required attributes --------------------------------------------------

    def test_has_name_attribute(self) -> None:
        """Step must expose a non-empty string ``name`` attribute."""
        step = self.make_step()
        assert hasattr(step, "name"), "step must have a 'name' attribute"
        assert isinstance(step.name, str)
        assert len(step.name) > 0

    def test_has_dependencies_attribute(self) -> None:
        """Step must expose a ``dependencies`` list attribute."""
        step = self.make_step()
        assert hasattr(step, "dependencies"), "step must have a 'dependencies' attribute"
        assert isinstance(step.dependencies, list)

    def test_dependencies_are_strings(self) -> None:
        """Every entry in ``dependencies`` must be a string."""
        step = self.make_step()
        for dep in step.dependencies:
            assert isinstance(dep, str), (
                f"dependency {dep!r} is not a string"
            )

    def test_has_execute_method(self) -> None:
        """Step must have an ``execute`` coroutine method."""
        step = self.make_step()
        assert hasattr(step, "execute"), "step must have an 'execute' method"
        assert callable(step.execute)

    # -- PipelineRunner integration -------------------------------------------

    def test_can_be_registered_in_runner(self) -> None:
        """Step must be registerable in a PipelineRunner without raising."""
        step = self.make_step()
        runner = PipelineRunner()
        runner.register(step)

    def test_duplicate_name_raises_value_error(self) -> None:
        """Registering two steps with the same name must raise ValueError."""
        step = self.make_step()
        step_name = step.name
        runner = PipelineRunner()
        runner.register(step)

        class _Duplicate:
            name = step_name
            dependencies: list[str] = []

            async def execute(self, ctx: PipelineContext) -> None:
                pass

        with pytest.raises(ValueError, match="already registered"):
            runner.register(_Duplicate())


# ---------------------------------------------------------------------------
# Concrete tests for each OSS step
# ---------------------------------------------------------------------------


class TestClassifyStepContract(PipelineStepContractTest):
    """Contract for the ``classify`` OSS step."""

    def make_step(self) -> PipelineStep:
        runner = build_oss_pipeline()
        return next(s for s in runner._steps if s.name == "classify")  # type: ignore[attr-defined]

    def test_name_is_classify(self) -> None:
        assert self.make_step().name == "classify"

    def test_has_no_dependencies(self) -> None:
        step = self.make_step()
        assert step.dependencies == []


class TestClusterStepContract(PipelineStepContractTest):
    """Contract for the ``cluster`` OSS step."""

    def make_step(self) -> PipelineStep:
        runner = build_oss_pipeline()
        return next(s for s in runner._steps if s.name == "cluster")  # type: ignore[attr-defined]

    def test_name_is_cluster(self) -> None:
        assert self.make_step().name == "cluster"

    def test_depends_on_classify(self) -> None:
        step = self.make_step()
        assert "classify" in step.dependencies


class TestLabelStepContract(PipelineStepContractTest):
    """Contract for the ``label`` OSS step."""

    def make_step(self) -> PipelineStep:
        runner = build_oss_pipeline()
        return next(s for s in runner._steps if s.name == "label")  # type: ignore[attr-defined]

    def test_name_is_label(self) -> None:
        assert self.make_step().name == "label"

    def test_depends_on_cluster(self) -> None:
        step = self.make_step()
        assert "cluster" in step.dependencies


class TestJoinPathsStepContract(PipelineStepContractTest):
    """Contract for the ``join_paths`` OSS step."""

    def make_step(self) -> PipelineStep:
        runner = build_oss_pipeline()
        return next(s for s in runner._steps if s.name == "join_paths")  # type: ignore[attr-defined]

    def test_name_is_join_paths(self) -> None:
        assert self.make_step().name == "join_paths"

    def test_depends_on_label(self) -> None:
        step = self.make_step()
        assert "label" in step.dependencies


# ---------------------------------------------------------------------------
# MinimalStep as an additional structural check
# ---------------------------------------------------------------------------


class TestMinimalStepContract(PipelineStepContractTest):
    """Verify that a duck-typed minimal step also satisfies the contract."""

    def make_step(self) -> PipelineStep:
        return _MinimalStep()

    def test_execute_sets_context_value(self) -> None:
        """The minimal step's execute must update the context."""
        step = _MinimalStep()
        ctx = PipelineContext()
        asyncio.run(step.execute(ctx))
        assert ctx.get("minimal_ran") is True


# ---------------------------------------------------------------------------
# OSS pipeline structural tests
# ---------------------------------------------------------------------------


def test_oss_pipeline_has_four_steps() -> None:
    """build_oss_pipeline() must return exactly four named steps."""
    runner = build_oss_pipeline()
    names = {s.name for s in runner._steps}  # type: ignore[attr-defined]
    assert names == {"classify", "cluster", "label", "join_paths"}


def test_oss_pipeline_step_dependency_chain() -> None:
    """The OSS dependency chain must form classify→cluster→label→join_paths."""
    runner = build_oss_pipeline()
    by_name = {s.name: s for s in runner._steps}  # type: ignore[attr-defined]
    assert by_name["cluster"].dependencies == ["classify"]
    assert by_name["label"].dependencies == ["cluster"]
    assert by_name["join_paths"].dependencies == ["label"]


def test_all_oss_steps_satisfy_protocol() -> None:
    """Every step in the OSS pipeline must satisfy PipelineStep."""
    runner = build_oss_pipeline()
    for step in runner._steps:  # type: ignore[attr-defined]
        assert isinstance(step, PipelineStep), (
            f"{type(step).__name__} does not satisfy PipelineStep"
        )
