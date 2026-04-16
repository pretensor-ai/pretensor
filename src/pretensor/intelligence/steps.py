"""PipelineStep Protocol and PipelineRunner for the intelligence pipeline.

The runner resolves step execution order from declared dependencies and executes
each step in sequence.  Cloud can register additional steps (e.g. ``llm_refine``,
``feedback_score``, ``semantic_propose``) by appending them before calling
:meth:`PipelineRunner.run`.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

__all__ = ["PipelineStep", "PipelineRunner", "PipelineContext", "CyclicDependencyError"]

logger = logging.getLogger(__name__)


class CyclicDependencyError(ValueError):
    """Raised when the pipeline step dependency graph contains a cycle."""


class PipelineContext:
    """Shared mutable state threaded through every pipeline step.

    Steps read and write named artifacts via :meth:`get` and :meth:`set` so
    that later steps can consume outputs produced by earlier ones without
    coupling them directly.
    """

    def __init__(self, **initial: Any) -> None:
        self._data: dict[str, Any] = dict(initial)

    def set(self, key: str, value: Any) -> None:  # noqa: A003
        """Store *value* under *key*."""
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value stored under *key*, or *default*."""
        return self._data.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data


@runtime_checkable
class PipelineStep(Protocol):
    """Structural interface every intelligence pipeline step must satisfy.

    Implementing classes do **not** need to inherit from this class — any
    object with the right attributes and an ``execute`` coroutine qualifies.

    Attributes:
        name: Stable, unique identifier for the step (e.g. ``"classify"``).
        dependencies: Names of steps that must complete before this one runs.
    """

    name: str
    dependencies: list[str]

    async def execute(self, ctx: PipelineContext) -> None:
        """Run this step, reading inputs from and writing outputs to *ctx*.

        Args:
            ctx: Shared pipeline context populated by upstream steps.
        """
        ...


class PipelineRunner:
    """Topological executor for a set of :class:`PipelineStep` objects.

    Steps are registered in any order; the runner resolves execution order
    from ``step.dependencies`` before the first :meth:`run` call.

    Args:
        steps: Initial list of steps.  Additional steps can be appended via
            :meth:`register` before calling :meth:`run`.
    """

    def __init__(self, steps: list[PipelineStep] | None = None) -> None:
        self._steps: list[PipelineStep] = list(steps or [])

    def register(self, step: PipelineStep) -> None:
        """Append *step* to the pipeline.

        Args:
            step: A :class:`PipelineStep` implementation to add.

        Raises:
            ValueError: If a step with the same name is already registered.
        """
        existing = {s.name for s in self._steps}
        if step.name in existing:
            raise ValueError(
                f"A step named {step.name!r} is already registered in this pipeline."
            )
        self._steps.append(step)

    def _ordered(self) -> list[PipelineStep]:
        """Return steps in topological (dependency-resolved) order.

        Raises:
            ValueError: If a step lists an unknown dependency name.
            CyclicDependencyError: If the dependency graph contains a cycle.
        """
        by_name = {s.name: s for s in self._steps}
        for step in self._steps:
            for dep in step.dependencies:
                if dep not in by_name:
                    raise ValueError(
                        f"Step {step.name!r} declares unknown dependency {dep!r}."
                    )

        # Kahn's algorithm for topological sort
        in_degree: dict[str, int] = {name: 0 for name in by_name}
        dependents: dict[str, list[str]] = {name: [] for name in by_name}
        for step in self._steps:
            for dep in step.dependencies:
                dependents[dep].append(step.name)
                in_degree[step.name] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        ordered: list[PipelineStep] = []

        while queue:
            name = queue.pop(0)
            ordered.append(by_name[name])
            for dependent in dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(ordered) != len(self._steps):
            cycle_nodes = [n for n, d in in_degree.items() if d > 0]
            raise CyclicDependencyError(
                f"Pipeline contains a dependency cycle involving: {cycle_nodes}"
            )

        return ordered

    async def run(self, ctx: PipelineContext) -> None:
        """Execute all registered steps in dependency order.

        Args:
            ctx: Shared pipeline context passed to every step.

        Raises:
            ValueError: If any step has an unknown dependency.
            CyclicDependencyError: If the dependency graph contains a cycle.
        """
        ordered = self._ordered()
        logger.info(
            "PipelineRunner: executing %d step(s): %s",
            len(ordered),
            [s.name for s in ordered],
        )
        for step in ordered:
            logger.debug("PipelineRunner: starting step %r", step.name)
            await step.execute(ctx)
            logger.debug("PipelineRunner: finished step %r", step.name)
