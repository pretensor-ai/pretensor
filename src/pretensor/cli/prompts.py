"""Prompt abstraction for interactive CLI flows.

Wizard logic depends on the ``Prompter`` protocol, never on Rich directly, so
every step is testable without a terminal and the prompt implementation can be
swapped without touching wizard code.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Protocol

from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt

__all__ = ["Prompter", "RichPrompter", "ScriptedPrompter", "is_noninteractive"]

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def is_noninteractive() -> bool:
    """Return True when no human is available to answer a prompt."""
    if os.environ.get("PRETENSOR_NONINTERACTIVE", "").strip().lower() in _TRUE_VALUES:
        return True
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return True
    try:
        return not stdin.isatty()
    except (AttributeError, ValueError):
        return True


class Prompter(Protocol):
    """The four question shapes the wizard needs."""

    def text(self, question: str, *, default: str | None = None) -> str: ...

    def secret(self, question: str) -> str: ...

    def confirm(self, question: str, *, default: bool = True) -> bool: ...

    def choose(
        self, question: str, options: Sequence[str], *, default: int = 0
    ) -> int: ...


class RichPrompter:
    """Terminal implementation backed by ``rich.prompt``."""

    def __init__(self, console: Console) -> None:
        self._console = console

    def text(self, question: str, *, default: str | None = None) -> str:
        return Prompt.ask(question, default=default or "", console=self._console)

    def secret(self, question: str) -> str:
        return Prompt.ask(question, password=True, console=self._console)

    def confirm(self, question: str, *, default: bool = True) -> bool:
        return Confirm.ask(question, default=default, console=self._console)

    def choose(self, question: str, options: Sequence[str], *, default: int = 0) -> int:
        for number, option in enumerate(options, start=1):
            self._console.print(f"  {number}. {option}")
        choices = [str(n) for n in range(1, len(options) + 1)]
        answer = IntPrompt.ask(
            question,
            default=default + 1,
            choices=choices,
            console=self._console,
        )
        return int(answer) - 1


class ScriptedPrompter:
    """Test double. Answers are consumed in order; ``None`` means "take the default"."""

    def __init__(self, answers: Sequence[object]) -> None:
        self._answers = list(answers)

    def _next(self) -> object:
        if not self._answers:
            raise AssertionError("ScriptedPrompter ran out of answers")
        return self._answers.pop(0)

    def text(self, question: str, *, default: str | None = None) -> str:
        value = self._next()
        return (default or "") if value is None else str(value)

    def secret(self, question: str) -> str:
        return str(self._next())

    def confirm(self, question: str, *, default: bool = True) -> bool:
        value = self._next()
        return default if value is None else bool(value)

    def choose(self, question: str, options: Sequence[str], *, default: int = 0) -> int:
        value = self._next()
        return default if value is None else int(value)  # type: ignore[arg-type]
