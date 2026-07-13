"""Fixtures shared across all CLI tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _deterministic_rich_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force deterministic Rich console output for every CLI test.

    NO_COLOR=1  – suppresses ANSI codes in Console() instances created during
                  test invocations (e.g. the inline Console() in main.py).
    COLUMNS=200 – Rich reads this lazily at each print call, so even the
                  module-level Console() (created at import time) uses a wide
                  fixed width, preventing line-folding that can break path
                  substring assertions.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
