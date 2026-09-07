"""Tests for the CLI prompt abstraction."""

from __future__ import annotations

import io

import pytest

from pretensor.cli.prompts import ScriptedPrompter, is_noninteractive


def test_scripted_prompter_returns_answers_in_order() -> None:
    p = ScriptedPrompter(["alice", 2, True])
    assert p.text("name?") == "alice"
    assert p.choose("pick?", ["a", "b", "c"]) == 2
    assert p.confirm("sure?") is True


def test_scripted_prompter_none_means_default() -> None:
    p = ScriptedPrompter([None, None])
    assert p.text("name?", default="bob") == "bob"
    assert p.confirm("sure?", default=False) is False


def test_scripted_prompter_raises_when_exhausted() -> None:
    p = ScriptedPrompter([])
    with pytest.raises(AssertionError, match="ran out of answers"):
        p.text("name?")


def test_noninteractive_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRETENSOR_NONINTERACTIVE", "1")
    assert is_noninteractive() is True


def test_noninteractive_when_stdin_not_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRETENSOR_NONINTERACTIVE", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("piped input"))
    assert is_noninteractive() is True
