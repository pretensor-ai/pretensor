"""LLM interface types and text utilities (Cloud LLM infrastructure removed)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "ChatMessage",
    "LlmUsage",
    "LlmBudgetExceededError",
    "strip_json_fence",
    "strip_markdown_fence",
    "parse_json_array",
]


class LlmBudgetExceededError(Exception):
    """Raised when estimated spend for an index run reaches the configured ceiling."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Single chat turn for the shared completion API."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class LlmUsage:
    """Token usage reported by a provider (zeros if absent)."""

    input_tokens: int
    output_tokens: int


def strip_json_fence(raw: str) -> str:
    """Remove optional ```json ... ``` wrapping from model output."""
    return strip_markdown_fence(raw)


def strip_markdown_fence(raw: str) -> str:
    """Remove optional ```lang ... ``` wrapping (json, yaml, or bare)."""
    s = raw.strip()
    m = re.match(r"^```\w*\s*([\s\S]*?)\s*```$", s)
    if m:
        return m.group(1).strip()
    return s


def parse_json_array(raw: str) -> list[Any]:
    """Parse a JSON array, tolerating markdown fences."""
    s = strip_json_fence(raw)
    data = json.loads(s)
    if not isinstance(data, list):
        raise ValueError("expected JSON array")
    return data
