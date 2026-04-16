"""LLM entity extraction Protocol and null client (Cloud extension point)."""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, Field

from pretensor.connectors.models import SchemaSnapshot
from pretensor.observability import run_timed_async

__all__ = [
    "ExtractedEntity",
    "LLMEntityExtractor",
    "LlmEntityClient",
    "NullLlmEntityClient",
]

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """One business entity grouping physical tables."""

    name: str
    description: str = ""
    tables: list[str] = Field(
        default_factory=list,
        description="Table refs: schema.name or bare name.",
    )


class LlmEntityClient(Protocol):
    """Async client returning YAML entity extraction (implement per provider)."""

    async def extract_entities(self, *, prompt: str) -> str:
        """Return YAML body (list of entity dicts)."""
        ...


class NullLlmEntityClient:
    """No-op client: returns an empty YAML list."""

    async def extract_entities(self, *, prompt: str) -> str:
        async def _empty_yaml() -> str:
            return "[]"

        return await run_timed_async(
            logger,
            event="llm.extract_entities",
            callback=_empty_yaml,
            provider="null",
            prompt_chars=len(prompt),
        )


class LLMEntityExtractor:
    """Runs entity extraction using a :class:`LlmEntityClient`."""

    def __init__(self, client: LlmEntityClient | None = None) -> None:
        self._client = client or NullLlmEntityClient()

    async def extract(self, snapshot: SchemaSnapshot) -> list[ExtractedEntity]:
        """Return extracted entities (empty in OSS; Cloud wires a real client)."""
        return []
