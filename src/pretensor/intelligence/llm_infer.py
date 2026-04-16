"""LLM relationship inference Protocol and null client (Cloud extension point)."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from pretensor.connectors.models import SchemaSnapshot
from pretensor.observability import run_timed_async

__all__ = [
    "LlmRelationshipClient",
    "NullLlmRelationshipClient",
    "parse_llm_join_response_json",
]

logger = logging.getLogger(__name__)


class LlmRelationshipClient(Protocol):
    """Async client that returns structured join hypotheses (implement per provider)."""

    async def suggest_joins(
        self,
        *,
        snapshot_summary: dict[str, Any],
        batch_table_names: list[str],
    ) -> list[dict[str, Any]]:
        """Return dicts with keys: source_table, source_column, target_table, target_column, confidence, reasoning."""
        ...


class NullLlmRelationshipClient:
    """No-op client: returns no candidates (default until a provider is wired)."""

    async def suggest_joins(
        self,
        *,
        snapshot_summary: dict[str, Any],
        batch_table_names: list[str],
    ) -> list[dict[str, Any]]:
        async def _no_results() -> list[dict[str, Any]]:
            return []

        return await run_timed_async(
            logger,
            event="llm.suggest_joins",
            callback=_no_results,
            provider="null",
            batch_size=len(batch_table_names),
            snapshot_table_count=len(snapshot_summary.get("tables", []))
            if isinstance(snapshot_summary.get("tables"), list)
            else None,
        )


def _snapshot_summary(snapshot: SchemaSnapshot) -> dict[str, Any]:
    return {
        "connection_name": snapshot.connection_name,
        "database": snapshot.database,
        "tables": [
            {
                "schema": t.schema_name,
                "name": t.name,
                "columns": [c.name for c in t.columns],
            }
            for t in snapshot.tables
        ],
    }


def parse_llm_join_response_json(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON array of join objects from an LLM response body."""
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("LLM join response must be a JSON array")
    return [x for x in data if isinstance(x, dict)]
