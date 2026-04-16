"""Tests for intelligence/llm_infer.py — Protocol, NullClient, parse helpers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.intelligence.llm_infer import (
    NullLlmRelationshipClient,
    _snapshot_summary,
    parse_llm_join_response_json,
)


def _snap() -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name="demo",
        database="db",
        schemas=["public"],
        tables=[
            Table(
                name="orders",
                schema_name="public",
                columns=[Column(name="id", data_type="int", is_primary_key=True)],
            ),
        ],
        introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def test_null_client_returns_empty() -> None:
    client = NullLlmRelationshipClient()
    result = asyncio.run(
        client.suggest_joins(snapshot_summary={}, batch_table_names=["t1"])
    )
    assert result == []


def test_null_client_emits_timing_log(caplog: pytest.LogCaptureFixture) -> None:
    """Null client logs a timed LLM event for observability."""
    client = NullLlmRelationshipClient()
    with caplog.at_level("INFO"):
        result = asyncio.run(
            client.suggest_joins(
                snapshot_summary={"tables": [{"name": "t1"}]},
                batch_table_names=["t1"],
            )
        )
    assert result == []
    records = [r for r in caplog.records if getattr(r, "event", "") == "llm.suggest_joins"]
    assert records
    rec = records[-1]
    assert getattr(rec, "status", None) == "ok"
    assert getattr(rec, "provider", None) == "null"


def test_snapshot_summary_structure() -> None:
    s = _snapshot_summary(_snap())
    assert s["connection_name"] == "demo"
    assert s["database"] == "db"
    assert len(s["tables"]) == 1
    assert s["tables"][0]["name"] == "orders"
    assert s["tables"][0]["columns"] == ["id"]


def test_parse_llm_join_response_json_valid() -> None:
    raw = '[{"source_table": "a", "target_table": "b"}]'
    result = parse_llm_join_response_json(raw)
    assert len(result) == 1
    assert result[0]["source_table"] == "a"


def test_parse_llm_join_response_json_not_array() -> None:
    with pytest.raises(ValueError, match="JSON array"):
        parse_llm_join_response_json('{"key": "value"}')


def test_parse_llm_join_response_json_filters_non_dicts() -> None:
    raw = '[{"ok": true}, "skip", 42]'
    result = parse_llm_join_response_json(raw)
    assert len(result) == 1
    assert result[0]["ok"] is True
