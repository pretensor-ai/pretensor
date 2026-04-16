"""Tests for entities/llm_extract.py — Protocol, NullClient, ExtractedEntity."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pytest

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.entities.llm_extract import (
    ExtractedEntity,
    LLMEntityExtractor,
    NullLlmEntityClient,
)


def _snap() -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name="c",
        database="d",
        schemas=["public"],
        tables=[
            Table(
                name="film",
                schema_name="public",
                columns=[Column(name="film_id", data_type="int")],
            )
        ],
        introspected_at=datetime.now(timezone.utc),
    )


def test_null_client_returns_empty_yaml() -> None:
    client = NullLlmEntityClient()
    result = asyncio.run(client.extract_entities(prompt="anything"))
    assert result == "[]"


def test_null_client_emits_timing_log(caplog: pytest.LogCaptureFixture) -> None:
    client = NullLlmEntityClient()
    with caplog.at_level(logging.INFO):
        result = asyncio.run(client.extract_entities(prompt="anything"))
    assert result == "[]"
    records = [
        r for r in caplog.records if getattr(r, "event", "") == "llm.extract_entities"
    ]
    assert records
    assert getattr(records[-1], "status", None) == "ok"


def test_extracted_entity_defaults() -> None:
    e = ExtractedEntity(name="Film")
    assert e.name == "Film"
    assert e.description == ""
    assert e.tables == []


def test_llm_entity_extractor_returns_empty_in_oss() -> None:
    ext = LLMEntityExtractor()
    got = asyncio.run(ext.extract(_snap()))
    assert got == []


def test_llm_entity_extractor_null_client_default() -> None:
    ext = LLMEntityExtractor(client=None)
    got = asyncio.run(ext.extract(_snap()))
    assert got == []
