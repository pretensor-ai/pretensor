"""Tests for PostgreSQL array literal parsing."""

from __future__ import annotations

from pretensor.connectors.pg_array_parse import parse_pg_array_literal


def test_parse_empty() -> None:
    assert parse_pg_array_literal(None) == []
    assert parse_pg_array_literal("") == []
    assert parse_pg_array_literal("{}") == []


def test_parse_simple_tokens() -> None:
    assert parse_pg_array_literal("{a,b,c}") == ["a", "b", "c"]


def test_parse_quoted_with_comma() -> None:
    assert parse_pg_array_literal('{hello,"a,b"}') == ["hello", "a,b"]


def test_parse_escaped_quote() -> None:
    assert parse_pg_array_literal(r'{"a\"b"}') == ['a"b']
