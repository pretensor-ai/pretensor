"""Tests for the SQL confidence classifier."""

from __future__ import annotations

import pytest

from pretensor.enrichment.analyze.classify import CONFIDENCE_FLOAT, classify_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, name FROM public.users",
        "SELECT * FROM foo",
        "select id from t",
        "INSERT INTO public.orders (id) VALUES (1)",
        "UPDATE public.accounts SET balance = 0 WHERE id = 1",
        "DELETE FROM public.tmp WHERE created < '2020-01-01'",
        "WITH cte AS (SELECT 1) SELECT * FROM cte",
        "MERGE INTO target USING source ON target.id = source.id WHEN MATCHED THEN UPDATE SET x = 1",
    ],
)
def test_valid_sql_returns_high(sql: str) -> None:
    bucket, conf = classify_sql(sql)
    assert bucket == "high"
    assert conf == pytest.approx(CONFIDENCE_FLOAT["high"])


def test_malformed_sql_returns_medium() -> None:
    bucket, conf = classify_sql("SELECT FROM WHERE")
    assert bucket == "medium"
    assert conf == pytest.approx(CONFIDENCE_FLOAT["medium"])


def test_empty_string_returns_not_sql() -> None:
    bucket, conf = classify_sql("")
    assert bucket == ""
    assert conf == 0.0


def test_whitespace_only_returns_not_sql() -> None:
    bucket, conf = classify_sql("   \n\t  ")
    assert bucket == ""
    assert conf == 0.0


def test_no_keyword_at_start_returns_not_sql() -> None:
    bucket, conf = classify_sql("WHERE id = 1")
    assert bucket == ""
    assert conf == 0.0


def test_keyword_in_middle_returns_not_sql() -> None:
    bucket, conf = classify_sql("some_function(SELECT 1)")
    assert bucket == ""
    assert conf == 0.0


def test_fragment_starting_with_from_returns_not_sql() -> None:
    bucket, conf = classify_sql("FROM public.users")
    assert bucket == ""
    assert conf == 0.0


def test_confidence_float_mapping_complete() -> None:
    assert set(CONFIDENCE_FLOAT.keys()) == {"high", "medium", "low"}
    assert (
        CONFIDENCE_FLOAT["high"] > CONFIDENCE_FLOAT["medium"] > CONFIDENCE_FLOAT["low"]
    )


def test_select_one_returns_high() -> None:
    bucket, conf = classify_sql("SELECT 1")
    assert bucket == "high"


def test_static_fstring_prefix_parses_partially() -> None:
    prefix = "SELECT * FROM public.users WHERE id = "
    bucket, conf = classify_sql(prefix)
    assert bucket in ("high", "medium")
    assert conf > 0.0


def test_parameterized_sql_classifies_high() -> None:
    bucket, conf = classify_sql(
        "INSERT INTO public.payment (customer_id, amount) VALUES (%s, %s)"
    )
    assert bucket == "high"
    assert conf == pytest.approx(CONFIDENCE_FLOAT["high"])


def test_named_bind_classifies_high() -> None:
    bucket, _ = classify_sql("SELECT * FROM t WHERE id = :ident")
    assert bucket == "high"


def test_replace_into_classifies_as_sql() -> None:
    bucket, conf = classify_sql("REPLACE INTO staff (id, name) VALUES (%s, %s)")
    assert bucket == "high"
    assert conf == pytest.approx(0.9)
