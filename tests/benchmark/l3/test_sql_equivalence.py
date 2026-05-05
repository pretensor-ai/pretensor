"""Unit tests for the L3 row-equivalence layer.

These tests are pure-Python and have no DB / LLM dependencies.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal

from pretensor.benchmark.l3.sql_equivalence import (
    gold_is_ordered,
    normalize_row,
    normalize_value,
    rows_equivalent,
)

# ---------------------------------------------------------------------------
# gold_is_ordered
# ---------------------------------------------------------------------------


def test_gold_is_ordered_detects_top_level_order_by() -> None:
    """A query with top-level ORDER BY is treated as ordered."""
    assert gold_is_ordered("SELECT a FROM t ORDER BY a") is True


def test_gold_is_ordered_false_when_no_order_by() -> None:
    assert gold_is_ordered("SELECT a FROM t") is False


def test_gold_is_ordered_ignores_order_by_in_subquery() -> None:
    """ORDER BY inside a subquery doesn't make the outer result ordered."""
    sql = "SELECT * FROM (SELECT a FROM t ORDER BY a) sq"
    assert gold_is_ordered(sql) is False


def test_gold_is_ordered_ignores_order_by_in_cte() -> None:
    sql = "WITH x AS (SELECT a FROM t ORDER BY a) SELECT * FROM x"
    assert gold_is_ordered(sql) is False


def test_gold_is_ordered_returns_false_on_parse_error() -> None:
    """Unparseable input falls back to permissive multiset semantics."""
    assert gold_is_ordered("this is not sql ::: $$$") is False


def test_gold_is_ordered_handles_order_by_with_limit() -> None:
    assert gold_is_ordered("SELECT a FROM t ORDER BY a LIMIT 5") is True


# ---------------------------------------------------------------------------
# normalize_value
# ---------------------------------------------------------------------------


def test_normalize_value_decimal_strips_trailing_zeros() -> None:
    """Decimal('1.00') and Decimal('1') compare equal post-normalisation."""
    assert normalize_value(Decimal("1.00")) == normalize_value(Decimal("1"))


def test_normalize_value_int_and_decimal_compare_equal() -> None:
    """int 1 and Decimal('1') collapse to the same canonical form."""
    assert normalize_value(1) == normalize_value(Decimal("1"))


def test_normalize_value_float_passes_through_str_first() -> None:
    """0.1 stays 0.1 — does not bloat through Decimal(0.1)."""
    assert normalize_value(0.1) == "0.1"


def test_normalize_value_int_and_float_compare_equal() -> None:
    """1 and 1.0 collapse together (rows shouldn't hinge on numeric typing)."""
    assert normalize_value(1) == normalize_value(1.0)


def test_normalize_value_date_uses_iso() -> None:
    assert normalize_value(date(2026, 4, 28)) == "2026-04-28"


def test_normalize_value_datetime_uses_iso() -> None:
    dt = datetime(2026, 4, 28, 12, 0, 0, tzinfo=timezone.utc)
    assert normalize_value(dt) == "2026-04-28T12:00:00+00:00"


def test_normalize_value_time_uses_iso() -> None:
    assert normalize_value(time(12, 30, 45)) == "12:30:45"


def test_normalize_value_none_is_distinct_from_string_none() -> None:
    """None must not collide with the string 'None'."""
    assert normalize_value(None) != normalize_value("None")


def test_normalize_value_bool_is_distinct_from_int() -> None:
    """True must not collide with 1 — bool is structurally different."""
    assert normalize_value(True) != normalize_value(1)
    assert normalize_value(False) != normalize_value(0)


def test_normalize_value_bytes_uses_hex() -> None:
    assert normalize_value(b"\xde\xad\xbe\xef") == "deadbeef"


def test_normalize_value_string_passthrough() -> None:
    assert normalize_value("hello") == "hello"


# ---------------------------------------------------------------------------
# normalize_row
# ---------------------------------------------------------------------------


def test_normalize_row_preserves_positional_order() -> None:
    row = (1, "x", None)
    assert normalize_row(row) == (
        normalize_value(1),
        normalize_value("x"),
        normalize_value(None),
    )


# ---------------------------------------------------------------------------
# rows_equivalent — happy paths
# ---------------------------------------------------------------------------


def test_rows_equivalent_unordered_multiset_matches_permutation() -> None:
    gold = [(1, "a"), (2, "b"), (3, "c")]
    agent = [(3, "c"), (1, "a"), (2, "b")]
    result = rows_equivalent(gold, agent, ordered=False)
    assert result.equivalent is True
    assert result.reason is None


def test_rows_equivalent_ordered_strict_rejects_permutation() -> None:
    gold = [(1, "a"), (2, "b"), (3, "c")]
    agent = [(3, "c"), (1, "a"), (2, "b")]
    result = rows_equivalent(gold, agent, ordered=True)
    assert result.equivalent is False


def test_rows_equivalent_decimal_normalises_across_rows() -> None:
    """Same numeric values, different Decimal scales — still equivalent."""
    gold = [(Decimal("1.00"),), (Decimal("2.0"),)]
    agent = [(Decimal("1"),), (Decimal("2"),)]
    result = rows_equivalent(gold, agent, ordered=True)
    assert result.equivalent is True


def test_rows_equivalent_int_vs_float_match() -> None:
    gold = [(1,), (2,), (3,)]
    agent = [(1.0,), (2.0,), (3.0,)]
    result = rows_equivalent(gold, agent, ordered=True)
    assert result.equivalent is True


def test_rows_equivalent_handles_empty_rowsets() -> None:
    result = rows_equivalent([], [], ordered=False)
    assert result.equivalent is True
    assert result.row_count_gold == 0
    assert result.row_count_agent == 0


def test_rows_equivalent_renamed_columns_compare_positionally() -> None:
    """Column rename in the agent SELECT doesn't break equivalence."""
    # Same VALUES; the SQL aliases differ but rows are positional tuples.
    gold = [("Comedy", 5)]
    agent = [("Comedy", 5)]
    result = rows_equivalent(gold, agent, ordered=False)
    assert result.equivalent is True


# ---------------------------------------------------------------------------
# rows_equivalent — failure modes
# ---------------------------------------------------------------------------


def test_rows_equivalent_row_count_mismatch_records_counts() -> None:
    gold = [(1,), (2,), (3,)]
    agent = [(1,), (2,)]
    result = rows_equivalent(gold, agent, ordered=False)
    assert result.equivalent is False
    assert result.row_count_gold == 3
    assert result.row_count_agent == 2
    assert "row count" in (result.reason or "")


def test_rows_equivalent_column_count_mismatch_returns_reason() -> None:
    gold = [(1, "a"), (2, "b")]
    agent = [(1,), (2,)]
    result = rows_equivalent(gold, agent, ordered=False)
    assert result.equivalent is False
    assert "column" in (result.reason or "")


def test_rows_equivalent_value_mismatch_returns_reason() -> None:
    gold = [(1, "a")]
    agent = [(1, "b")]
    result = rows_equivalent(gold, agent, ordered=False)
    assert result.equivalent is False
    assert result.reason is not None


def test_rows_equivalent_none_vs_zero_distinct() -> None:
    """NULL must not match 0 — common LLM mistake we MUST flag."""
    gold = [(None,)]
    agent = [(0,)]
    result = rows_equivalent(gold, agent, ordered=False)
    assert result.equivalent is False
