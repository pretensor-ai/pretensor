"""Row-set equivalence for L3 NL-to-SQL grading.

The L3 baseline (and forthcoming pretensor) runner asks an LLM to translate
a natural-language question into SQL, then compares the result rows to those
returned by the gold SQL. "Equivalence" must tolerate cosmetic differences
the spec deems irrelevant while still catching real correctness failures.

Rules implemented here:

* **Ordered vs multiset** — when the gold query has a top-level ``ORDER BY``,
  rows are compared as an ordered list; otherwise as a sorted multiset. The
  detection ignores ``ORDER BY`` nested inside subqueries / CTEs / window
  frames.
* **Column comparison is positional** — a SELECT list with renamed columns
  is still equivalent. The gold author chose the column order; the agent
  matches positions, not names.
* **Value normalisation is deterministic** — ``Decimal('1.00')`` and
  ``Decimal('1')`` compare equal; dates / datetimes use ISO 8601; ``bytes``
  use hex; ``None`` is its own bucket. Floats are coerced through
  ``Decimal`` so ``1`` and ``1.0`` collapse together.

The module deliberately exposes only pure functions — no DB access, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

__all__ = [
    "EquivalenceResult",
    "gold_is_ordered",
    "normalize_row",
    "normalize_value",
    "rows_equivalent",
]


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    """Outcome of comparing gold vs agent rows.

    ``equivalent`` is the boolean grade the runner records. ``reason`` is a
    short string suitable for the per-item ``error`` field when the rows
    don't match (``None`` when equivalent). ``row_count_gold`` /
    ``row_count_agent`` make mismatches diagnosable from the JSON envelope
    alone — the most common failure mode is "agent returned 999 rows; gold
    returned 1000" and inspecting that without the JSON diff is painful.
    """

    equivalent: bool
    reason: str | None
    row_count_gold: int
    row_count_agent: int


def gold_is_ordered(gold_sql: str) -> bool:
    """Return ``True`` iff the gold query has a *top-level* ``ORDER BY``.

    Subquery / CTE / window-frame ``ORDER BY`` clauses are ignored — they
    don't affect the ordering of the outermost result set. Returns ``False``
    on parse failure so the comparison falls back to multiset semantics
    (the safer, more permissive choice).
    """
    try:
        parsed = sqlglot.parse_one(gold_sql, dialect="postgres")
    except SqlglotError:
        return False
    if parsed is None:
        return False
    # parsed.args["order"] is the top-level Order node; nested ORDER BYs
    # live inside the relevant Subquery/Window/CTE child trees and don't
    # appear here.
    order_node = parsed.args.get("order")
    return isinstance(order_node, exp.Order)


def normalize_value(value: Any) -> str:
    """Coerce a single cell to a deterministic string for comparison.

    Numerics route through ``Decimal`` so ``1``, ``1.0``, and
    ``Decimal('1.00')`` all collapse to the same canonical form.
    Temporal values use ISO 8601. Bytes use lowercase hex. ``None`` gets
    a sentinel so nullness is distinguishable from string ``"None"``.
    """
    if value is None:
        return "\x00NULL\x00"
    if isinstance(value, bool):
        # Must precede int branch — bool is a subclass of int in Python.
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        # ``Decimal`` normalises trailing zeros (1.00 → 1); float passes
        # through ``str`` first to avoid Decimal(0.1) inflating to a long
        # repeating expansion.
        as_decimal = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
        return format(as_decimal.normalize(), "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value)


def normalize_row(row: tuple[Any, ...] | list[Any]) -> tuple[str, ...]:
    """Apply ``normalize_value`` element-wise; preserves positional order."""
    return tuple(normalize_value(v) for v in row)


def rows_equivalent(
    gold_rows: list[tuple[Any, ...]] | list[list[Any]],
    agent_rows: list[tuple[Any, ...]] | list[list[Any]],
    *,
    ordered: bool,
) -> EquivalenceResult:
    """Decide whether two row lists are equivalent under the chosen rule.

    Column count must match (we never compare a 3-column gold against a
    2-column agent — that's a structural correctness failure). When
    ``ordered=True`` the rows are compared in order; when ``False`` they
    are compared as multisets via sorted normalised tuples.
    """
    n_gold = len(gold_rows)
    n_agent = len(agent_rows)
    if n_gold != n_agent:
        return EquivalenceResult(
            equivalent=False,
            reason=f"row count mismatch: gold={n_gold}, agent={n_agent}",
            row_count_gold=n_gold,
            row_count_agent=n_agent,
        )

    if n_gold == 0:
        return EquivalenceResult(
            equivalent=True,
            reason=None,
            row_count_gold=0,
            row_count_agent=0,
        )

    gold_norm = [normalize_row(r) for r in gold_rows]
    agent_norm = [normalize_row(r) for r in agent_rows]

    gold_widths = {len(r) for r in gold_norm}
    agent_widths = {len(r) for r in agent_norm}
    if gold_widths != agent_widths or len(gold_widths) != 1:
        return EquivalenceResult(
            equivalent=False,
            reason=(
                f"column count mismatch: gold={sorted(gold_widths)}, "
                f"agent={sorted(agent_widths)}"
            ),
            row_count_gold=n_gold,
            row_count_agent=n_agent,
        )

    if ordered:
        match = gold_norm == agent_norm
    else:
        match = sorted(gold_norm) == sorted(agent_norm)

    return EquivalenceResult(
        equivalent=match,
        reason=None if match else "row contents differ",
        row_count_gold=n_gold,
        row_count_agent=n_agent,
    )
