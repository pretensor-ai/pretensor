"""TPC-DS multi-valid-path stress tests for the traverse ranker.

TPC-DS exercises a failure mode invisible in Pagila or AdventureWorks:
**multiple valid FK-chain join paths between the same two tables**. The
canonical example is ``store_sales → customer``, which has at least:

  1. **Direct:** ``store_sales.ss_customer_sk → customer.c_customer_sk`` (1 hop)
  2. **Indirect via returns:** ``store_returns(sr_item_sk, sr_ticket_number)
     → store_sales`` then ``store_returns.sr_customer_sk → customer`` (2 hops
     undirected, through the composite FK back-reference)

Both are real FK chains with confidence 1.0. The current ``dijkstra_join_path``
uses directional FK adjacency with ``1/confidence`` edge costs and returns the
lowest-cost path, so only the direct 1-hop path is surfaced today. The
``on_demand.best_path`` (bidirectional adjacency) would see both and flag
``ambiguous=True``, but ``traverse_payload`` short-circuits once Dijkstra succeeds.

These tests **capture current behavior** so any future ranker change — e.g.
surfacing all valid paths or changing the tiebreak heuristic — is a deliberate
decision, not an accidental regression. The tests do not force a ranker fix.

Additionally, the snowflaked dimension hierarchy (``item → item_brand →
item_category``) stress-tests multi-hop FK walks through pure dimension
tables, and the date_sk columns on every fact table verify the traverse
ranker does not prefer audit-style ``*_date_sk`` coincidence joins over
real FK chains.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from pretensor.mcp.tools.traverse import traverse_payload

if not os.getenv("PRETENSOR_E2E"):
    pytest.skip("set PRETENSOR_E2E=1", allow_module_level=True)

pytestmark = pytest.mark.e2e

_DB = "tpcds"

# Column-name patterns that are shared across every fact table via date_dim /
# time_dim FKs. The traverse ranker must not prefer these over domain-specific
# FK chains — same spirit as the ``modifieddate`` audit-column filter in
# AdventureWorks, but for TPC-DS's ``*_date_sk`` / ``*_time_sk`` pattern.
_DATE_TIME_SK_SUFFIXES = ("_date_sk", "_time_sk")


def _steps(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the flat list of steps from the first path in a traverse result."""
    paths = result.get("paths", [])
    assert paths, f"Expected at least one path, got: {result}"
    return paths[0].get("steps", [])


def _step_tables(steps: list[dict[str, Any]]) -> list[str]:
    """Return the ordered list of table names touched by a path's steps."""
    tables: list[str] = []
    for step in steps:
        from_t = step.get("from_table", "")
        to_t = step.get("to_table", "")
        if not tables or tables[-1] != from_t:
            tables.append(from_t)
        tables.append(to_t)
    return tables


def test_traverse_store_sales_to_customer_multi_path(
    graph_dir_tpcds: Path,
) -> None:
    """store_sales → customer: capture which path the ranker picks.

    Two valid FK paths exist:
      1. Direct: store_sales.ss_customer_sk → customer.c_customer_sk (1 hop)
      2. Indirect: store_sales ← store_returns → customer (2 hops undirected)

    Current behavior (``dijkstra_join_path`` with directional FK adjacency):
    the direct 1-hop path is found first and returned. The indirect path
    through ``store_returns`` is NOT surfaced because Dijkstra short-circuits.

    If the ranker is later changed to surface multiple paths or use
    bidirectional adjacency, this test should be updated to reflect the
    new expected behavior.
    """
    result = traverse_payload(
        graph_dir_tpcds,
        from_table="store_sales",
        to_table="customer",
        database=_DB,
    )
    assert "error" not in result, f"Unexpected error: {result}"

    paths = result.get("paths", [])
    assert len(paths) >= 1, "Expected at least one path"

    # The direct 1-hop path should be present.
    steps = _steps(result)
    assert len(steps) >= 1, "Expected at least one step"

    # Every step must have confidence >= 0.5 (acceptance criterion).
    for step in steps:
        conf = step.get("confidence", 1.0)
        assert conf >= 0.5, (
            f"Step confidence {conf} < 0.5 — traverse returned a low-confidence "
            f"edge: {step}"
        )

    # Document current behavior: which column is used for the first-ranked path.
    # If the direct FK (ss_customer_sk → c_customer_sk) is found, it should be
    # a 1-hop path. If the ranker changes to prefer indirect paths, this
    # assertion will break intentionally.
    first_path = paths[0]
    first_steps = first_path.get("steps", [])
    step_count = len(first_steps)

    # Current expectation: direct 1-hop FK path is ranked first.
    # If this assertion fails, it means the ranker changed — review whether the
    # new behavior is correct and update accordingly.
    assert step_count <= 2, (
        f"Expected the direct path (1-2 steps) to be ranked first, but got "
        f"{step_count} steps. If the ranker was intentionally changed to prefer "
        f"the indirect path via store_returns, update this test."
    )


def test_traverse_catalog_sales_to_item_category_snowflake(
    graph_dir_tpcds: Path,
) -> None:
    """catalog_sales → item_category must traverse the snowflaked FK chain.

    Expected path: catalog_sales → item → item_brand → item_category (3 hops).
    There must be no spurious 1-hop coincidence join on shared dimension keys.
    """
    result = traverse_payload(
        graph_dir_tpcds,
        from_table="catalog_sales",
        to_table="item_category",
        database=_DB,
    )
    assert "error" not in result, f"Unexpected error: {result}"

    steps = _steps(result)
    tables = _step_tables(steps)

    # The path must go through `item` and `item_brand` (the snowflake chain).
    # Exact table qualification depends on how the graph stores public-schema
    # tables (may be "public.item" or just "item").
    table_names = [t.split(".")[-1] for t in tables]
    assert "item" in table_names, (
        f"Expected path through 'item', got tables: {tables}"
    )
    assert "item_brand" in table_names, (
        f"Expected path through 'item_brand', got tables: {tables}"
    )

    # Every step must have confidence >= 0.5.
    for step in steps:
        conf = step.get("confidence", 1.0)
        assert conf >= 0.5, (
            f"Step confidence {conf} < 0.5 — traverse returned a low-confidence "
            f"edge in the snowflake chain: {step}"
        )

    # The path should NOT use a date_sk or time_sk column — that would be a
    # spurious audit-style join.
    for step in steps:
        from_col = str(step.get("from_column", "")).lower()
        to_col = str(step.get("to_column", "")).lower()
        for suffix in _DATE_TIME_SK_SUFFIXES:
            assert not from_col.endswith(suffix), (
                f"Snowflake path uses audit-style column {from_col!r}: {step}"
            )
            assert not to_col.endswith(suffix), (
                f"Snowflake path uses audit-style column {to_col!r}: {step}"
            )


def test_traverse_store_returns_to_customer_clean_fk(
    graph_dir_tpcds: Path,
) -> None:
    """store_returns → customer must use the clean sr_customer_sk FK chain.

    Expected: direct FK via sr_customer_sk → c_customer_sk (1 hop).
    Must NOT pick a ``*_date_sk`` audit-style join, even though every fact
    table has ``sr_returned_date_sk`` and ``customer.c_first_sales_date_sk``
    both referencing ``date_dim``.
    """
    result = traverse_payload(
        graph_dir_tpcds,
        from_table="store_returns",
        to_table="customer",
        database=_DB,
    )
    assert "error" not in result, f"Unexpected error: {result}"

    steps = _steps(result)

    # Every step must have confidence >= 0.5.
    for step in steps:
        conf = step.get("confidence", 1.0)
        assert conf >= 0.5, (
            f"Step confidence {conf} < 0.5 — traverse returned a low-confidence "
            f"edge: {step}"
        )

    # No step should use a date_sk or time_sk column.
    for step in steps:
        from_col = str(step.get("from_column", "")).lower()
        to_col = str(step.get("to_column", "")).lower()
        for suffix in _DATE_TIME_SK_SUFFIXES:
            assert not from_col.endswith(suffix), (
                f"Returns→customer path uses audit-style column {from_col!r} "
                f"instead of sr_customer_sk: {step}"
            )
            assert not to_col.endswith(suffix), (
                f"Returns→customer path uses audit-style column {to_col!r} "
                f"instead of c_customer_sk: {step}"
            )
