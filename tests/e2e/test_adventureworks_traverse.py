"""Regression: traverse must not emit audit-column coincidence joins.

The bug: the MCP ``traverse`` ranker once preferred spurious 1-hop joins on
``modifieddate = modifieddate`` (or other shared audit columns) over real
multi-hop FK chains. AdventureWorks puts ``modifieddate`` on every table across
five schemas (~40 tables here) — exactly the pathology Pagila is too small to
exercise meaningfully.

The fix lives in ``intelligence/heuristic.py::_BORING_SHARED_NAMES`` — audit
column names (``modifieddate``, ``rowguid``, ``last_update``) are dropped from
same-name candidate generation. These tests assert the fix holds end-to-end:
no returned step in any AW traverse result uses an audit-column coincidence.

The tests deliberately do **not** assert overall path confidence: AW exercises
a separate (out-of-scope) gap in the Postgres FK introspection query that can
cause legitimate same-name heuristics (``productid``, ``businessentityid``) to
appear in lower-confidence paths. Those are unrelated and should not cause this
regression suite to flake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pretensor.mcp.tools.traverse import traverse_payload

_DB = "adventureworks"

# Column names the audit-column fix explicitly filters from same-name heuristics.
# Keep in sync with ``_BORING_SHARED_NAMES`` in
# ``src/pretensor/intelligence/heuristic.py``.
_AUDIT_COLUMN_NAMES = frozenset(
    {
        "modifieddate",
        "rowguid",
        "last_update",
        "lastupdate",
        "modified_at",
        "created_at",
        "updated_at",
        "deleted_at",
    }
)


def _assert_no_audit_column_coincidence(result: dict[str, Any]) -> None:
    assert isinstance(result, dict), f"Expected dict result, got {type(result)}"
    paths = result.get("paths", [])
    for path in paths:
        for step in path.get("steps", []):
            if step.get("edge_type") != "inferred":
                continue
            from_col = str(step.get("from_column", "")).lower()
            to_col = str(step.get("to_column", "")).lower()
            via = str(step.get("via", "")).lower()
            for audit in _AUDIT_COLUMN_NAMES:
                if from_col == audit or to_col == audit or audit in via:
                    raise AssertionError(
                        "Traverse returned an audit-column coincidence join "
                        f"(audit-column regression on {audit!r}): {step}"
                    )


def test_traverse_product_to_customer_prefers_fk_chain(
    graph_dir_adventureworks: Path,
) -> None:
    """production.product → sales.customer must traverse the real FK chain.

    Real path (4 undirected hops):
        product → specialofferproduct → salesorderdetail
              → salesorderheader → customer
    Spurious path (if the audit-column fix regresses): 1-hop product↔customer via
    ``modifieddate = modifieddate`` at confidence 0.25.
    """
    result = traverse_payload(
        graph_dir_adventureworks,
        from_table="production.product",
        to_table="sales.customer",
        database=_DB,
    )
    _assert_no_audit_column_coincidence(result)


def test_traverse_employee_to_salesorderheader_prefers_fk_chain(
    graph_dir_adventureworks: Path,
) -> None:
    """humanresources.employee → sales.salesorderheader must go through salesperson.

    Real path (2 hops): employee → salesperson → salesorderheader.
    Spurious (regression): 1-hop employee↔salesorderheader via modifieddate.
    """
    result = traverse_payload(
        graph_dir_adventureworks,
        from_table="humanresources.employee",
        to_table="sales.salesorderheader",
        database=_DB,
    )
    _assert_no_audit_column_coincidence(result)


def test_traverse_person_to_product_no_spurious_path(
    graph_dir_adventureworks: Path,
) -> None:
    """person.person → production.product has no real FK chain within max_depth=4.

    The shortest real chain is ≥5 hops (person → customer → salesorderheader
    → salesorderdetail → specialofferproduct → product), which exceeds the
    default ``max_depth=4``. The tool must therefore return either no paths
    or only high-confidence paths — never a ``modifieddate`` coincidence join.
    """
    result = traverse_payload(
        graph_dir_adventureworks,
        from_table="person.person",
        to_table="production.product",
        database=_DB,
    )
    _assert_no_audit_column_coincidence(result)
