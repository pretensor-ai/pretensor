"""Control case: traverse on TPC-H must return clean FK chains only.

TPC-H has zero audit columns — no ``last_update``, ``created_at``,
``modifieddate``, or ``rowguid``.  Every column name is unique to its
table (``l_orderkey``, ``o_custkey``, ``ps_partkey``, etc.), so the
same-column-name heuristic in ``intelligence/heuristic.py`` has no fuel
to generate spurious coincidence joins.

This makes TPC-H the **negative-space test** for the audit-column fix:
traverse output here should be byte-identical before and after the fix.
If it diverges, the fix has over-corrected — it is removing real edges,
not just audit-column noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pretensor.mcp.tools.traverse import traverse_payload

_DB = "tpch"

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
    """Fail if any returned path step uses an audit-column coincidence join."""
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


def _assert_all_steps_are_fk(result: dict[str, Any]) -> None:
    """Assert every step in every path uses a real FK edge, not an inferred one.

    TPC-H has explicit foreign keys for all relationships, so traverse
    should never need to fall back to heuristic inferred edges.
    """
    paths = result.get("paths", [])
    assert paths, "Expected at least one path, got none"
    for path in paths:
        for step in path.get("steps", []):
            assert step.get("edge_type") == "fk", (
                f"Expected edge_type='fk' but got {step.get('edge_type')!r}: {step}"
            )


def test_traverse_lineitem_to_customer(graph_dir_tpch: Path) -> None:
    """lineitem → customer must traverse the 2-hop FK chain via orders.

    Path: lineitem → orders (l_orderkey → o_orderkey)
          → customer (o_custkey → c_custkey)
    """
    result = traverse_payload(
        graph_dir_tpch,
        from_table="lineitem",
        to_table="customer",
        database=_DB,
    )
    _assert_no_audit_column_coincidence(result)
    _assert_all_steps_are_fk(result)


def test_traverse_part_to_customer(graph_dir_tpch: Path) -> None:
    """part → customer must traverse the 4-hop FK chain.

    Path: part → partsupp (p_partkey → ps_partkey)
          → lineitem (composite FK)
          → orders (l_orderkey → o_orderkey)
          → customer (o_custkey → c_custkey)
    """
    result = traverse_payload(
        graph_dir_tpch,
        from_table="part",
        to_table="customer",
        database=_DB,
    )
    _assert_no_audit_column_coincidence(result)
    _assert_all_steps_are_fk(result)


def test_traverse_supplier_to_nation(graph_dir_tpch: Path) -> None:
    """supplier → nation is a direct 1-hop FK via s_nationkey.

    Path: supplier → nation (s_nationkey → n_nationkey)
    """
    result = traverse_payload(
        graph_dir_tpch,
        from_table="supplier",
        to_table="nation",
        database=_DB,
    )
    _assert_no_audit_column_coincidence(result)
    _assert_all_steps_are_fk(result)
