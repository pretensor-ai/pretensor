"""Tests for heuristic relationship discovery."""

from __future__ import annotations

from datetime import datetime, timezone

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.ids import table_node_id
from pretensor.intelligence.heuristic import discover_heuristic_candidates


def _snap(*tables: Table, connection_name: str = "demo") -> SchemaSnapshot:
    return SchemaSnapshot(
        connection_name=connection_name,
        database="db",
        schemas=["public"],
        tables=list(tables),
        introspected_at=datetime.now(timezone.utc),
    )


def test_customer_id_to_customers_id_high_confidence() -> None:
    customers = Table(
        name="customers",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="customer_id", data_type="int"),
        ],
    )
    snap = _snap(customers, orders)
    cands = discover_heuristic_candidates(snap)
    src = table_node_id("demo", "public", "orders")
    dst = table_node_id("demo", "public", "customers")
    match = [c for c in cands if c.source_node_id == src and c.target_node_id == dst]
    assert match, "expected orders.customer_id → customers.id"
    assert match[0].source_column == "customer_id"
    assert match[0].target_column == "id"
    assert match[0].confidence >= 0.8
    assert match[0].source == "heuristic"


def test_product_fk_suffix_high_confidence() -> None:
    products = Table(
        name="products",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    line_items = Table(
        name="line_items",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="product_fk", data_type="int"),
        ],
    )
    snap = _snap(products, line_items)
    cands = discover_heuristic_candidates(snap)
    src = table_node_id("demo", "public", "line_items")
    dst = table_node_id("demo", "public", "products")
    match = [c for c in cands if c.source_node_id == src and c.target_node_id == dst]
    assert match
    assert match[0].source_column == "product_fk"
    assert match[0].target_column == "id"


def test_abbreviated_cust_id_medium_confidence() -> None:
    customers = Table(
        name="customers",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="cust_id", data_type="int"),
        ],
    )
    snap = _snap(customers, orders)
    cands = discover_heuristic_candidates(snap)
    src = table_node_id("demo", "public", "orders")
    dst = table_node_id("demo", "public", "customers")
    match = [c for c in cands if c.source_node_id == src and c.target_node_id == dst]
    assert match
    assert match[0].source_column == "cust_id"
    assert 0.5 <= match[0].confidence < 0.8


def test_same_column_name_low_confidence_bidirectional() -> None:
    users = Table(
        name="users",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="email", data_type="text"),
        ],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="email", data_type="text"),
        ],
    )
    snap = _snap(users, orders)
    cands = discover_heuristic_candidates(snap)
    low = [c for c in cands if "shared column name 'email'" in c.reasoning]
    assert len(low) >= 2
    assert all(c.confidence <= 0.35 for c in low)


def test_unique_target_index_boosts_confidence() -> None:
    customers = Table(
        name="customers",
        schema_name="public",
        columns=[
            Column(
                name="id",
                data_type="int",
                is_primary_key=True,
                index_is_unique=True,
            ),
        ],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="customer_id", data_type="int"),
        ],
    )
    snap = _snap(customers, orders)
    cands = discover_heuristic_candidates(snap)
    src = table_node_id("demo", "public", "orders")
    dst = table_node_id("demo", "public", "customers")
    match = [c for c in cands if c.source_node_id == src and c.target_node_id == dst]
    assert match
    assert match[0].confidence == 1.0


def test_gin_on_source_vetoes_heuristic_same_name() -> None:
    users = Table(
        name="users",
        schema_name="public",
        row_count=50_000,
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="ref_code", data_type="text", column_cardinality=5),
        ],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        row_count=50_000,
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(
                name="ref_code",
                data_type="text",
                column_cardinality=5,
                index_type="gin",
            ),
        ],
    )
    snap = _snap(users, orders)
    cands = discover_heuristic_candidates(snap)
    from_orders = [
        c
        for c in cands
        if c.candidate_id.startswith("heuristic_same_name:")
        and "ref_code" in c.reasoning
        and c.source_node_id == table_node_id("demo", "public", "orders")
    ]
    assert from_orders == []


def test_gin_on_source_penalizes_heuristic_id() -> None:
    statuses = Table(
        name="statuses",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(
                name="status_id",
                data_type="int",
                index_type="gin",
                index_is_unique=False,
            ),
        ],
    )
    snap = _snap(statuses, orders)
    cands = discover_heuristic_candidates(snap)
    src = table_node_id("demo", "public", "orders")
    dst = table_node_id("demo", "public", "statuses")
    match = [c for c in cands if c.source_node_id == src and c.target_node_id == dst]
    assert match
    assert match[0].candidate_id.startswith("heuristic_id:")
    assert abs(match[0].confidence - 0.55) < 1e-6


def test_low_cardinality_large_table_penalizes_heuristic_id() -> None:
    statuses = Table(
        name="statuses",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        row_count=50_000,
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="status_id", data_type="int", column_cardinality=3),
        ],
    )
    snap = _snap(statuses, orders)
    cands = discover_heuristic_candidates(snap)
    src = table_node_id("demo", "public", "orders")
    dst = table_node_id("demo", "public", "statuses")
    match = [c for c in cands if c.source_node_id == src and c.target_node_id == dst]
    assert match
    assert match[0].candidate_id.startswith("heuristic_id:")
    assert abs(match[0].confidence - 0.65) < 1e-6


def test_high_null_penalty_on_source() -> None:
    customers = Table(
        name="customers",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="customer_id", data_type="int", null_percentage=90.0),
        ],
    )
    snap = _snap(customers, orders)
    cands = discover_heuristic_candidates(snap)
    src = table_node_id("demo", "public", "orders")
    dst = table_node_id("demo", "public", "customers")
    match = [c for c in cands if c.source_node_id == src and c.target_node_id == dst]
    assert match
    assert abs(match[0].confidence - 0.70) < 1e-6


def test_severe_penalties_clamp_to_zero() -> None:
    """Medium-confidence path: penalties exceed base and clamp to 0."""
    customers = Table(
        name="customers",
        schema_name="public",
        columns=[
            Column(
                name="id",
                data_type="int",
                is_primary_key=True,
                null_percentage=90.0,
            ),
        ],
    )
    orders = Table(
        name="orders",
        schema_name="public",
        row_count=50_000,
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(
                name="cust_id",
                data_type="int",
                column_cardinality=5,
                index_type="gin",
                null_percentage=90.0,
            ),
        ],
    )
    snap = _snap(customers, orders)
    cands = discover_heuristic_candidates(snap)
    src = table_node_id("demo", "public", "orders")
    dst = table_node_id("demo", "public", "customers")
    match = [c for c in cands if c.source_node_id == src and c.target_node_id == dst]
    assert match
    assert match[0].confidence == 0.0


def test_same_name_rule_filters_audit_columns_by_name() -> None:
    """Regression: ``ModifiedDate``-style audit columns must not emit
    same-name candidates regardless of how many tables they appear on.

    AdventureWorks puts ``ModifiedDate`` and ``rowguid`` on ~60 tables. Without
    a name-based filter, that's N*(N-1) ≈ 7,000 spurious directed candidates
    per column, which inflates adjacency in the join-path DFS and lets the
    traverse ranker prefer 1-hop ``ModifiedDate = ModifiedDate`` "joins" over
    real multi-hop FK chains. The fix lives in ``_BORING_SHARED_NAMES``: audit
    columns are filtered by name (case-insensitive), not by table count.
    """
    tables = [
        Table(
            name=f"t{i}",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="ModifiedDate", data_type="timestamp"),
                Column(name="rowguid", data_type="uuid"),
                Column(name="last_update", data_type="timestamp"),
            ],
        )
        for i in range(8)
    ]
    snap = _snap(*tables)
    cands = discover_heuristic_candidates(snap)
    same_name = [c for c in cands if c.candidate_id.startswith("heuristic_same_name:")]
    assert same_name == [], (
        f"expected no same-name candidates for audit columns, got {len(same_name)}"
    )


def test_same_name_rule_emits_for_small_shared_groups() -> None:
    """Sanity: small groups of legitimate non-audit shared columns emit candidates."""
    tables = [
        Table(
            name=f"t{i}",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="external_ref", data_type="text"),
            ],
        )
        for i in range(3)
    ]
    snap = _snap(*tables)
    cands = discover_heuristic_candidates(snap)
    same_name = [c for c in cands if c.candidate_id.startswith("heuristic_same_name:")]
    # 3 tables → 3*2 = 6 directed candidates for one shared column.
    assert len(same_name) == 6


def test_expanded_deny_list_filters_generic_columns() -> None:
    """Generic descriptor columns (status, type, name, etc.) must not emit
    same-name candidates — they are near-zero selectivity join keys."""
    tables = [
        Table(
            name=f"t{i}",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="status", data_type="varchar"),
                Column(name="type", data_type="varchar"),
                Column(name="name", data_type="varchar"),
                Column(name="category", data_type="varchar"),
            ],
        )
        for i in range(4)
    ]
    snap = _snap(*tables)
    cands = discover_heuristic_candidates(snap)
    same_name = [c for c in cands if c.candidate_id.startswith("heuristic_same_name:")]
    assert same_name == [], (
        f"expected no same-name candidates for generic columns, got {len(same_name)}"
    )


def test_deny_list_suppresses_dimensional_attributes() -> None:
    """`rating` and `length` are dimensional attributes: high cardinality but
    no cross-table join meaning. Film.rating = category.rating would be a
    nonsensical inferred join.
    """
    tables = [
        Table(
            name=f"t{i}",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="rating", data_type="varchar"),
                Column(name="length", data_type="smallint"),
            ],
        )
        for i in range(3)
    ]
    snap = _snap(*tables)
    cands = discover_heuristic_candidates(snap)
    same_name = [c for c in cands if c.candidate_id.startswith("heuristic_same_name:")]
    assert same_name == [], (
        f"expected no same-name candidates for rating/length, got {len(same_name)}"
    )


def test_deny_list_suppresses_generic_numerics() -> None:
    """Generic numeric columns (amount, count, quantity, price, ...) must not
    emit same-name candidates. The type-family veto (numeric ↔ numeric) cannot
    rule these out on its own; the name-based deny-list is the only stop.
    """
    numeric_cols = [
        "amount",
        "count",
        "quantity",
        "qty",
        "total",
        "price",
        "weight",
        "height",
        "width",
        "score",
        "rank",
        "position",
        "order",
        "level",
    ]
    tables = [
        Table(
            name=f"t{i}",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                *(Column(name=n, data_type="numeric") for n in numeric_cols),
            ],
        )
        for i in range(3)
    ]
    snap = _snap(*tables)
    cands = discover_heuristic_candidates(snap)
    same_name = [c for c in cands if c.candidate_id.startswith("heuristic_same_name:")]
    assert same_name == [], (
        f"expected no same-name candidates for generic numerics, "
        f"got {len(same_name)}: {[c.candidate_id for c in same_name]}"
    )


def test_type_mismatch_vetoes_same_name() -> None:
    """Same column name with incompatible types (varchar vs int) must be vetoed."""
    t1 = Table(
        name="products",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="code", data_type="varchar"),
        ],
    )
    t2 = Table(
        name="warehouses",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="code", data_type="int"),
        ],
    )
    snap = _snap(t1, t2)
    cands = discover_heuristic_candidates(snap)
    same_name = [
        c
        for c in cands
        if c.candidate_id.startswith("heuristic_same_name:")
        and "code" in c.reasoning
    ]
    assert same_name == [], "type-mismatched same-name candidates should be vetoed"


def test_type_match_preserves_same_name() -> None:
    """Same column name with compatible types must preserve candidates."""
    t1 = Table(
        name="products",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="sku", data_type="varchar"),
        ],
    )
    t2 = Table(
        name="warehouses",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="sku", data_type="text"),
        ],
    )
    snap = _snap(t1, t2)
    cands = discover_heuristic_candidates(snap)
    same_name = [
        c
        for c in cands
        if c.candidate_id.startswith("heuristic_same_name:")
        and "sku" in c.reasoning
    ]
    assert len(same_name) == 2, "type-compatible same-name candidates should be kept"


def test_unknown_type_no_veto() -> None:
    """Unknown types (e.g. jsonb) must not trigger a veto."""
    t1 = Table(
        name="events",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="payload", data_type="jsonb"),
        ],
    )
    t2 = Table(
        name="logs",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="payload", data_type="text"),
        ],
    )
    snap = _snap(t1, t2)
    cands = discover_heuristic_candidates(snap)
    same_name = [
        c
        for c in cands
        if c.candidate_id.startswith("heuristic_same_name:")
        and "payload" in c.reasoning
    ]
    assert len(same_name) == 2, "unknown type should not veto"


def test_type_mismatch_vetoes_fk_heuristic() -> None:
    """``*_fk`` candidates with mismatched types must be vetoed (regression: AW productreview.productid → emailaddress.emailaddress)."""
    t1 = Table(
        name="review",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            # review.product_fk is a varchar — e.g. external vendor code
            Column(name="product_fk", data_type="varchar"),
        ],
    )
    t2 = Table(
        name="product",
        schema_name="public",
        columns=[
            # product.id is int; joining varchar→int would be garbage.
            Column(name="id", data_type="int", is_primary_key=True),
        ],
    )
    snap = _snap(t1, t2)
    cands = discover_heuristic_candidates(snap)
    fk = [c for c in cands if c.candidate_id.startswith("heuristic_fk:")]
    assert fk == [], "cross-family _fk candidates should be vetoed"


def test_type_mismatch_vetoes_id_heuristic() -> None:
    """``*_id`` candidates with mismatched types must be vetoed."""
    t1 = Table(
        name="review",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="product_id", data_type="varchar"),
        ],
    )
    t2 = Table(
        name="product",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    snap = _snap(t1, t2)
    cands = discover_heuristic_candidates(snap)
    fk_like = [
        c
        for c in cands
        if c.candidate_id.startswith("heuristic_id:")
        or c.candidate_id.startswith("heuristic_fk:")
    ]
    assert fk_like == [], "cross-family _id candidates should be vetoed"


def test_parameterized_type_normalized() -> None:
    """Parameterized types like varchar(255) must normalize to the base type."""
    t1 = Table(
        name="products",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="sku", data_type="varchar(255)"),
        ],
    )
    t2 = Table(
        name="warehouses",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="sku", data_type="text"),
        ],
    )
    snap = _snap(t1, t2)
    cands = discover_heuristic_candidates(snap)
    same_name = [
        c
        for c in cands
        if c.candidate_id.startswith("heuristic_same_name:")
        and "sku" in c.reasoning
    ]
    assert len(same_name) == 2, "parameterized types should normalize correctly"


def test_deny_list_preserves_legitimate_same_name_columns() -> None:
    """Non-generic shared columns like region_code must still emit candidates."""
    t1 = Table(
        name="orders",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="region_code", data_type="varchar"),
        ],
    )
    t2 = Table(
        name="customers",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="region_code", data_type="varchar"),
        ],
    )
    snap = _snap(t1, t2)
    cands = discover_heuristic_candidates(snap)
    same_name = [
        c
        for c in cands
        if c.candidate_id.startswith("heuristic_same_name:")
        and "region_code" in c.reasoning
    ]
    assert len(same_name) == 2, "expected bidirectional same-name candidates"


def test_per_pair_cap_limits_candidates() -> None:
    """Two tables sharing 5 columns must emit at most 3 per direction (6 total)."""
    t1 = Table(
        name="alpha",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="col_a", data_type="text"),
            Column(name="col_b", data_type="text"),
            Column(name="col_c", data_type="text"),
            Column(name="col_d", data_type="text"),
            Column(name="col_e", data_type="text"),
        ],
    )
    t2 = Table(
        name="beta",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="col_a", data_type="text"),
            Column(name="col_b", data_type="text"),
            Column(name="col_c", data_type="text"),
            Column(name="col_d", data_type="text"),
            Column(name="col_e", data_type="text"),
        ],
    )
    snap = _snap(t1, t2)
    cands = discover_heuristic_candidates(snap)
    same_name = [c for c in cands if c.candidate_id.startswith("heuristic_same_name:")]
    assert len(same_name) == 6, f"expected 6 (3 per direction), got {len(same_name)}"


def test_per_pair_cap_keeps_most_selective() -> None:
    """The retained candidates should be the most selective (fewest tables)."""
    # rare_col on 2 tables (selectivity=0.5), common_col_{1-4} on 3 tables (0.33)
    t1 = Table(
        name="alpha",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="rare_col", data_type="text"),
            Column(name="common_col_1", data_type="text"),
            Column(name="common_col_2", data_type="text"),
            Column(name="common_col_3", data_type="text"),
            Column(name="common_col_4", data_type="text"),
        ],
    )
    t2 = Table(
        name="beta",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="rare_col", data_type="text"),
            Column(name="common_col_1", data_type="text"),
            Column(name="common_col_2", data_type="text"),
            Column(name="common_col_3", data_type="text"),
            Column(name="common_col_4", data_type="text"),
        ],
    )
    t3 = Table(
        name="gamma",
        schema_name="public",
        columns=[
            Column(name="id", data_type="int", is_primary_key=True),
            Column(name="common_col_1", data_type="text"),
            Column(name="common_col_2", data_type="text"),
            Column(name="common_col_3", data_type="text"),
            Column(name="common_col_4", data_type="text"),
        ],
    )
    snap = _snap(t1, t2, t3)
    cands = discover_heuristic_candidates(snap)
    # For the (alpha, beta) pair, rare_col (selectivity 0.5) should be retained.
    alpha_beta = [
        c
        for c in cands
        if c.candidate_id.startswith("heuristic_same_name:")
        and c.source_node_id == table_node_id("demo", "public", "alpha")
        and c.target_node_id == table_node_id("demo", "public", "beta")
    ]
    assert len(alpha_beta) == 3, f"expected 3 per direction, got {len(alpha_beta)}"
    col_names = {c.source_column for c in alpha_beta}
    assert "rare_col" in col_names, "most selective column should be retained"


def test_per_pair_cap_single_column_preserves_all() -> None:
    """When each pair has <=3 shared columns, all candidates are preserved."""
    tables = [
        Table(
            name=f"t{i}",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="ref_code", data_type="text"),
            ],
        )
        for i in range(4)
    ]
    snap = _snap(*tables)
    cands = discover_heuristic_candidates(snap)
    same_name = [c for c in cands if c.candidate_id.startswith("heuristic_same_name:")]
    # 4 tables, 1 shared column → 4*3 = 12 directed candidates, all under cap
    assert len(same_name) == 12, f"expected 12, got {len(same_name)}"


def test_same_name_rule_emits_for_widely_shared_non_audit_columns() -> None:
    """Wide non-audit shared columns must still emit candidates.

    Removing the prior table-count cap means a denormalized column
    like ``region_code`` appearing on many tables must produce same-name
    candidates — only audit-named columns are now suppressed. This guards
    against re-introducing a blunt count cap as a "perf fix": the perf safety
    net belongs in the join-path DFS budget, not in candidate generation.
    """
    tables = [
        Table(
            name=f"t{i}",
            schema_name="public",
            columns=[
                Column(name="id", data_type="int", is_primary_key=True),
                Column(name="region_code", data_type="text"),
            ],
        )
        for i in range(10)
    ]
    snap = _snap(*tables)
    cands = discover_heuristic_candidates(snap)
    same_name = [c for c in cands if c.candidate_id.startswith("heuristic_same_name:")]
    # 10 tables → 10*9 = 90 directed candidates for one shared column.
    assert len(same_name) == 90
