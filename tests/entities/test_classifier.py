"""Tests for :mod:`pretensor.entities.classifier`."""

from __future__ import annotations

import pytest

from pretensor.entities.classifier import (
    TableClassification,
    TableClassifier,
    TableClassifierInput,
)

_CLASSIFIER = TableClassifier()


@pytest.mark.parametrize(
    ("name", "columns", "expected_role"),
    [
        ("user_roles", ["user_id", "role_id"], "junction"),
        ("stg_raw_events", [], "staging"),
        ("orders_audit", ["id", "old_value"], "audit"),
        ("dim_customer", ["customer_id", "name"], "dimension"),
        ("products", ["id", "name"], "entity_candidate"),
        ("migrations", ["version"], "system"),
        ("schema_migrations", ["version"], "system"),
        ("ar_internal_metadata", ["key"], "system"),
        ("pg_stat_statements", ["query"], "system"),
        ("staging_load", [], "staging"),
    ],
)
def test_classifier_expected_role(
    name: str, columns: list[str], expected_role: str
) -> None:
    t = TableClassifierInput(name=name, schema_name="public", columns=columns)
    r = _CLASSIFIER.classify(t)
    assert isinstance(r, TableClassification)
    assert r.role == expected_role


def test_fk_fan_out_boosts_fact() -> None:
    inp = TableClassifierInput(
        name="sales_lineitems",
        schema_name="public",
        columns=["id", "order_id", "product_id", "customer_id", "amount"],
        fk_out_degree=4,
        fk_in_degree=0,
        row_count=2_000_000,
    )
    r = _CLASSIFIER.classify(inp)
    assert r.role == "fact"
    assert r.confidence >= 0.5


def test_seq_idx_scan_ratio_boosts_large_table_fact_staging_signal() -> None:
    """seq/(seq+idx) > 0.9 on row_count > 10k adds fact/staging score."""
    base = TableClassifierInput(
        name="wide_scan_table",
        schema_name="public",
        columns=["id", "a", "b", "c"],
        fk_out_degree=3,
        fk_in_degree=0,
        row_count=20_000,
        seq_scan_count=950,
    )
    with_idx = base.model_copy(update={"idx_scan_count": 50})
    r_base = _CLASSIFIER.classify(base)
    r_idx = _CLASSIFIER.classify(with_idx)
    assert r_idx.confidence >= r_base.confidence
    assert any("seq vs idx" in s for s in r_idx.signals)
