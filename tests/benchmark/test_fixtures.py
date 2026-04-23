"""Tests for ``pretensor.benchmark.fixtures.load_dataset``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pretensor.benchmark import Dataset, Fixture, load_dataset
from pretensor.connectors.models import SchemaSnapshot

_QUESTION_KEYS = {"id", "question", "expected_sql"}


def test_load_pagila_has_all_paths() -> None:
    fx = load_dataset("pagila")
    assert isinstance(fx, Fixture)
    assert fx.name is Dataset.PAGILA
    assert fx.schema_yaml_path.exists()
    assert fx.ddl_sql_path is not None and fx.ddl_sql_path.exists()
    assert fx.questions_path is not None and fx.questions_path.exists()

    # Schema YAML round-trips.
    snap = SchemaSnapshot.from_yaml(fx.schema_yaml_path.read_text())
    assert snap.connection_name == "pagila"
    assert snap.tables, "pagila snapshot has no tables"

    # DDL is CREATE TABLE-shaped.
    assert "CREATE TABLE" in fx.ddl_sql_path.read_text()


def test_load_adventureworks_has_all_paths() -> None:
    fx = load_dataset("adventureworks")
    assert fx.name is Dataset.ADVENTUREWORKS
    assert fx.schema_yaml_path.exists()
    assert fx.ddl_sql_path is not None and fx.ddl_sql_path.exists()
    assert fx.questions_path is not None and fx.questions_path.exists()

    snap = SchemaSnapshot.from_yaml(fx.schema_yaml_path.read_text())
    assert snap.connection_name == "adventureworks"
    # AdventureWorks has five business schemas + dozens of tables.
    assert set(snap.schemas) >= {
        "person",
        "humanresources",
        "production",
        "purchasing",
        "sales",
    }

    questions = json.loads(fx.questions_path.read_text())
    assert len(questions) >= 20
    for q in questions:
        assert set(q.keys()) == _QUESTION_KEYS


def test_load_tpch_has_all_paths() -> None:
    fx = load_dataset("tpch")
    assert fx.name is Dataset.TPCH
    assert fx.ddl_sql_path is not None and fx.ddl_sql_path.exists()
    assert fx.questions_path is not None and fx.questions_path.exists()

    snap = SchemaSnapshot.from_yaml(fx.schema_yaml_path.read_text())
    assert snap.connection_name == "tpch"

    ddl = fx.ddl_sql_path.read_text()
    assert "CREATE TABLE" in ddl
    # Pure-DDL invariant: no data-loading statements snuck in.
    assert "INSERT" not in ddl.upper()

    questions = json.loads(fx.questions_path.read_text())
    assert len(questions) >= 20
    for q in questions:
        assert set(q.keys()) == _QUESTION_KEYS


def test_load_dataset_accepts_enum_and_string() -> None:
    from_enum = load_dataset(Dataset.PAGILA)
    from_str = load_dataset("pagila")
    assert from_enum == from_str


def test_load_dataset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark dataset"):
        load_dataset("not_a_real_dataset")


@pytest.mark.parametrize("name", ["adversarial", "analytics_dwh", "saas_multitenant"])
def test_datasets_without_ddl_or_questions_return_none(name: str) -> None:
    # ``schema_yaml_path.exists()`` would be trivially true here — ``load_dataset``
    # raises ``FileNotFoundError`` before returning when the YAML is absent — so
    # only the nullable-optional fields are worth asserting on this path.
    fx = load_dataset(name)
    assert fx.ddl_sql_path is None
    assert fx.questions_path is None


def test_load_dataset_raises_when_schema_yaml_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Redirect the module's schema-dir resolver to a non-existent path so
    # ``load_dataset`` hits the missing-YAML branch for a known-good key.
    from pretensor.benchmark import fixtures as fixtures_mod

    monkeypatch.setattr(fixtures_mod, "_SCHEMAS_DIR", tmp_path / "no-such-dir")
    with pytest.raises(FileNotFoundError, match="Schema YAML missing"):
        load_dataset(Dataset.PAGILA)


def test_fixture_is_immutable() -> None:
    # ``@dataclass(frozen=True, slots=True)`` raises ``FrozenInstanceError``
    # (an ``AttributeError`` subclass) on any attribute reassignment.
    fx = load_dataset("pagila")
    with pytest.raises(AttributeError):
        fx.name = Dataset.TPCH  # type: ignore[misc]
