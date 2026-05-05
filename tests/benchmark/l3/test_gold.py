"""Unit tests for the L3 gold-question loader.

Reads the real bundled Pagila / TPC-H / AdventureWorks question files
to confirm their on-disk shape matches the L3 contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pretensor.benchmark.fixtures import Fixture, load_dataset
from pretensor.benchmark.l3.gold import L3GoldEntry, load_l3_gold
from pretensor.benchmark.runner import Dataset


@pytest.mark.parametrize(
    "dataset",
    [Dataset.PAGILA, Dataset.TPCH, Dataset.ADVENTUREWORKS],
    ids=lambda d: d.value,
)
def test_load_l3_gold_returns_sorted_entries_for_real_fixture(
    dataset: Dataset,
) -> None:
    fixture = load_dataset(dataset)
    questions_path, raw_bytes, entries = load_l3_gold(fixture)
    assert entries, f"expected at least one question for {dataset.value}"
    ids = [entry.id for entry in entries]
    assert ids == sorted(ids)
    # The returned path must point at an existing file; this is the
    # contract callers (e.g. the runner's fixture_sha computation) rely on.
    assert questions_path.exists()
    assert questions_path == fixture.questions_path
    # raw_bytes is exactly the file's contents — callers fingerprint
    # the fixture from these bytes without a second I/O.
    assert raw_bytes == questions_path.read_bytes()


def test_load_l3_gold_carries_question_and_expected_sql() -> None:
    fixture = load_dataset(Dataset.PAGILA)
    _path, _bytes, entries = load_l3_gold(fixture)
    sample = entries[0]
    assert isinstance(sample, L3GoldEntry)
    assert sample.id
    assert sample.question.strip()
    assert sample.expected_sql.strip().upper().startswith(("SELECT", "WITH"))


def test_load_l3_gold_raises_when_questions_path_missing(tmp_path: Path) -> None:
    fixture = Fixture(
        name=Dataset.PAGILA,
        schema_yaml_path=tmp_path / "schema.yaml",
        ddl_sql_path=None,
        questions_path=None,
        metric_templates_path=None,
    )
    with pytest.raises(FileNotFoundError, match="no NL-to-SQL gold"):
        load_l3_gold(fixture)


def test_load_l3_gold_raises_on_non_array_root(tmp_path: Path) -> None:
    questions = tmp_path / "q.json"
    questions.write_text("{}")
    fixture = Fixture(
        name=Dataset.PAGILA,
        schema_yaml_path=tmp_path / "schema.yaml",
        ddl_sql_path=None,
        questions_path=questions,
        metric_templates_path=None,
    )
    with pytest.raises(ValueError, match="JSON array"):
        load_l3_gold(fixture)


def test_load_l3_gold_raises_on_missing_required_field(tmp_path: Path) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([{"id": "x", "question": "?"}]))  # no expected_sql
    fixture = Fixture(
        name=Dataset.PAGILA,
        schema_yaml_path=tmp_path / "schema.yaml",
        ddl_sql_path=None,
        questions_path=questions,
        metric_templates_path=None,
    )
    with pytest.raises(ValueError, match="expected_sql"):
        load_l3_gold(fixture)


def test_load_l3_gold_raises_on_record_not_object(tmp_path: Path) -> None:
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps(["not-an-object"]))
    fixture = Fixture(
        name=Dataset.PAGILA,
        schema_yaml_path=tmp_path / "schema.yaml",
        ddl_sql_path=None,
        questions_path=questions,
        metric_templates_path=None,
    )
    with pytest.raises(ValueError, match="expected an object"):
        load_l3_gold(fixture)
