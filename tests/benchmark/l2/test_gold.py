"""Tests for the L2 gold-data loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pretensor.benchmark import Dataset, Fixture
from pretensor.benchmark.l2.gold import (
    QueryGoldEntry,
    TraverseGoldEntry,
    load_metric_templates,
    load_query_gold,
    load_traverse_gold,
)


def _fixture(
    *,
    name: Dataset,
    questions_path: Path | None,
    templates_path: Path | None = None,
) -> Fixture:
    return Fixture(
        name=name,
        # ``schema_yaml_path`` is unused by the gold loaders but the
        # dataclass requires a value — point at the questions file as a
        # harmless stand-in.
        schema_yaml_path=questions_path or Path("/dev/null"),
        ddl_sql_path=None,
        questions_path=questions_path,
        metric_templates_path=templates_path,
    )


# ---------------------------------------------------------------------------
# load_query_gold
# ---------------------------------------------------------------------------


def test_load_query_gold_returns_empty_when_no_questions() -> None:
    fx = _fixture(name=Dataset.PAGILA, questions_path=None)
    assert load_query_gold(fx) == []


def test_load_query_gold_derives_tables_via_sqlglot(tmp_path: Path) -> None:
    qpath = tmp_path / "pagila_nl2sql_bench.json"
    qpath.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Films and their categories.",
                    "expected_sql": (
                        "SELECT f.title FROM public.film AS f "
                        "JOIN public.film_category AS fc ON f.film_id = fc.film_id"
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    fx = _fixture(name=Dataset.PAGILA, questions_path=qpath)
    out = load_query_gold(fx)
    assert len(out) == 1
    entry = out[0]
    assert isinstance(entry, QueryGoldEntry)
    assert entry.id == "q1"
    assert entry.tables_touched == frozenset({"film", "film_category"})


def test_load_query_gold_uses_explicit_tables_touched_when_provided(
    tmp_path: Path,
) -> None:
    qpath = tmp_path / "x_nl2sql_bench.json"
    qpath.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Q.",
                    # SQL would parse to {a, b}, but the explicit list overrides.
                    "expected_sql": "SELECT * FROM public.a JOIN public.b ON true",
                    "tables_touched": ["override_one", "override_two"],
                }
            ]
        ),
        encoding="utf-8",
    )
    fx = _fixture(name=Dataset.PAGILA, questions_path=qpath)
    out = load_query_gold(fx)
    assert out[0].tables_touched == frozenset({"override_one", "override_two"})


def test_load_query_gold_strips_schema_prefix_in_explicit_tables(
    tmp_path: Path,
) -> None:
    qpath = tmp_path / "x_nl2sql_bench.json"
    qpath.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Q.",
                    "expected_sql": "SELECT 1",
                    "tables_touched": ["public.actor", "actor"],
                }
            ]
        ),
        encoding="utf-8",
    )
    fx = _fixture(name=Dataset.PAGILA, questions_path=qpath)
    # Both inputs collapse to ``actor``.
    assert load_query_gold(fx)[0].tables_touched == frozenset({"actor"})


def test_load_query_gold_handles_unparseable_sql(tmp_path: Path) -> None:
    qpath = tmp_path / "x_nl2sql_bench.json"
    qpath.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Q.",
                    "expected_sql": "this is not sql at all",
                }
            ]
        ),
        encoding="utf-8",
    )
    fx = _fixture(name=Dataset.PAGILA, questions_path=qpath)
    out = load_query_gold(fx)
    # Parse failure → empty tables; runner skips empty-gold observations.
    assert out[0].tables_touched == frozenset()


# ---------------------------------------------------------------------------
# load_traverse_gold
# ---------------------------------------------------------------------------


def test_load_traverse_gold_filters_entries_without_gold_path(
    tmp_path: Path,
) -> None:
    qpath = tmp_path / "x_nl2sql_bench.json"
    qpath.write_text(
        json.dumps(
            [
                {"id": "q1", "question": "Q.", "expected_sql": "SELECT 1"},
                {
                    "id": "q2",
                    "question": "Q.",
                    "expected_sql": "SELECT 1",
                    "gold_path": [
                        {"from_table": "public.actor", "to_table": "public.film_actor"},
                        {"from_table": "public.film_actor", "to_table": "public.film"},
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    fx = _fixture(name=Dataset.PAGILA, questions_path=qpath)
    out = load_traverse_gold(fx)
    assert len(out) == 1
    entry = out[0]
    assert isinstance(entry, TraverseGoldEntry)
    assert entry.id == "q2"
    assert entry.from_table == "public.actor"
    assert entry.to_table == "public.film"
    assert entry.gold_path == (
        ("public.actor", "public.film_actor"),
        ("public.film_actor", "public.film"),
    )


def test_load_traverse_gold_skips_malformed_entries(tmp_path: Path) -> None:
    qpath = tmp_path / "x_nl2sql_bench.json"
    qpath.write_text(
        json.dumps(
            [
                # gold_path is not a list
                {
                    "id": "bad1",
                    "question": "Q.",
                    "expected_sql": "SELECT 1",
                    "gold_path": "not a list",
                },
                # gold_path is empty
                {
                    "id": "bad2",
                    "question": "Q.",
                    "expected_sql": "SELECT 1",
                    "gold_path": [],
                },
                # gold_path hop missing required key
                {
                    "id": "bad3",
                    "question": "Q.",
                    "expected_sql": "SELECT 1",
                    "gold_path": [{"from_table": "a"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    fx = _fixture(name=Dataset.PAGILA, questions_path=qpath)
    assert load_traverse_gold(fx) == []


# ---------------------------------------------------------------------------
# load_metric_templates
# ---------------------------------------------------------------------------


def test_load_metric_templates_returns_empty_when_path_absent() -> None:
    fx = _fixture(name=Dataset.PAGILA, questions_path=None, templates_path=None)
    assert load_metric_templates(fx) == []


def test_load_metric_templates_parses_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "pagila_metric_templates.yaml"
    yaml_path.write_text(
        """
connection_name: pagila
templates:
  - metric: total_rentals
    notes: Count of rentals
    semantic_yaml: |
      connection_name: pagila
      domains:
        - name: rentals
          entities:
            - name: rental
              source_table: public.rental
              attributes:
                - name: rental_id
                  source_column: rental_id
                  role: identifier
              metrics:
                - name: total_rentals
                  type: count
                  field: rental_id
""",
        encoding="utf-8",
    )
    fx = _fixture(name=Dataset.PAGILA, questions_path=None, templates_path=yaml_path)
    out = load_metric_templates(fx)
    assert len(out) == 1
    tpl = out[0]
    assert tpl.metric == "total_rentals"
    assert tpl.database == "pagila"
    assert "connection_name: pagila" in tpl.semantic_yaml


def test_load_metric_templates_skips_invalid_entries(tmp_path: Path) -> None:
    yaml_path = tmp_path / "x_metric_templates.yaml"
    yaml_path.write_text(
        """
connection_name: x
templates:
  - metric: ""
    semantic_yaml: |
      connection_name: x
  - metric: ok
    semantic_yaml: ""
""",
        encoding="utf-8",
    )
    fx = _fixture(name=Dataset.PAGILA, questions_path=None, templates_path=yaml_path)
    assert load_metric_templates(fx) == []


def test_load_metric_templates_falls_back_to_dataset_name(tmp_path: Path) -> None:
    yaml_path = tmp_path / "x_metric_templates.yaml"
    yaml_path.write_text(
        """
templates:
  - metric: m
    semantic_yaml: "x"
""",
        encoding="utf-8",
    )
    fx = _fixture(name=Dataset.PAGILA, questions_path=None, templates_path=yaml_path)
    out = load_metric_templates(fx)
    assert len(out) == 1
    # No top-level connection_name → fall back to the dataset name.
    assert out[0].database == "pagila"


@pytest.mark.parametrize(
    "raw",
    [
        "not_a_dict",  # YAML scalar
        "[1, 2, 3]",  # YAML list
    ],
)
def test_load_metric_templates_handles_non_mapping_root(
    tmp_path: Path, raw: str
) -> None:
    yaml_path = tmp_path / "x_metric_templates.yaml"
    yaml_path.write_text(raw, encoding="utf-8")
    fx = _fixture(name=Dataset.PAGILA, questions_path=None, templates_path=yaml_path)
    assert load_metric_templates(fx) == []
