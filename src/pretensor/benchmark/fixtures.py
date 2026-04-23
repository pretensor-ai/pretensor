"""Benchmark-fixture loader.

Every L1 / L2 / L3 runner resolves datasets through :func:`load_dataset`, so
the set of supported keys and the physical file layout lives in one place.
Fixtures are resolved relative to the repo root (this module's grandparent of
``src``) — callers never need to know where the files live.

The returned :class:`Fixture` names the dataset, points to its schema-snapshot
YAML (always present), and optionally to a DDL-only ``schema.sql`` and a gold
NL-to-SQL question set. Datasets without DDL or a question set (``adversarial``,
``analytics_dwh``, ``saas_multitenant``) return ``None`` for those fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pretensor.benchmark.runner import Dataset

__all__ = ["Fixture", "load_dataset"]

# Walk up from src/pretensor/benchmark/fixtures.py to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMAS_DIR = _REPO_ROOT / "tests" / "fixtures" / "schemas"
_DATA_DIR = _REPO_ROOT / "scripts" / "data"


@dataclass(frozen=True, slots=True)
class Fixture:
    """One benchmark dataset's on-disk layout.

    ``name`` is the canonical :class:`Dataset` enum key. ``schema_yaml_path``
    is always set and points to a file that :meth:`SchemaSnapshot.from_yaml`
    can parse. The other two fields are ``None`` for datasets that have no
    DDL dump or no gold question set checked in.
    """

    name: Dataset
    schema_yaml_path: Path
    ddl_sql_path: Path | None
    questions_path: Path | None


def load_dataset(name: str | Dataset) -> Fixture:
    """Resolve a benchmark dataset's fixture paths.

    Accepts a :class:`Dataset` enum value or its string form. Unknown keys
    raise :class:`ValueError`. The schema YAML must exist — if missing,
    :class:`FileNotFoundError` is raised. DDL and question-set paths are
    returned as ``None`` when the dataset has no such file on disk.
    """
    if isinstance(name, Dataset):
        key = name
    else:
        try:
            key = Dataset(name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown benchmark dataset: {name!r}. "
                f"Known: {[d.value for d in Dataset]}"
            ) from exc

    schema_path = _SCHEMAS_DIR / f"{key.value}.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema YAML missing for dataset {key.value!r}: {schema_path}"
        )

    ddl_path = _DATA_DIR / key.value / "schema.sql"
    questions_path = _DATA_DIR / f"{key.value}_nl2sql_bench.json"

    return Fixture(
        name=key,
        schema_yaml_path=schema_path,
        ddl_sql_path=ddl_path if ddl_path.exists() else None,
        questions_path=questions_path if questions_path.exists() else None,
    )
