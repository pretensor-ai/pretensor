"""Pytest fixtures for pretensor."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Editable install (`pip install -e .`) adds `pretensor`; for bare pytest, put `src` on path.
_ROOT = Path(__file__).resolve().parents[1]
_src = _ROOT / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from pretensor.connectors.models import SchemaSnapshot, Table  # noqa: E402
from pretensor.core.store import KuzuStore  # noqa: E402

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── Snapshot helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def make_snapshot():
    """Return a factory that builds a :class:`SchemaSnapshot` from a table list."""

    def _factory(
        tables: list[Table],
        *,
        connection_name: str = "demo",
        database: str = "db",
        schemas: list[str] | None = None,
    ) -> SchemaSnapshot:
        return SchemaSnapshot(
            connection_name=connection_name,
            database=database,
            schemas=schemas or ["public"],
            tables=tables,
            introspected_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    return _factory


@pytest.fixture
def load_schema():
    """Load a named YAML schema fixture from ``tests/fixtures/schemas/``."""
    schemas_dir = _FIXTURES_DIR / "schemas"

    def _load(name: str) -> SchemaSnapshot:
        return SchemaSnapshot.from_yaml((schemas_dir / f"{name}.yaml").read_text())

    return _load


# ── KuzuStore fixture ─────────────────────────────────────────────────────────


@pytest.fixture
def graph_store(tmp_path: Path):
    """Yield an auto-closing :class:`KuzuStore` with schema ensured."""
    store = KuzuStore(tmp_path / "test.kuzu")
    store.ensure_schema()
    try:
        yield store
    finally:
        store.close()
