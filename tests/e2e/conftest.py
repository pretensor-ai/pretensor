"""Session-scoped E2E fixtures: Postgres container → DDL → full index pipeline."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

# Skip guard MUST come before any optional-extra imports so that normal CI
# (which installs only --extra dev, not --extra e2e) can collect tests/
# without hitting a ModuleNotFoundError for testcontainers.
if not os.getenv("PRETENSOR_E2E"):
    pytest.skip("set PRETENSOR_E2E=1", allow_module_level=True)

import subprocess  # noqa: E402

import psycopg2  # noqa: E402
from testcontainers.postgres import PostgresContainer  # noqa: E402

from pretensor.cli.constants import REGISTRY_FILENAME  # noqa: E402
from pretensor.connectors.inspect import inspect  # noqa: E402
from pretensor.core.builder import GraphBuilder  # noqa: E402
from pretensor.core.registry import GraphRegistry  # noqa: E402
from pretensor.core.store import KuzuStore  # noqa: E402
from pretensor.introspection.models.dsn import connection_config_from_url  # noqa: E402
from pretensor.staleness.snapshot_store import SnapshotStore  # noqa: E402
from tests.fixtures.dbt.synthetic_warehouse import (  # noqa: E402
    SyntheticManifestSpec,
    write_synthetic_manifest,
)

_FIXTURE_SQL_DIR = Path(__file__).parent / "fixtures" / "sql"
_PAGILA_DDL_PATH = _FIXTURE_SQL_DIR / "pagila_ddl.sql"
_ADVENTUREWORKS_DDL_PATH = _FIXTURE_SQL_DIR / "adventureworks_ddl.sql"
_TPCDS_DDL_PATH = _FIXTURE_SQL_DIR / "tpcds_ddl.sql"
_TPCH_DDL_PATH = _FIXTURE_SQL_DIR / "tpch_ddl.sql"
_POSTGRES_IMAGE = "postgres:16-alpine"


def _ensure_image(image: str) -> None:
    """Pull ``image`` if it is not already in the local Docker cache.

    Uses the Docker CLI directly rather than docker-py so that the check and
    pull are not affected by docker-py exception-wrapping differences across
    versions (some raise ImageNotFound, others raise raw HTTPError for 404).
    """
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(["docker", "pull", image], check=True)


def _start_postgres_with_ddl(ddl_path: Path) -> Iterator[str]:
    """Boot a session-scoped Postgres container, apply ``ddl_path``, yield DSN."""
    _ensure_image(_POSTGRES_IMAGE)
    with PostgresContainer(_POSTGRES_IMAGE) as pg:
        # testcontainers v4 returns postgresql+psycopg2://... — strip the driver suffix
        raw_url: str = pg.get_connection_url()
        dsn = raw_url.replace("postgresql+psycopg2://", "postgresql://")

        # Apply schema DDL
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(ddl_path.read_text(encoding="utf-8"))
        cur.close()
        conn.close()

        yield dsn


def _index_database(
    dsn: str,
    connection_name: str,
    state_dir: Path,
    *,
    run_relationship_discovery: bool = True,
) -> Path:
    """Run the full introspect → snapshot → build → register pipeline.

    Mirrors the steps the production ``pretensor index`` CLI performs, kept
    in-process so the e2e fixtures don't shell out. ``state_dir`` is mutated
    (graphs/, snapshot store, registry.json) and returned for caller convenience.

    ``run_relationship_discovery`` defaults to ``True`` so existing fixtures
    (pagila, AdventureWorks) keep their behaviour. The synthetic-warehouse
    fixture for the dbt enrichment test passes ``False`` because (a) the
    warehouse has no real FK constraints, (b) heuristic discovery on 1,500
    bare ``id`` columns would be slow and produce noise unrelated to the dbt
    enrichment under test.
    """
    # 1. Introspect live schema
    cfg = connection_config_from_url(dsn, connection_name)
    snapshot = inspect(cfg)

    # 2. Persist snapshot for drift detection
    SnapshotStore(state_dir).save(connection_name, snapshot)

    # 3. Build Kuzu graph
    graph_path = state_dir / "graphs" / f"{connection_name}.kuzu"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph_path)
    GraphBuilder().build(
        snapshot, store, run_relationship_discovery=run_relationship_discovery
    )
    store.close()

    # 4. Register connection in registry.json
    registry = GraphRegistry(state_dir / REGISTRY_FILENAME).load()
    registry.upsert(
        connection_name=connection_name,
        database=snapshot.database,
        dsn=dsn,
        graph_path=graph_path,
        table_count=len(snapshot.tables),
    )
    registry.save()

    return state_dir


@pytest.fixture(scope="session")
def pagila_dsn(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Start a Postgres container, apply pagila DDL, yield the DSN, stop on teardown."""
    yield from _start_postgres_with_ddl(_PAGILA_DDL_PATH)


@pytest.fixture(scope="session")
def indexed_state(
    pagila_dsn: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run the full pretensor index pipeline; return the state_dir path (read-only)."""
    return _index_database(
        pagila_dsn,
        "pagila",
        tmp_path_factory.mktemp("e2e_state"),
    )


@pytest.fixture(scope="session")
def graph_dir(indexed_state: Path) -> Path:
    """Alias of ``indexed_state`` used by MCP tool tests."""
    return indexed_state


@pytest.fixture(scope="session")
def adventureworks_dsn(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    """Start a Postgres container, apply AdventureWorks DDL, yield the DSN.

    Uses a separate session-scoped container from ``pagila_dsn`` so the two
    fixtures stay independent. The container is reused across the entire
    AdventureWorks test module, amortizing the ~40-table DDL load.
    """
    yield from _start_postgres_with_ddl(_ADVENTUREWORKS_DDL_PATH)


@pytest.fixture(scope="session")
def indexed_state_adventureworks(
    adventureworks_dsn: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run the full pretensor index pipeline against AdventureWorks."""
    return _index_database(
        adventureworks_dsn,
        "adventureworks",
        tmp_path_factory.mktemp("e2e_state_aw"),
    )


@pytest.fixture(scope="session")
def graph_dir_adventureworks(indexed_state_adventureworks: Path) -> Path:
    """Alias of ``indexed_state_adventureworks`` used by AdventureWorks tests."""
    return indexed_state_adventureworks


# ----------------------------------------------------------------------------
# TPC-DS fixture: 24-table snowflake schema for multi-valid-path stress tests
# ----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tpcds_dsn(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    """Start a Postgres container, apply TPC-DS DDL, yield the DSN.

    Uses a separate session-scoped container so TPC-DS fixtures stay
    independent from Pagila and AdventureWorks. The container is reused
    across the entire TPC-DS test module, amortizing the 24-table DDL load.
    """
    yield from _start_postgres_with_ddl(_TPCDS_DDL_PATH)


@pytest.fixture(scope="session")
def indexed_state_tpcds(
    tpcds_dsn: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run the full pretensor index pipeline against TPC-DS."""
    return _index_database(
        tpcds_dsn,
        "tpcds",
        tmp_path_factory.mktemp("e2e_state_tpcds"),
    )


@pytest.fixture(scope="session")
def graph_dir_tpcds(indexed_state_tpcds: Path) -> Path:
    """Alias of ``indexed_state_tpcds`` used by TPC-DS traverse tests."""
    return indexed_state_tpcds


# ----------------------------------------------------------------------------
# TPC-H fixture — audit-column-free control case (no audit columns)
# ----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tpch_dsn(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    """Start a Postgres container, apply TPC-H DDL, yield the DSN.

    Uses a separate session-scoped container so TPC-H stays independent
    from Pagila and AdventureWorks.
    """
    yield from _start_postgres_with_ddl(_TPCH_DDL_PATH)


@pytest.fixture(scope="session")
def indexed_state_tpch(
    tpch_dsn: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run the full pretensor index pipeline against TPC-H."""
    return _index_database(
        tpch_dsn,
        "tpch",
        tmp_path_factory.mktemp("e2e_state_tpch"),
    )


@pytest.fixture(scope="session")
def graph_dir_tpch(indexed_state_tpch: Path) -> Path:
    """Alias of ``indexed_state_tpch`` used by TPC-H tests."""
    return indexed_state_tpch


# ----------------------------------------------------------------------------
# Synthetic warehouse fixture for the dbt enrichment regression test
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticDbtWarehouse:
    """Bundle returned by :func:`synthetic_dbt_warehouse`.

    Holds everything the warehouse-scale dbt enrichment test needs after the
    expensive setup steps have run once: where the manifest lives on disk,
    the reference metadata produced at generation time, and the indexed Kuzu
    graph directory.
    """

    manifest_path: Path
    spec: SyntheticManifestSpec
    state_dir: Path
    graph_path: Path


def _apply_synthetic_warehouse_ddl(dsn: str, spec: SyntheticManifestSpec) -> None:
    """Materialize one schema and one ``(id int)`` table per resolver target.

    The dbt enrichment writers in ``src/pretensor/enrichment/dbt/`` skip any
    dbt id whose ``(connection, schema, physical_name)`` does not match a
    ``SchemaTable`` row, so the synthetic warehouse must contain a row for
    every ``(schema, physical_name)`` the manifest references. The DDL is
    minimal — no FKs, no constraints, no real data — so the test can validate
    the dbt-enrichment-only happy path without any introspection noise.
    """
    schemas: set[str] = {schema for schema, _ in spec.physical_tables}
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        for schema in sorted(schemas):
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')
        for schema, name in spec.physical_tables:
            cur.execute(f'CREATE TABLE "{schema}"."{name}" (id int);')
        conn.commit()
        cur.close()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@pytest.fixture(scope="session")
def synthetic_dbt_warehouse(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[SyntheticDbtWarehouse]:
    """Generate a 1,500-model synthetic dbt manifest, build a Postgres warehouse
    matching its physical identities, and run the pretensor index pipeline.

    Session-scoped because the setup is expensive (~1,500 ``CREATE TABLE`` +
    full introspection + Kuzu build). The Postgres container is torn down
    immediately after indexing finishes — the test only needs the on-disk
    Kuzu graph plus the generated manifest path.

    Returns a :class:`SyntheticDbtWarehouse` bundle ready for the
    enrichment-and-assertions phase in
    ``tests/e2e/test_dbt_warehouse_scale_enrichment.py``.
    """
    fixture_dir = tmp_path_factory.mktemp("synthetic_dbt")
    manifest_path = fixture_dir / "manifest.json"
    spec = write_synthetic_manifest(manifest_path)

    state_dir = tmp_path_factory.mktemp("e2e_state_synth_dbt")

    _ensure_image(_POSTGRES_IMAGE)
    with PostgresContainer(_POSTGRES_IMAGE) as pg:
        raw_url: str = pg.get_connection_url()
        dsn = raw_url.replace("postgresql+psycopg2://", "postgresql://")
        _apply_synthetic_warehouse_ddl(dsn, spec)
        _index_database(
            dsn,
            spec.connection_name,
            state_dir,
            run_relationship_discovery=False,
        )

    graph_path = state_dir / "graphs" / f"{spec.connection_name}.kuzu"
    yield SyntheticDbtWarehouse(
        manifest_path=manifest_path,
        spec=spec,
        state_dir=state_dir,
        graph_path=graph_path,
    )
