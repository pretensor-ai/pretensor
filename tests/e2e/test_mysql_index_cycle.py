"""E2E: full MySQL index cycle — introspect → snapshot → Kuzu graph.

Requires Docker and PRETENSOR_E2E=1. Uses the Sakila MySQL fixture defined
in tests/e2e/fixtures/sql/sakila_mysql_ddl.sql.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

if not os.getenv("PRETENSOR_E2E"):
    pytest.skip("set PRETENSOR_E2E=1", allow_module_level=True)

from pretensor.connectors.inspect import inspect  # noqa: E402
from pretensor.core.builder import GraphBuilder  # noqa: E402
from pretensor.core.store import KuzuStore  # noqa: E402
from pretensor.introspection.models.dsn import connection_config_from_url  # noqa: E402

_EXPECTED_TABLES = {"actor", "film", "film_actor", "customer", "rental", "payment"}
_EXPECTED_FK_PAIRS = {
    # (source_table, target_table)
    ("film_actor", "actor"),
    ("film_actor", "film"),
    ("rental", "customer"),
    ("payment", "customer"),
    ("payment", "rental"),
}


@pytest.mark.e2e
def test_mysql_index_builds_kuzu_graph(
    indexed_state_mysql: Path,
) -> None:
    """Kuzu graph directory exists and contains the expected node file."""
    graph_path = indexed_state_mysql / "graphs" / "sakila_mysql.kuzu"
    assert graph_path.exists(), f"Expected Kuzu graph at {graph_path}"


@pytest.mark.e2e
def test_mysql_tables_discovered(
    sakila_mysql_dsn: str,
) -> None:
    """MySQLConnector discovers all Sakila tables and the view."""
    cfg = connection_config_from_url(sakila_mysql_dsn, "sakila_mysql")
    snapshot = inspect(cfg)

    table_names = {t.name for t in snapshot.tables}
    assert _EXPECTED_TABLES <= table_names, f"Missing tables: {_EXPECTED_TABLES - table_names}"

    actor = next(t for t in snapshot.tables if t.name == "actor")
    assert actor.comment == "Actor catalog"
    assert actor.table_type == "table"

    view_names = {t.name for t in snapshot.tables if t.table_type == "view"}
    assert "actor_info" in view_names


@pytest.mark.e2e
def test_mysql_columns_introspected(
    sakila_mysql_dsn: str,
) -> None:
    """Columns for actor table have correct PK, types, and index flags."""
    cfg = connection_config_from_url(sakila_mysql_dsn, "sakila_mysql")
    snapshot = inspect(cfg)

    actor = next(t for t in snapshot.tables if t.name == "actor")
    col_by_name = {c.name: c for c in actor.columns}

    assert col_by_name["actor_id"].is_primary_key is True
    assert col_by_name["actor_id"].is_indexed is True
    assert col_by_name["last_name"].is_indexed is True  # KEY idx_actor_last_name


@pytest.mark.e2e
def test_mysql_fk_edges_present(
    sakila_mysql_dsn: str,
    tmp_path: Path,
) -> None:
    """FK relationships are discovered and written to the Kuzu graph."""
    cfg = connection_config_from_url(sakila_mysql_dsn, "sakila_mysql")
    snapshot = inspect(cfg)

    # Verify FK discovery at snapshot level
    fk_pairs = {
        (fk.source_table, fk.target_table)
        for t in snapshot.tables
        for fk in t.foreign_keys
    }
    assert _EXPECTED_FK_PAIRS <= fk_pairs, f"Missing FKs: {_EXPECTED_FK_PAIRS - fk_pairs}"

    # Build graph and verify FK_REFERENCES edges
    graph_path = tmp_path / "sakila_fk.kuzu"
    store = KuzuStore(graph_path)
    GraphBuilder().build(snapshot, store, run_relationship_discovery=False)

    rows = store.execute(
        "MATCH (src:SchemaTable)-[r:FK_REFERENCES]->(tgt:SchemaTable) "
        "RETURN src.table_name AS src, tgt.table_name AS tgt"
    )
    store.close()

    edge_pairs = {(r["src"], r["tgt"]) for r in rows}
    assert _EXPECTED_FK_PAIRS <= edge_pairs, f"Missing FK edges in graph: {_EXPECTED_FK_PAIRS - edge_pairs}"


@pytest.mark.e2e
def test_mysql_view_lineage_discovered(
    sakila_mysql_dsn: str,
) -> None:
    """actor_info view lineage references actor and film_actor tables."""
    cfg = connection_config_from_url(sakila_mysql_dsn, "sakila_mysql")
    snapshot = inspect(cfg)

    lineage_targets = {dep.target_table for dep in snapshot.view_dependencies}
    assert "actor_info" in lineage_targets

    actor_info_deps = [
        dep for dep in snapshot.view_dependencies if dep.target_table == "actor_info"
    ]
    source_tables = {dep.source_table for dep in actor_info_deps}
    assert "actor" in source_tables
    assert "film_actor" in source_tables
