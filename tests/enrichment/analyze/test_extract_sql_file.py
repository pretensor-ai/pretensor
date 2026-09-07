"""Tests for the bare .sql file extractor (unit + end-to-end pipeline)."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.enrichment.analyze.extract_sql_file import (
    SqlFileParseError,
    extract_sql_file_candidates,
    parse_sql_file,
)
from pretensor.enrichment.analyze.pipeline import run_analyze_enrichment
from pretensor.graph_models.node import GraphNode

_FIXTURES = Path(__file__).parent / "fixtures"


def _minimal_table(
    connection: str, schema: str, name: str, *, database: str = "testdb"
) -> GraphNode:
    return GraphNode(
        node_id=table_node_id(connection, schema, name),
        connection_name=connection,
        database=database,
        schema_name=schema,
        table_name=name,
        row_count=None,
        comment=None,
        entity_type=None,
        table_type="table",
        seq_scan_count=None,
        idx_scan_count=None,
        insert_count=None,
        update_count=None,
        delete_count=None,
        is_partitioned=None,
        partition_key=None,
        grants_json=None,
        access_read_count=None,
        access_write_count=None,
        days_since_last_access=None,
        potentially_unused=None,
        table_bytes=None,
        clustering_key=None,
    )


@pytest.fixture
def store() -> Iterator[KuzuStore]:
    s = KuzuStore(Path(":memory:"))
    s.ensure_schema()
    try:
        yield s
    finally:
        s.close()


def _repo_with(tmp_path: Path, *fixture_names: str) -> Path:
    """Copy named fixture files into a temp repo directory."""
    for name in fixture_names:
        shutil.copy(_FIXTURES / name, tmp_path / name)
    return tmp_path


class TestExtractSqlFileCandidates:
    def test_single_whole_file_candidate(self, tmp_path: Path) -> None:
        sql_file = tmp_path / "report.sql"
        sql_file.write_text("SELECT id FROM users;\n", encoding="utf-8")

        candidates = extract_sql_file_candidates(sql_file)

        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.kind == "sql_file"
        assert cand.symbol == "report.sql"
        assert cand.confidence_bucket == "high"
        assert cand.line_start == 1
        assert cand.text == "SELECT id FROM users;\n"

    def test_empty_file_yields_nothing(self, tmp_path: Path) -> None:
        sql_file = tmp_path / "empty.sql"
        sql_file.write_text("   \n\n", encoding="utf-8")
        assert extract_sql_file_candidates(sql_file) == []

    def test_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        assert extract_sql_file_candidates(tmp_path / "gone.sql") == []


class TestParseSqlFile:
    def test_multi_statement_unions_refs(self) -> None:
        text = (
            "SELECT id FROM users;\n"
            "INSERT INTO orders (id) VALUES (1);\n"
            "UPDATE accounts SET x = 1 WHERE id = 2;\n"
        )
        parsed = parse_sql_file(text)
        tables = {t for _, t in parsed.table_refs}
        assert {"users", "orders", "accounts"} <= tables
        write_tables = {t for _, t in parsed.write_targets}
        assert write_tables == {"orders", "accounts"}

    def test_unparseable_raises(self) -> None:
        with pytest.raises(SqlFileParseError):
            parse_sql_file("this is not sql at all {{{ %%% )))\n")

    def test_valid_sql_without_tables_does_not_raise(self) -> None:
        parsed = parse_sql_file("SELECT 1;\n")
        assert parsed.table_refs == []
        assert parsed.write_targets == []


class TestSqlFilePipeline:
    def test_simple_sql_file_produces_consumer_and_edge(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "simple.sql")

        summary = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        assert summary.consumers_written == 1
        assert summary.edges_written == 1
        assert summary.sql_candidates_failed == 0

        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer)-[r:CONSUMES]->(t:SchemaTable) "
            "RETURN c.kind, c.symbol, c.language, c.file_path, "
            "c.sql_fingerprint, t.table_name, r.op"
        )
        assert len(rows) == 1
        kind, symbol, language, file_path, fingerprint, table_name, op = rows[0]
        assert kind == "sql_file"
        assert symbol == "simple.sql"
        assert language == "sql"
        assert file_path == "simple.sql"
        assert fingerprint is not None and len(fingerprint) == 16
        assert table_name == "users"
        assert op == "read"

    def test_multi_statement_file_unions_refs_into_one_consumer(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        store.upsert_table(_minimal_table(cn, "public", "orders"))
        store.upsert_table(_minimal_table(cn, "public", "accounts"))
        repo = _repo_with(tmp_path, "multi_statement.sql")

        summary = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        assert summary.consumers_written == 1
        assert summary.edges_written == 3

        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer)-[r:CONSUMES]->(t:SchemaTable) "
            "RETURN t.table_name, r.op"
        )
        assert set(rows) == {
            ("users", "read"),
            ("orders", "write"),
            ("accounts", "write"),
        }

    def test_unparseable_file_counts_failed_and_writes_nothing(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "unparseable.sql")

        summary = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        assert summary.sql_candidates_failed == 1
        assert summary.consumers_written == 0
        assert summary.edges_written == 0
        rows = store.query_all_rows("MATCH ()-[r:CONSUMES]->() RETURN count(*)")
        assert int(rows[0][0]) == 0

    def test_excludes_pattern_skips_sql_files(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "simple.sql", "simple_select.py")

        summary = run_analyze_enrichment(
            repo, store, cn, scan_run_id="run1", excludes=["*.sql"]
        )

        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN c.kind")
        kinds = {r[0] for r in rows}
        assert "sql_file" not in kinds
        assert summary.consumers_written >= 1  # the Python consumer still lands

    def test_includes_pattern_limits_to_sql_files(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "simple.sql", "simple_select.py")

        summary = run_analyze_enrichment(
            repo, store, cn, scan_run_id="run1", includes=["*.sql"]
        )

        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN c.kind")
        assert {r[0] for r in rows} == {"sql_file"}
        assert summary.consumers_written == 1
