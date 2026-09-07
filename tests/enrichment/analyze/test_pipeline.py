"""End-to-end tests for run_analyze_enrichment."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.enrichment.analyze.pipeline import (
    resolve_default_schema,
    run_analyze_enrichment,
)
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


class TestEmptyGraphGuard:
    def test_raises_when_no_schema_tables(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="pretensor index --connection myconn"):
            run_analyze_enrichment(tmp_path, store, "myconn")

    def test_error_message_contains_connection_name(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            run_analyze_enrichment(tmp_path, store, "prod_warehouse")
        assert "prod_warehouse" in str(exc_info.value)


class TestSimpleSelect:
    def test_produces_consumer_and_edge(self, store: KuzuStore, tmp_path: Path) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "simple_select.py")

        summary = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        assert summary.consumers_written >= 1
        assert summary.edges_written >= 1
        assert summary.files_scanned >= 1

    def test_graph_has_consumer_and_consumes_edge(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "simple_select.py")

        run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer)-[:CONSUMES]->(t:SchemaTable) RETURN t.table_name"
        )
        assert len(rows) >= 1
        table_names = {r[0] for r in rows}
        assert "users" in table_names

    def test_idempotent_same_scan_run_id(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "simple_select.py")

        s1 = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")
        s2 = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        assert s1.consumers_written == s2.consumers_written
        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN count(*)")
        assert int(rows[0][0]) == s1.consumers_written

    def test_stale_sweep_on_new_scan_run_id(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "simple_select.py")

        run_analyze_enrichment(repo, store, cn, scan_run_id="run1")
        run_analyze_enrichment(repo, store, cn, scan_run_id="run2")

        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN c.scan_run_id")
        run_ids = {r[0] for r in rows}
        assert "run1" not in run_ids
        assert "run2" in run_ids


class TestNoqaOptout:
    def test_noqa_lines_produce_zero_consumers_for_opted_out(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        # Insert tables referenced by noqa_optout.py so we can measure suppression
        store.upsert_table(_minimal_table(cn, "public", "noqa_table"))
        store.upsert_table(_minimal_table(cn, "public", "noqa_table2"))
        store.upsert_table(_minimal_table(cn, "public", "normal_table"))
        repo = _repo_with(tmp_path, "noqa_optout.py")

        summary = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer)-[:CONSUMES]->(t:SchemaTable) "
            "WHERE t.table_name IN ['noqa_table', 'noqa_table2'] RETURN t.table_name"
        )
        assert len(rows) == 0
        assert summary.consumers_written == 1


class TestCrossConnection:
    def test_cross_connection_produces_zero_edges(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "something"))
        repo = _repo_with(tmp_path, "cross_connection.py")

        summary = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        rows = store.query_all_rows("MATCH ()-[r:CONSUMES]->() RETURN count(*)")
        assert int(rows[0][0]) == 0
        assert summary.cross_connection_dropped > 0


class TestDynamicOnly:
    def test_dynamic_only_produces_zero_edges(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "dynamic_only.py")

        summary = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        assert summary.edges_written == 0
        assert summary.consumers_written == 0


class TestFstringPrefix:
    def test_fstring_prefix_with_relaxed_min_confidence_writes_consumer(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "fstring_prefix.py")

        summary = run_analyze_enrichment(
            repo, store, cn, scan_run_id="run1", min_confidence=0.0
        )

        assert summary.consumers_written >= 1

    def test_fstring_prefix_excluded_by_default_min_confidence(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "fstring_prefix.py")

        summary = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        assert summary.consumers_written == 0


class TestDefaultSchema:
    def test_unqualified_ref_resolves_via_default_schema(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "analytics", "users"))
        (tmp_path / "app.py").write_text(
            'query = "SELECT id FROM users"\n', encoding="utf-8"
        )

        summary = run_analyze_enrichment(
            tmp_path, store, cn, scan_run_id="run1", default_schema="analytics"
        )

        assert summary.consumers_written == 1
        rows = store.query_all_rows(
            "MATCH (:ExternalConsumer)-[:CONSUMES]->(t:SchemaTable) "
            "RETURN t.schema_name, t.table_name"
        )
        assert rows == [("analytics", "users")]

    def test_unqualified_ref_drops_without_matching_default_schema(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "analytics", "users"))
        (tmp_path / "app.py").write_text(
            'query = "SELECT id FROM users"\n', encoding="utf-8"
        )

        # Explicit "public" mis-resolves the unqualified ref → dropped, not
        # written — the explicit flag always beats the derived schema.
        summary = run_analyze_enrichment(
            tmp_path, store, cn, scan_run_id="run1", default_schema="public"
        )

        assert summary.consumers_written == 0
        assert summary.cross_connection_dropped == 1
        assert summary.default_schema == "public"


class TestDerivedDefaultSchema:
    def test_single_schema_graph_derives_schema_without_flag(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "analytics", "users"))
        (tmp_path / "app.py").write_text(
            'query = "SELECT id FROM users"\n', encoding="utf-8"
        )

        summary = run_analyze_enrichment(tmp_path, store, cn, scan_run_id="run1")

        assert summary.default_schema == "analytics"
        assert summary.consumers_written == 1
        rows = store.query_all_rows(
            "MATCH (:ExternalConsumer)-[:CONSUMES]->(t:SchemaTable) "
            "RETURN t.schema_name, t.table_name"
        )
        assert rows == [("analytics", "users")]

    def test_mysql_multi_schema_falls_back_to_database_name(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "appdb", "users"))
        store.upsert_table(_minimal_table(cn, "otherdb", "logs"))
        (tmp_path / "app.py").write_text(
            'query = "SELECT id FROM users"\n', encoding="utf-8"
        )

        summary = run_analyze_enrichment(
            tmp_path,
            store,
            cn,
            scan_run_id="run1",
            dialect="mysql",
            database="appdb",
        )

        assert summary.default_schema == "appdb"
        assert summary.consumers_written == 1

    def test_snowflake_multi_schema_falls_back_to_upper_public(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "PUBLIC", "USERS"))
        store.upsert_table(_minimal_table(cn, "MARTS", "ORDERS"))
        (tmp_path / "app.py").write_text(
            'query = "SELECT id FROM users"\n', encoding="utf-8"
        )

        summary = run_analyze_enrichment(
            tmp_path,
            store,
            cn,
            scan_run_id="run1",
            dialect="snowflake",
            database="warehouse",
        )

        assert summary.default_schema == "PUBLIC"
        assert summary.consumers_written == 1

    def test_bigquery_falls_back_to_dataset_portion_of_database(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "sales", "orders"))
        store.upsert_table(_minimal_table(cn, "staging", "orders_raw"))
        (tmp_path / "app.py").write_text(
            'query = "SELECT id FROM orders"\n', encoding="utf-8"
        )

        summary = run_analyze_enrichment(
            tmp_path,
            store,
            cn,
            scan_run_id="run1",
            dialect="bigquery",
            database="my-project/sales",
        )

        assert summary.default_schema == "sales"
        assert summary.consumers_written == 1

    def test_no_metadata_multi_schema_falls_back_to_public(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "analytics", "users"))
        store.upsert_table(_minimal_table(cn, "public", "orders"))
        (tmp_path / "app.py").write_text(
            'query = "SELECT id FROM orders"\n', encoding="utf-8"
        )

        summary = run_analyze_enrichment(tmp_path, store, cn, scan_run_id="run1")

        assert summary.default_schema == "public"
        assert summary.consumers_written == 1

    def test_explicit_empty_string_schema_is_not_derived(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        """Only ``None`` triggers derivation — a falsy explicit value is honored."""
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "analytics", "users"))
        (tmp_path / "app.py").write_text(
            'query = "SELECT id FROM users"\n', encoding="utf-8"
        )

        summary = run_analyze_enrichment(
            tmp_path, store, cn, scan_run_id="run1", default_schema=""
        )

        # Derivation would resolve "analytics" and write a consumer; the
        # explicit empty string must pass through untouched instead.
        assert summary.default_schema == ""
        assert summary.consumers_written == 0


class TestResolveDefaultSchema:
    def _span_two_schemas(self, store: KuzuStore, cn: str) -> None:
        """Defeat the single-schema snapshot signal."""
        store.upsert_table(_minimal_table(cn, "alpha", "users"))
        store.upsert_table(_minimal_table(cn, "beta", "orders"))

    def test_bigquery_database_without_slash_uses_whole_string(
        self, store: KuzuStore
    ) -> None:
        cn = "myconn"
        self._span_two_schemas(store, cn)

        resolved = resolve_default_schema(
            store, cn, dialect="bigquery", database="sales"
        )

        assert resolved == "sales"

    def test_mysql_without_database_falls_back_to_public(
        self, store: KuzuStore
    ) -> None:
        cn = "myconn"
        self._span_two_schemas(store, cn)

        assert resolve_default_schema(store, cn, dialect="mysql") == "public"

    def test_bigquery_without_database_falls_back_to_public(
        self, store: KuzuStore
    ) -> None:
        cn = "myconn"
        self._span_two_schemas(store, cn)

        assert resolve_default_schema(store, cn, dialect="bigquery") == "public"


class TestNoSqlTextField:
    def test_external_consumer_has_no_sql_text(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        repo = _repo_with(tmp_path, "simple_select.py")
        run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer) "
            "RETURN c.node_id, c.sql_fingerprint, c.service_name, c.file_path, "
            "c.symbol, c.line_start, c.line_end, c.confidence, "
            "c.dialect_used, c.scan_run_id, c.connection_name"
        )
        assert len(rows) >= 1
        fingerprint = rows[0][1]
        assert fingerprint is not None and len(fingerprint) == 16


class TestNestedGitignore:
    def test_nested_gitignore_suppresses_consumer(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        (tmp_path / "app.py").write_text(
            'query = "SELECT id FROM public.users"\n', encoding="utf-8"
        )
        gen = tmp_path / "gen"
        gen.mkdir()
        (gen / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
        (gen / "ignored.py").write_text(
            'query = "SELECT id FROM public.users"\n', encoding="utf-8"
        )

        summary = run_analyze_enrichment(tmp_path, store, cn, scan_run_id="run1")

        assert summary.files_scanned == 1
        assert summary.consumers_written == 1
        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN c.file_path")
        assert [r[0] for r in rows] == ["app.py"]


class TestDmlWrites:
    def test_insert_statement_produces_consumer(
        self, store: KuzuStore, tmp_path: Path
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "orders"))
        repo = _repo_with(tmp_path, "dml_writes.py")

        summary = run_analyze_enrichment(repo, store, cn, scan_run_id="run1")

        assert summary.consumers_written >= 1
        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer)-[r:CONSUMES]->(t:SchemaTable) "
            "RETURN t.table_name, r.op"
        )
        assert ("orders", "write") in rows  # INSERT target must carry op=write
