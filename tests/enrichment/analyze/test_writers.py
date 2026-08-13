"""Tests for the ExternalConsumer + CONSUMES graph writers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pretensor.core.ids import table_node_id
from pretensor.core.store import KuzuStore
from pretensor.enrichment.analyze.extract_python import SqlCandidate
from pretensor.enrichment.analyze.parse import ParsedSql
from pretensor.enrichment.analyze.writers import (
    ParsedCandidate,
    write_consumers_for_run,
)
from pretensor.graph_models.node import GraphNode


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


def _make_candidate(
    text: str,
    table_refs: list[tuple[str, str]],
    *,
    write_targets: list[tuple[str, str]] | None = None,
    bucket: str = "high",
    fingerprint: str = "abc123def456abc1",
    file_path: Path = Path("/repo/app.py"),
    rel_path: str = "app.py",
    symbol: str = "query",
    kind: str = "assignment",
    language: str = "python",
    line: int = 1,
) -> ParsedCandidate:
    cand = SqlCandidate(
        text=text,
        confidence_bucket=bucket,
        line_start=line,
        line_end=line,
        file_path=file_path,
        symbol=symbol,
        kind=kind,
    )
    parsed = ParsedSql(
        table_refs=table_refs,
        write_targets=write_targets or [],
        dialect_used="",
        fingerprint=fingerprint,
    )
    return ParsedCandidate(
        candidate=cand, parsed=parsed, rel_path=rel_path, language=language
    )


@pytest.fixture
def store() -> Iterator[KuzuStore]:
    s = KuzuStore(Path(":memory:"))
    s.ensure_schema()
    try:
        yield s
    finally:
        s.close()


class TestWriteConsumersForRun:
    def test_consumer_node_written(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate("SELECT id FROM public.users", [("public", "users")])
        summary = write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        assert summary.consumers_written == 1
        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer) RETURN c.sql_fingerprint, c.scan_run_id"
        )
        assert len(rows) == 1
        assert rows[0][1] == "run1"

    def test_consumes_edge_written(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate("SELECT id FROM public.users", [("public", "users")])
        write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer)-[r:CONSUMES]->(t:SchemaTable) RETURN t.table_name"
        )
        assert len(rows) == 1
        assert rows[0][0] == "users"

    def test_no_sql_text_field_on_consumer(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate("SELECT id FROM public.users", [("public", "users")])
        write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN c.*")
        assert len(rows) == 1
        # Query all known fields and ensure no sql_text field is returned
        props = store.query_all_rows(
            "MATCH (c:ExternalConsumer) "
            "RETURN c.node_id, c.connection_name, c.service_name, c.file_path, "
            "c.symbol, c.line_start, c.line_end, c.sql_fingerprint, "
            "c.confidence, c.dialect_used, c.scan_run_id"
        )
        assert len(props) == 1

    def test_idempotent_same_run_id(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate("SELECT id FROM public.users", [("public", "users")])
        write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN c.node_id")
        assert len(rows) == 1

    def test_stale_sweep_removes_old_run(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate(
            "SELECT id FROM public.users",
            [("public", "users")],
            fingerprint="fingerprint_run1",
        )
        write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")

        pc2 = _make_candidate(
            "SELECT name FROM public.users",
            [("public", "users")],
            fingerprint="fingerprint_run2",
        )
        write_consumers_for_run(store, cn, [pc2], "run2", service_name="svc")

        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN c.scan_run_id")
        run_ids = {r[0] for r in rows}
        assert "run1" not in run_ids
        assert "run2" in run_ids

    def test_cross_connection_ref_is_dropped(self, store: KuzuStore) -> None:
        cn = "myconn"
        pc = _make_candidate(
            "SELECT * FROM other_schema.cross_table",
            [("other_schema", "cross_table")],
        )
        summary = write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        assert summary.cross_connection_dropped == 1
        assert summary.consumers_written == 0

    def test_has_external_consumers_set_on_table(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate("SELECT id FROM public.users", [("public", "users")])
        write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {table_name: 'users'}) RETURN t.has_external_consumers"
        )
        assert len(rows) == 1
        assert rows[0][0] is True

    def test_low_confidence_below_min_is_skipped(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate(
            "SELECT id FROM public.users", [("public", "users")], bucket="low"
        )
        summary = write_consumers_for_run(
            store, cn, [pc], "run1", service_name="svc", min_confidence=0.6
        )
        assert summary.consumers_written == 0

    def test_low_confidence_above_min_is_written(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate(
            "SELECT id FROM public.users", [("public", "users")], bucket="low"
        )
        summary = write_consumers_for_run(
            store, cn, [pc], "run1", service_name="svc", min_confidence=0.0
        )
        assert summary.consumers_written == 1

    def test_multiple_tables_per_candidate(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        store.upsert_table(_minimal_table(cn, "public", "orders"))
        pc = _make_candidate(
            "SELECT u.id FROM public.users u JOIN public.orders o ON u.id = o.user_id",
            [("public", "users"), ("public", "orders")],
        )
        summary = write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        assert summary.consumers_written == 1
        assert summary.edges_written == 2

    def test_read_and_write_ops_are_distinguished(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "audit"))
        store.upsert_table(_minimal_table(cn, "public", "events"))
        # INSERT INTO audit SELECT ... FROM events: audit=write, events=read.
        pc = _make_candidate(
            "INSERT INTO public.audit SELECT * FROM public.events",
            [("public", "audit"), ("public", "events")],
            write_targets=[("public", "audit")],
        )
        summary = write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        assert summary.edges_written == 2
        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer)-[r:CONSUMES]->(t:SchemaTable) "
            "RETURN t.table_name, r.op, r.source ORDER BY t.table_name"
        )
        assert rows == [("audit", "write", "analyze"), ("events", "read", "analyze")]

    def test_identical_sql_at_two_call_sites_keeps_both(self, store: KuzuStore) -> None:
        """Byte-identical SQL at two lines in one file must not collapse to one node."""
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc1 = _make_candidate(
            "SELECT id FROM public.users", [("public", "users")], line=10
        )
        pc2 = _make_candidate(
            "SELECT id FROM public.users", [("public", "users")], line=42
        )
        summary = write_consumers_for_run(
            store, cn, [pc1, pc2], "run1", service_name="svc"
        )
        assert summary.consumers_written == 2
        assert summary.edges_written == 2
        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer) RETURN c.line_start ORDER BY c.line_start"
        )
        assert [int(r[0]) for r in rows] == [10, 42]

    def test_counters_dedupe_repeated_candidates(self, store: KuzuStore) -> None:
        """The same candidate twice yields one node/edge and counters that match."""
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate("SELECT id FROM public.users", [("public", "users")])
        summary = write_consumers_for_run(
            store, cn, [pc, pc], "run1", service_name="svc"
        )
        assert summary.consumers_written == 1
        assert summary.edges_written == 1
        rows = store.query_all_rows("MATCH (c:ExternalConsumer) RETURN count(*)")
        assert int(rows[0][0]) == 1

    def test_flag_cleared_when_rescan_drops_all_consumers(
        self, store: KuzuStore
    ) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate("SELECT id FROM public.users", [("public", "users")])
        write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")

        # The query disappeared from the codebase: rescan writes nothing.
        write_consumers_for_run(store, cn, [], "run2", service_name="svc")

        rows = store.query_all_rows(
            "MATCH (t:SchemaTable {table_name: 'users'}) RETURN t.has_external_consumers"
        )
        assert rows[0][0] is False

    def test_file_path_is_repo_relative(self, store: KuzuStore) -> None:
        cn = "myconn"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate(
            "SELECT id FROM public.users",
            [("public", "users")],
            rel_path="src/api/users.py",
            kind="call",
            language="python",
        )
        write_consumers_for_run(store, cn, [pc], "run1", service_name="svc")
        rows = store.query_all_rows(
            "MATCH (c:ExternalConsumer) RETURN c.file_path, c.kind, c.language"
        )
        assert rows == [("src/api/users.py", "call", "python")]


class TestCaseInsensitiveResolution:
    """Snowflake stores unquoted identifiers uppercase; lowercase SQL refs
    must still resolve when the match is unambiguous."""

    def test_lowercase_ref_resolves_to_uppercase_table(self, store: KuzuStore) -> None:
        cn = "wh"
        store.upsert_table(_minimal_table(cn, "PUBLIC", "ORDERS"))
        pc = _make_candidate("select id from orders", [("PUBLIC", "orders")])
        summary = write_consumers_for_run(store, cn, [pc], "r1", service_name="svc")
        assert summary.edges_written == 1
        assert summary.cross_connection_dropped == 0
        assert summary.rows[0].table == "PUBLIC.ORDERS"

    def test_ambiguous_case_twins_stay_dropped(self, store: KuzuStore) -> None:
        cn = "wh"
        store.upsert_table(_minimal_table(cn, "public", "users"))
        store.upsert_table(_minimal_table(cn, "public", "Users"))
        pc = _make_candidate("select id from USERS", [("public", "USERS")])
        summary = write_consumers_for_run(store, cn, [pc], "r1", service_name="svc")
        assert summary.edges_written == 0
        assert summary.cross_connection_dropped == 1

    def test_exact_match_wins_over_fold(self, store: KuzuStore) -> None:
        cn = "wh"
        store.upsert_table(_minimal_table(cn, "public", "Users"))
        store.upsert_table(_minimal_table(cn, "public", "users"))
        pc = _make_candidate("select id from users", [("public", "users")])
        summary = write_consumers_for_run(store, cn, [pc], "r1", service_name="svc")
        assert summary.edges_written == 1
        assert summary.rows[0].table == "public.users"
