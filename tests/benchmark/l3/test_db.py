"""Unit tests for L3 db helpers (env-var resolution + SELECT guard +
``execute_query`` error wrapping).

The happy-path of ``execute_query`` against a real database is exercised
by the e2e test under ``tests/e2e/test_l3_baseline.py`` (gated on
``PRETENSOR_E2E=1``); the unit tests below cover the refusal guard and
the SQLAlchemy error wrap without standing up a real Postgres.
"""

from __future__ import annotations

import pytest

import pretensor.benchmark.l3.db as db_mod
from pretensor.benchmark.l3.db import (
    QueryExecutionError,
    RefusedNonSelectError,
    execute_query,
    is_select_only,
    resolve_database_url,
)
from pretensor.benchmark.runner import Dataset

# ---------------------------------------------------------------------------
# resolve_database_url
# ---------------------------------------------------------------------------


def test_resolve_database_url_reads_pagila_env() -> None:
    env = {"PAGILA_DATABASE_URL": "postgresql://x/p"}
    assert resolve_database_url(Dataset.PAGILA, env=env) == "postgresql://x/p"


def test_resolve_database_url_reads_tpch_env() -> None:
    env = {"TPCH_DATABASE_URL": "postgresql://x/t"}
    assert resolve_database_url(Dataset.TPCH, env=env) == "postgresql://x/t"


def test_resolve_database_url_reads_adventureworks_env() -> None:
    env = {"ADVENTUREWORKS_DATABASE_URL": "postgresql://x/a"}
    assert resolve_database_url(Dataset.ADVENTUREWORKS, env=env) == "postgresql://x/a"


def test_resolve_database_url_raises_helpful_error_when_unset() -> None:
    with pytest.raises(LookupError, match="PAGILA_DATABASE_URL"):
        resolve_database_url(Dataset.PAGILA, env={})


def test_resolve_database_url_raises_when_value_empty() -> None:
    with pytest.raises(LookupError, match="PAGILA_DATABASE_URL"):
        resolve_database_url(Dataset.PAGILA, env={"PAGILA_DATABASE_URL": ""})


def test_resolve_database_url_rejects_unsupported_dataset() -> None:
    """Synthetic datasets without DDL must produce a clear error."""
    with pytest.raises(LookupError, match="does not support"):
        resolve_database_url(Dataset.ANALYTICS_DWH, env={})


def test_resolve_database_url_uses_os_environ_when_env_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAGILA_DATABASE_URL", "postgresql://defaulted")
    assert resolve_database_url(Dataset.PAGILA) == "postgresql://defaulted"


# ---------------------------------------------------------------------------
# is_select_only
# ---------------------------------------------------------------------------


def test_is_select_only_accepts_plain_select() -> None:
    assert is_select_only("SELECT 1") is True


def test_is_select_only_accepts_select_lowercase() -> None:
    assert is_select_only("select * from t") is True


def test_is_select_only_accepts_with_cte() -> None:
    assert is_select_only("WITH x AS (SELECT 1) SELECT * FROM x") is True


def test_is_select_only_accepts_parenthesised_select() -> None:
    assert is_select_only("(SELECT 1) UNION (SELECT 2)") is True


def test_is_select_only_accepts_table_form() -> None:
    """PostgreSQL ``TABLE foo`` is sugar for ``SELECT * FROM foo``."""
    assert is_select_only("TABLE film") is True


def test_is_select_only_accepts_values_form() -> None:
    assert is_select_only("VALUES (1), (2)") is True


def test_is_select_only_rejects_drop_table() -> None:
    assert is_select_only("DROP TABLE film") is False


def test_is_select_only_rejects_delete() -> None:
    assert is_select_only("DELETE FROM film") is False


def test_is_select_only_rejects_update() -> None:
    assert is_select_only("UPDATE film SET title = 'x'") is False


def test_is_select_only_rejects_insert() -> None:
    assert is_select_only("INSERT INTO film VALUES (1)") is False


def test_is_select_only_rejects_truncate() -> None:
    assert is_select_only("TRUNCATE film") is False


def test_is_select_only_rejects_empty_string() -> None:
    assert is_select_only("") is False
    assert is_select_only("   \n  ") is False


def test_is_select_only_handles_leading_line_comment() -> None:
    sql = "-- the model added a comment\nSELECT 1"
    assert is_select_only(sql) is True


def test_is_select_only_handles_leading_block_comment() -> None:
    sql = "/* explanation */ SELECT 1"
    assert is_select_only(sql) is True


def test_is_select_only_rejects_drop_hidden_after_comment() -> None:
    """A comment can't smuggle in a DROP — guard checks first real token."""
    sql = "-- innocent\nDROP TABLE film"
    assert is_select_only(sql) is False


def test_is_select_only_rejects_lookalike_keyword() -> None:
    """``SELECTED`` is not ``SELECT`` — the guard is keyword-aware."""
    assert is_select_only("SELECTED 1") is False


# ---------------------------------------------------------------------------
# error type hierarchy (regression: callers rely on this for filtering)
# ---------------------------------------------------------------------------


def test_refused_non_select_subclasses_query_execution_error() -> None:
    assert issubclass(RefusedNonSelectError, QueryExecutionError)


# ---------------------------------------------------------------------------
# execute_query: refusal guard fires before the connection is opened
# ---------------------------------------------------------------------------


def test_execute_query_refuses_non_select_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SELECT-only guard must fire before any DB connection is opened."""
    create_engine_calls: list[str] = []

    def fake_create_engine(dsn: str) -> object:
        create_engine_calls.append(dsn)
        raise AssertionError(
            "execute_query must reject non-SELECT SQL before opening a connection"
        )

    monkeypatch.setattr(db_mod.sqlalchemy, "create_engine", fake_create_engine)

    with pytest.raises(RefusedNonSelectError):
        execute_query(
            "postgresql://does-not-matter",
            "DROP TABLE film",
            enforce_select_only=True,
        )
    assert create_engine_calls == [], (
        "create_engine must NOT be invoked when the agent SQL is refused"
    )


def test_execute_query_does_not_apply_guard_for_gold_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without enforce_select_only=True, the runner reaches create_engine."""
    create_engine_calls: list[str] = []

    def fake_create_engine(dsn: str) -> object:
        create_engine_calls.append(dsn)
        raise RuntimeError(
            "stop here — we just wanted to confirm the guard didn't fire"
        )

    monkeypatch.setattr(db_mod.sqlalchemy, "create_engine", fake_create_engine)

    # A statement the SELECT-only guard would reject: shows that
    # enforce_select_only=False genuinely skips the guard.
    with pytest.raises(RuntimeError, match="stop here"):
        execute_query(
            "postgresql://does-not-matter",
            "UPDATE film SET title = 'x'",
            enforce_select_only=False,
        )
    assert create_engine_calls == ["postgresql://does-not-matter"]


# ---------------------------------------------------------------------------
# execute_query: SQLAlchemy errors are wrapped in QueryExecutionError
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Stand-in for the result of ``conn.execute`` that raises on use."""

    def fetchall(self) -> list[tuple[object, ...]]:
        raise db_mod.SQLAlchemyError("simulated driver fault")

    def keys(self) -> list[str]:
        return []


class _FakeConnection:
    def __init__(self, raise_on_execute: bool = False) -> None:
        self._raise_on_execute = raise_on_execute
        self.rolled_back = False

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def begin(self) -> _FakeConnection:
        return self  # the same object models both connection and transaction

    def execute(self, _stmt: object) -> _FakeCursor:
        if self._raise_on_execute:
            raise db_mod.SQLAlchemyError("relation 'nope' does not exist")
        return _FakeCursor()

    def rollback(self) -> None:
        self.rolled_back = True


class _FakeEngine:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn
        self.disposed = False

    def connect(self) -> _FakeConnection:
        return self._conn

    def dispose(self) -> None:
        self.disposed = True


def test_execute_query_wraps_sqlalchemy_error_from_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SQLAlchemyError raised by the driver becomes a QueryExecutionError."""
    fake_conn = _FakeConnection(raise_on_execute=True)
    fake_engine = _FakeEngine(fake_conn)
    monkeypatch.setattr(db_mod.sqlalchemy, "create_engine", lambda _dsn: fake_engine)

    with pytest.raises(QueryExecutionError, match="relation 'nope'"):
        execute_query("postgresql://stub", "SELECT * FROM nope")

    # Defense-in-depth invariants: the transaction was rolled back and the
    # engine was disposed even on the error path.
    assert fake_conn.rolled_back is True
    assert fake_engine.disposed is True


def test_execute_query_wraps_sqlalchemy_error_from_fetchall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SQLAlchemyError raised by cursor.fetchall is also wrapped."""
    fake_conn = _FakeConnection(raise_on_execute=False)
    fake_engine = _FakeEngine(fake_conn)
    monkeypatch.setattr(db_mod.sqlalchemy, "create_engine", lambda _dsn: fake_engine)

    with pytest.raises(QueryExecutionError, match="simulated driver fault"):
        execute_query("postgresql://stub", "SELECT 1")
    assert fake_conn.rolled_back is True
    assert fake_engine.disposed is True
