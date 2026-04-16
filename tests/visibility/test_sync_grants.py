"""Tests for grant-driven ``visibility.yml`` merge (``pretensor sync-grants``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from pretensor.connectors.base import BaseConnector, TableGrant
from pretensor.introspection.models.config import (
    ConnectionConfig,
    DatabaseType,
    SchemaFilter,
)
from pretensor.visibility.config import load_visibility_config
from pretensor.visibility.sync_grants import (
    allowed_table_patterns_for_grants,
    merge_grant_profiles_into_visibility_doc,
    run_sync_grants,
)


class _StubGrantsConnector(BaseConnector):
    """Minimal connector that only implements ``get_table_grants`` for tests."""

    def __init__(self, grants: list[TableGrant]) -> None:
        super().__init__(
            ConnectionConfig(name="demo", type=DatabaseType.POSTGRES),
        )
        self._grants = grants

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def get_tables(self, schema_filter: SchemaFilter | None = None) -> list[Any]:
        _ = schema_filter
        return []

    def get_columns(self, table_name: str, schema_name: str) -> list[Any]:
        _ = table_name, schema_name
        return []

    def get_foreign_keys(self) -> list[Any]:
        return []

    def get_table_row_count(self, table_name: str, schema_name: str) -> int:
        _ = table_name, schema_name
        return 0

    def get_column_stats(
        self, table_name: str, column_name: str, schema_name: str
    ) -> Any:
        _ = table_name, column_name, schema_name
        raise NotImplementedError

    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        _ = sql
        return []

    def get_table_grants(
        self, schema_filter: SchemaFilter | None = None
    ) -> list[TableGrant]:
        _ = schema_filter
        return list(self._grants)


def test_allowed_table_patterns_for_grants_groups_and_sorts() -> None:
    grants = [
        TableGrant(grantee="b", schema_name="public", table_name="z"),
        TableGrant(grantee="a", schema_name="public", table_name="t1"),
        TableGrant(grantee="a", schema_name="public", table_name="t2"),
    ]
    got = allowed_table_patterns_for_grants(grants, "mydb")
    assert list(got.keys()) == ["a", "b"]
    assert got["a"] == ["mydb::public.t1", "mydb::public.t2"]
    assert got["b"] == ["mydb::public.z"]


def test_merge_preserves_base_and_unrelated_profiles() -> None:
    existing = {
        "hidden_tables": ["secret.*"],
        "profiles": {
            "analyst": {"allowed_tables": ["old::public.old"]},
            "custom": {"hidden_tables": ["x.*"]},
        },
    }
    merged = merge_grant_profiles_into_visibility_doc(
        existing_root=existing,
        grantee_to_tables={"analyst": ["mydb::public.orders"]},
        roles_filter=None,
    )
    assert merged["hidden_tables"] == ["secret.*"]
    assert merged["profiles"]["custom"] == {"hidden_tables": ["x.*"]}
    assert merged["profiles"]["analyst"]["allowed_tables"] == ["mydb::public.orders"]


def test_merge_preserves_other_keys_on_grant_profile() -> None:
    existing = {
        "profiles": {
            "analyst": {
                "allowed_tables": ["x"],
                "hidden_columns": ["p.t.secret"],
            },
        },
    }
    merged = merge_grant_profiles_into_visibility_doc(
        existing_root=existing,
        grantee_to_tables={"analyst": ["mydb::public.a"]},
        roles_filter=None,
    )
    body = merged["profiles"]["analyst"]
    assert body["allowed_tables"] == ["mydb::public.a"]
    assert body["hidden_columns"] == ["p.t.secret"]


def test_merge_removes_stale_grant_only_profile_when_full_resync() -> None:
    existing = {
        "profiles": {
            "gone_role": {"allowed_tables": ["mydb::public.stale"]},
            "keeps_manual": {"allowed_tables": ["x"], "hidden_tables": ["y"]},
        },
    }
    merged = merge_grant_profiles_into_visibility_doc(
        existing_root=existing,
        grantee_to_tables={"analyst": ["mydb::public.t"]},
        roles_filter=None,
    )
    profiles = merged["profiles"]
    assert "gone_role" not in profiles
    assert "keeps_manual" in profiles
    assert profiles["analyst"]["allowed_tables"] == ["mydb::public.t"]


def test_merge_roles_filter_only_touches_named_roles() -> None:
    existing = {
        "profiles": {
            "analyst": {"allowed_tables": ["mydb::public.old"]},
            "reporter": {"allowed_tables": ["mydb::public.keep"]},
        },
    }
    merged = merge_grant_profiles_into_visibility_doc(
        existing_root=existing,
        grantee_to_tables={
            "analyst": ["mydb::public.a"],
            "reporter": ["mydb::public.b"],
        },
        roles_filter=frozenset({"analyst"}),
    )
    assert merged["profiles"]["analyst"]["allowed_tables"] == ["mydb::public.a"]
    assert merged["profiles"]["reporter"]["allowed_tables"] == ["mydb::public.keep"]


def test_run_sync_grants_writes_round_trip(tmp_path: Path) -> None:
    grants = [
        TableGrant(grantee="analyst", schema_name="public", table_name="orders"),
    ]
    conn = _StubGrantsConnector(grants)
    conn.connect()
    out = tmp_path / "visibility.yml"
    grantee_count = run_sync_grants(
        conn,
        output_path=out,
        connection_name="mydb",
        roles_filter=None,
    )
    assert grantee_count == 1
    cfg = load_visibility_config(out)
    assert cfg.profiles["analyst"].allowed_tables == ["mydb::public.orders"]
    loaded = YAML(typ="safe")
    data = loaded.load(out.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["profiles"]["analyst"]["allowed_tables"] == ["mydb::public.orders"]


def test_run_sync_grants_merges_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "visibility.yml"
    path.write_text(
        "hidden_tables:\n  - audit.*\nprofiles:\n  other:\n    hidden_tables: [z.*]\n",
        encoding="utf-8",
    )
    grants = [TableGrant(grantee="r1", schema_name="s", table_name="t")]
    conn = _StubGrantsConnector(grants)
    conn.connect()
    run_sync_grants(conn, output_path=path, connection_name="db", roles_filter=None)
    cfg = load_visibility_config(path)
    assert cfg.hidden_tables == ["audit.*"]
    assert "other" in cfg.profiles
    assert cfg.profiles["r1"].allowed_tables == ["db::s.t"]


def test_run_sync_grants_empty_grants_returns_zero_and_no_profiles(
    tmp_path: Path,
) -> None:
    """F4: connector with no grants yields grantee_count=0 and no ``profiles`` key.

    Uses the ``connection_name`` override path so the result mirrors what the
    CLI produces when --name is passed.
    """
    conn = _StubGrantsConnector([])
    conn.connect()
    out = tmp_path / "visibility.yml"
    grantee_count = run_sync_grants(
        conn,
        output_path=out,
        connection_name="prod_db",
        roles_filter=None,
    )
    assert grantee_count == 0
    assert out.exists()
    data = YAML(typ="safe").load(out.read_text(encoding="utf-8"))
    # Empty grants from scratch → file is just an empty mapping (no profiles key).
    assert data is None or "profiles" not in data


def test_run_sync_grants_uses_connection_name_override(tmp_path: Path) -> None:
    """F1: the ``connection_name`` argument flows into the emitted patterns.

    This mirrors the behavior of ``--name mydb``: regardless of the DSN, the
    grant patterns must use the supplied logical name so they match what
    ``pretensor index --name mydb`` wrote to the graph at index time.
    """
    grants = [TableGrant(grantee="analyst", schema_name="public", table_name="orders")]
    conn = _StubGrantsConnector(grants)
    conn.connect()
    out = tmp_path / "visibility.yml"
    run_sync_grants(
        conn,
        output_path=out,
        connection_name="custom_name",
        roles_filter=None,
    )
    cfg = load_visibility_config(out)
    assert cfg.profiles["analyst"].allowed_tables == ["custom_name::public.orders"]


def test_write_visibility_yaml_document_is_atomic_on_failure(tmp_path: Path) -> None:
    """F3: an interrupted write must leave the original file intact.

    Simulates a crash mid-write by making ``yaml.dump`` raise. The existing
    file on disk must be unchanged, and no partial ``.tmp`` file should linger.
    """
    from unittest.mock import patch

    from pretensor.visibility.sync_grants import write_visibility_yaml_document

    path = tmp_path / "visibility.yml"
    original = "hidden_tables:\n  - audit.*\n"
    path.write_text(original, encoding="utf-8")

    class _BoomYAML:
        default_flow_style = False

        def indent(self, **_: Any) -> None:
            return None

        def dump(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("simulated mid-write failure")

    with patch("pretensor.visibility.sync_grants.YAML", return_value=_BoomYAML()):
        try:
            write_visibility_yaml_document(path, {"profiles": {"r": {"allowed_tables": ["x"]}}})
        except RuntimeError:
            pass
        else:  # pragma: no cover - defensive
            raise AssertionError("expected RuntimeError to propagate")

    # Original file still intact.
    assert path.read_text(encoding="utf-8") == original
    # No leftover .tmp sibling.
    assert not (tmp_path / "visibility.yml.tmp").exists()
