"""Unit tests for CLI path helpers (pretensor.cli.paths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pretensor.cli.constants import (
    DEFAULT_STATE_DIR,
    GRAPHS_SUBDIR,
    REGISTRY_FILENAME,
    UNIFIED_GRAPH_BASENAME,
)
from pretensor.cli.paths import (
    default_connection_name,
    graph_file_for_connection,
    keystore_path,
    unified_graph_path,
)


class TestDefaultConnectionName:
    def test_postgres_uses_database_name(self) -> None:
        assert default_connection_name("postgresql://u:p@h/mydb") == "mydb"

    def test_postgres_scheme_also_works(self) -> None:
        assert default_connection_name("postgres://u:p@h/shopdb") == "shopdb"

    def test_snowflake_uses_database_name(self) -> None:
        assert default_connection_name("snowflake://u:p@acct/warehouse/schema") == "warehouse"

    def test_no_database_returns_default(self) -> None:
        result = default_connection_name("postgresql://u:p@h/")
        assert result == "default"

    def test_strips_whitespace_from_dsn(self) -> None:
        result = default_connection_name("  postgresql://u:p@h/mydb  ")
        assert result == "mydb"

    def test_invalid_dsn_raises_value_error(self) -> None:
        """An invalid DSN (no scheme) raises ValueError from the URL parser."""
        with pytest.raises(ValueError):
            default_connection_name("not-a-dsn")

    def test_slash_in_database_replaced(self) -> None:
        """Slashes in database names are replaced with underscores."""
        result = default_connection_name("postgresql://u:p@h/schema/db")
        assert "/" not in result

    def test_bigquery_dsn_with_slash_uses_dataset(self) -> None:
        """BigQuery DSNs with ``project/dataset`` return the dataset part."""
        result = default_connection_name("bigquery://project/dataset")
        assert result == "dataset"

    def test_bigquery_dsn_without_slash_uses_database(self) -> None:
        """BigQuery DSNs without a slash in the database return the database name."""
        result = default_connection_name("bigquery://project/myproject")
        assert result == "myproject"


class TestGraphFileForConnection:
    def test_returns_path_under_graphs_subdir(self, tmp_path: Path) -> None:
        result = graph_file_for_connection(tmp_path, "mydb")
        assert result == tmp_path / GRAPHS_SUBDIR / "mydb.kuzu"

    def test_slash_in_name_replaced(self, tmp_path: Path) -> None:
        result = graph_file_for_connection(tmp_path, "project/dataset")
        assert "/" not in str(result.name)
        assert "_" in str(result.name)

    def test_connection_name_preserved(self, tmp_path: Path) -> None:
        result = graph_file_for_connection(tmp_path, "warehouse")
        assert "warehouse" in result.name


class TestUnifiedGraphPath:
    def test_returns_unified_kuzu_file(self, tmp_path: Path) -> None:
        result = unified_graph_path(tmp_path)
        assert result == tmp_path / GRAPHS_SUBDIR / UNIFIED_GRAPH_BASENAME
        assert result.name == "unified.kuzu"


class TestKeystorePath:
    def test_returns_keystore_under_state_dir(self, tmp_path: Path) -> None:
        result = keystore_path(tmp_path)
        assert result == tmp_path / "keystore"


class TestConstants:
    def test_default_state_dir_is_pretensor(self) -> None:
        assert DEFAULT_STATE_DIR == Path(".pretensor")

    def test_graphs_subdir_name(self) -> None:
        assert GRAPHS_SUBDIR == "graphs"

    def test_registry_filename(self) -> None:
        assert REGISTRY_FILENAME == "registry.json"

    def test_unified_graph_basename(self) -> None:
        assert UNIFIED_GRAPH_BASENAME == "unified.kuzu"
