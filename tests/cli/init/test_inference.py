"""Tests for wizard inference helpers."""

from __future__ import annotations

from pathlib import Path

from pretensor.cli.init.inference import (
    count_supported_files,
    infer_dsn_from_env,
    infer_repo,
)
from pretensor.enrichment.analyze.walker import supported_extensions


def test_supported_extensions_reports_python() -> None:
    assert ".py" in supported_extensions()


def test_infer_dsn_prefers_database_url() -> None:
    found = infer_dsn_from_env({"DATABASE_URL": "postgresql://u:p@h:5432/db"})
    assert found is not None
    assert found.dsn == "postgresql://u:p@h:5432/db"
    assert found.env_var == "DATABASE_URL"


def test_infer_dsn_assembles_pg_variables() -> None:
    found = infer_dsn_from_env(
        {
            "PGHOST": "h",
            "PGPORT": "5432",
            "PGUSER": "u",
            "PGPASSWORD": "p",
            "PGDATABASE": "db",
        }
    )
    assert found is not None
    assert found.dsn == "postgresql://u:p@h:5432/db"
    assert found.env_var is None


def test_infer_dsn_percent_encodes_pg_password_with_special_chars() -> None:
    found = infer_dsn_from_env(
        {
            "PGHOST": "h",
            "PGPORT": "5432",
            "PGUSER": "u",
            "PGPASSWORD": "p@ss",
            "PGDATABASE": "db",
        }
    )
    assert found is not None
    assert found.dsn == "postgresql://u:p%40ss@h:5432/db"
    assert found.env_var is None


def test_infer_dsn_returns_none_when_nothing_set() -> None:
    assert infer_dsn_from_env({}) is None


def test_count_supported_files_counts_python_only(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2", encoding="utf-8")
    (tmp_path / "c.ts").write_text("const z = 3", encoding="utf-8")
    assert count_supported_files(tmp_path) == 2


def test_infer_repo_returns_none_without_supported_files(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "index.ts").write_text("const z = 3", encoding="utf-8")
    assert infer_repo(tmp_path) is None


def test_infer_repo_returns_git_root_with_python(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
    found = infer_repo(tmp_path)
    assert found is not None
    assert found.path == tmp_path
    assert found.file_count == 1
