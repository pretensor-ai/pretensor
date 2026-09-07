"""Tests for the gitignore-aware repository walker."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from pretensor.enrichment.analyze.walker import SKIP_DIRS, walk_repo


def _collect(
    repo: Path,
    *,
    includes: Sequence[str] = (),
    excludes: Sequence[str] = (),
    max_file_bytes: int = 1_000_000,
) -> list[str]:
    """Collect relative path strings from walk_repo."""
    return sorted(
        str(p.relative_to(repo))
        for p, _lang in walk_repo(
            repo,
            includes=includes,
            excludes=excludes,
            max_file_bytes=max_file_bytes,
        )
    )


def _write(path: Path, content: str = "# placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_basic_python_file_is_yielded(tmp_path: Path) -> None:
    _write(tmp_path / "main.py")
    result = _collect(tmp_path)
    assert result == ["main.py"]


def test_non_python_file_is_not_yielded(tmp_path: Path) -> None:
    _write(tmp_path / "config.yaml")
    _write(tmp_path / "notes.txt")
    _write(tmp_path / "app.py")
    result = _collect(tmp_path)
    assert result == ["app.py"]


def test_sql_file_is_yielded_with_sql_language(tmp_path: Path) -> None:
    _write(tmp_path / "report.sql", "SELECT id FROM orders;\n")
    results = list(walk_repo(tmp_path))
    assert len(results) == 1
    path, lang = results[0]
    assert path == tmp_path / "report.sql"
    assert lang == "sql"


@pytest.mark.parametrize("skip_dir", sorted(SKIP_DIRS))
def test_skip_dirs_are_pruned(tmp_path: Path, skip_dir: str) -> None:
    _write(tmp_path / skip_dir / "code.py")
    _write(tmp_path / "app.py")
    result = _collect(tmp_path)
    assert result == ["app.py"]
    assert not any(skip_dir in r for r in result)


def test_gitignore_excludes_file(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("secrets.py\n", encoding="utf-8")
    _write(tmp_path / "secrets.py")
    _write(tmp_path / "main.py")
    result = _collect(tmp_path)
    assert result == ["main.py"]


def test_gitignore_excludes_directory(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("generated/\n", encoding="utf-8")
    _write(tmp_path / "generated" / "output.py")
    _write(tmp_path / "src" / "app.py")
    result = _collect(tmp_path)
    assert result == [str(Path("src") / "app.py")]


def test_nested_gitignore_excludes_file_in_its_dir(tmp_path: Path) -> None:
    _write(tmp_path / "sub" / ".gitignore", "secret.py\n")
    _write(tmp_path / "sub" / "secret.py")
    _write(tmp_path / "sub" / "app.py")
    _write(tmp_path / "secret.py")  # outside the nested file's scope
    result = _collect(tmp_path)
    assert result == ["secret.py", str(Path("sub") / "app.py")]


def test_nested_gitignore_patterns_are_relative_to_their_dir(tmp_path: Path) -> None:
    _write(tmp_path / "sub" / ".gitignore", "generated/\n")
    _write(tmp_path / "sub" / "generated" / "output.py")
    _write(tmp_path / "generated" / "kept.py")
    result = _collect(tmp_path)
    assert result == [str(Path("generated") / "kept.py")]


def test_nested_negation_reincludes_at_its_level(tmp_path: Path) -> None:
    _write(tmp_path / "sub" / ".gitignore", "*.py\n!keep.py\n")
    _write(tmp_path / "sub" / "keep.py")
    _write(tmp_path / "sub" / "drop.py")
    _write(tmp_path / "main.py")
    result = _collect(tmp_path)
    assert result == ["main.py", str(Path("sub") / "keep.py")]


def test_nested_negation_overrides_root_ignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("secrets.py\n", encoding="utf-8")
    _write(tmp_path / "sub" / ".gitignore", "!secrets.py\n")
    _write(tmp_path / "sub" / "secrets.py")
    _write(tmp_path / "secrets.py")
    result = _collect(tmp_path)
    assert result == [str(Path("sub") / "secrets.py")]


def test_deeper_gitignore_takes_precedence_over_shallower(tmp_path: Path) -> None:
    _write(tmp_path / "sub" / ".gitignore", "!gen.py\n")
    _write(tmp_path / "sub" / "deep" / ".gitignore", "gen.py\n")
    (tmp_path / ".gitignore").write_text("gen.py\n", encoding="utf-8")
    _write(tmp_path / "sub" / "gen.py")  # re-included by sub/.gitignore
    _write(tmp_path / "sub" / "deep" / "gen.py")  # re-ignored by deep/.gitignore
    _write(tmp_path / "gen.py")  # ignored by root
    result = _collect(tmp_path)
    assert result == [str(Path("sub") / "gen.py")]


def test_negation_cannot_reinclude_inside_ignored_dir(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("generated/\n", encoding="utf-8")
    _write(tmp_path / "generated" / ".gitignore", "!keep.py\n")
    _write(tmp_path / "generated" / "keep.py")
    _write(tmp_path / "main.py")
    result = _collect(tmp_path)
    assert result == ["main.py"]


def test_file_size_cap(tmp_path: Path) -> None:
    large = tmp_path / "big.py"
    large.write_bytes(b"x" * 100)
    _write(tmp_path / "small.py")
    result = _collect(tmp_path, max_file_bytes=50)
    assert result == ["small.py"]


def test_excludes_pattern(tmp_path: Path) -> None:
    _write(tmp_path / "tests" / "test_foo.py")
    _write(tmp_path / "src" / "app.py")
    result = _collect(tmp_path, excludes=["tests/**"])
    assert result == [str(Path("src") / "app.py")]


def test_includes_pattern_filters(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "app.py")
    _write(tmp_path / "tools" / "helper.py")
    result = _collect(tmp_path, includes=["src/**"])
    assert result == [str(Path("src") / "app.py")]


def test_nested_python_files_found(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "b" / "c.py")
    _write(tmp_path / "d.py")
    result = _collect(tmp_path)
    assert str(Path("a") / "b" / "c.py") in result
    assert "d.py" in result


def test_empty_repo_yields_nothing(tmp_path: Path) -> None:
    assert _collect(tmp_path) == []


def test_no_gitignore_does_not_crash(tmp_path: Path) -> None:
    _write(tmp_path / "main.py")
    result = _collect(tmp_path)
    assert result == ["main.py"]
