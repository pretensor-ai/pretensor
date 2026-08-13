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
