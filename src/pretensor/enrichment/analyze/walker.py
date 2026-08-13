"""Gitignore-aware repository file walker for the analyze enrichment pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import pathspec

__all__ = ["walk_repo", "SKIP_DIRS"]

logger = logging.getLogger(__name__)

SKIP_DIRS: frozenset[str] = frozenset(
    {"node_modules", ".venv", "dist", "build", ".tox", ".git"}
)

_LANG_MAP: dict[str, str] = {".py": "python"}


def walk_repo(
    repo_path: Path,
    *,
    includes: Sequence[str] = (),
    excludes: Sequence[str] = (),
    max_file_bytes: int = 1_000_000,
) -> Iterator[tuple[Path, str]]:
    """Yield ``(absolute_file_path, language)`` for each candidate source file.

    Respects ``.gitignore`` in ``repo_path``, prunes ``SKIP_DIRS``, skips files
    exceeding ``max_file_bytes``, and filters by ``includes``/``excludes`` glob
    patterns when provided.
    """
    ignore_spec = _load_gitignore_spec(repo_path)
    include_spec = (
        pathspec.PathSpec.from_lines("gitignore", includes) if includes else None
    )
    exclude_spec = (
        pathspec.PathSpec.from_lines("gitignore", excludes) if excludes else None
    )

    for file_path in _iter_files(repo_path):
        rel = file_path.relative_to(repo_path)
        rel_str = str(rel)

        if ignore_spec and ignore_spec.match_file(rel_str):
            continue

        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        if size > max_file_bytes:
            logger.debug("walker: skipping large file %s (%d bytes)", rel, size)
            continue

        lang = _LANG_MAP.get(file_path.suffix)
        if lang is None:
            continue

        if include_spec and not include_spec.match_file(rel_str):
            continue
        if exclude_spec and exclude_spec.match_file(rel_str):
            continue

        yield file_path, lang


def _iter_files(directory: Path) -> Iterator[Path]:
    """Recursively yield files, pruning SKIP_DIRS directories early."""
    try:
        entries = list(directory.iterdir())
    except PermissionError:
        return
    for entry in entries:
        if entry.is_symlink():
            # Never follow symlinks: a link cycle would recurse forever, and a
            # link out of the repo would break repo-relative path invariants.
            continue
        if entry.is_dir():
            if entry.name in SKIP_DIRS:
                continue
            yield from _iter_files(entry)
        elif entry.is_file():
            yield entry


def _load_gitignore_spec(repo_root: Path) -> pathspec.PathSpec | None:
    """Load ``.gitignore`` from the repo root into a PathSpec (best-effort)."""
    gitignore = repo_root / ".gitignore"
    if not gitignore.is_file():
        return None
    try:
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return pathspec.PathSpec.from_lines("gitignore", lines)
