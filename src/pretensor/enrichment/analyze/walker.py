"""Gitignore-aware repository file walker for the analyze enrichment pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import pathspec

__all__ = ["walk_repo", "SKIP_DIRS", "supported_extensions"]

logger = logging.getLogger(__name__)

SKIP_DIRS: frozenset[str] = frozenset(
    {"node_modules", ".venv", "dist", "build", ".tox", ".git"}
)

_LANG_MAP: dict[str, str] = {".py": "python", ".sql": "sql"}

# Stack of (directory, spec) pairs, ordered root -> deepest. Each spec's
# patterns match paths relative to its own directory, mirroring git semantics.
_SpecStack = tuple[tuple[Path, pathspec.PathSpec], ...]


def walk_repo(
    repo_path: Path,
    *,
    includes: Sequence[str] = (),
    excludes: Sequence[str] = (),
    max_file_bytes: int = 1_000_000,
) -> Iterator[tuple[Path, str]]:
    """Yield ``(absolute_file_path, language)`` for each candidate source file.

    Respects ``.gitignore`` files throughout the tree — patterns in a nested
    ``.gitignore`` match relative to that file's directory, deeper files take
    precedence over shallower ones, and negations (``!keep.py``) re-include at
    their level, per git's resolution rules. Prunes ``SKIP_DIRS``, skips files
    exceeding ``max_file_bytes``, and filters by ``includes``/``excludes`` glob
    patterns when provided.
    """
    include_spec = (
        pathspec.PathSpec.from_lines("gitignore", includes) if includes else None
    )
    exclude_spec = (
        pathspec.PathSpec.from_lines("gitignore", excludes) if excludes else None
    )

    for file_path in _iter_files(repo_path):
        rel = file_path.relative_to(repo_path)
        rel_str = str(rel)

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


def supported_extensions() -> tuple[str, ...]:
    """File extensions the analyze walker currently understands."""
    return tuple(sorted(_LANG_MAP))


def _iter_files(directory: Path, spec_stack: _SpecStack = ()) -> Iterator[Path]:
    """Recursively yield non-gitignored files, pruning SKIP_DIRS directories early."""
    spec = _load_gitignore_spec(directory)
    if spec is not None:
        spec_stack = (*spec_stack, (directory, spec))
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
            if _is_ignored(entry, spec_stack, is_dir=True):
                # Pruning here also matches git: nothing inside an ignored
                # directory can be re-included by a deeper negation.
                continue
            yield from _iter_files(entry, spec_stack)
        elif entry.is_file():
            if _is_ignored(entry, spec_stack, is_dir=False):
                continue
            yield entry


def _is_ignored(path: Path, spec_stack: _SpecStack, *, is_dir: bool) -> bool:
    """Resolve the gitignore decision for ``path`` against the spec stack.

    Specs are consulted root -> deepest; the deepest spec with an opinion
    (a matching pattern, negated or not) wins, per git's precedence rules.
    """
    ignored = False
    for base, spec in spec_stack:
        rel = path.relative_to(base).as_posix()
        if is_dir:
            # Trailing slash lets directory-only patterns ("generated/") match.
            rel += "/"
        decision = spec.check_file(rel).include
        if decision is not None:
            ignored = decision
    return ignored


def _load_gitignore_spec(directory: Path) -> pathspec.PathSpec | None:
    """Load ``.gitignore`` from ``directory`` into a PathSpec (best-effort)."""
    gitignore = directory / ".gitignore"
    if not gitignore.is_file():
        return None
    try:
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return pathspec.PathSpec.from_lines("gitignore", lines)
