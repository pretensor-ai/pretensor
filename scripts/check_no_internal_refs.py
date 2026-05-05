#!/usr/bin/env python3
"""Block internal tracker references in tracked content.

Fails if a line contains an issue ID of the form ``<prefix>-<digits>`` that
matches a known internal tracker, or a tracker hostname. Lines carrying a
``# allowlist: <reason>`` marker are skipped so legitimate historical
mentions stay visible in diffs rather than vanishing behind ``--no-verify``.

Invocation:

    python3 scripts/check_no_internal_refs.py <file> [<file> ...]

Exit code is 0 when clean, 1 when any offending line is found.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections.abc import Iterable

# Patterns that identify internal tracker references. Kept deliberately
# narrow to avoid false positives on unrelated identifiers like ``POST-``
# response codes or Apache ``LINEAR.APP`` constants.
_ISSUE_ID = re.compile(r"\b(?:PRE)-\d+\b")
_TRACKER_URL = re.compile(r"\blinear\.app\b")
_PATTERNS: tuple[re.Pattern[str], ...] = (_ISSUE_ID, _TRACKER_URL)

_ALLOWLIST = re.compile(r"allowlist:\s*\S+")

# Extensions we never inspect — binary or auto-generated content.
_SKIP_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".whl",
        ".woff",
        ".woff2",
        ".ttf",
        ".lock",
    }
)

# Paths that are allowed to contain the patterns by virtue of being the
# enforcement itself. The check file lists the patterns it matches.
_SELF_PATHS = frozenset(
    {
        "scripts/check_no_internal_refs.py",
    }
)


def _iter_offenses(path: str, text: str) -> Iterable[tuple[int, str]]:
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _ALLOWLIST.search(line):
            continue
        for pat in _PATTERNS:
            match = pat.search(line)
            if match:
                yield lineno, match.group(0)
                break


def _check_file(path: str) -> list[str]:
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in _SKIP_SUFFIXES:
        return []
    if path in _SELF_PATHS:
        return []
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, IsADirectoryError):
        return []
    return [
        f"{path}:{lineno}: internal tracker reference {token!r}"
        for lineno, token in _iter_offenses(path, text)
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="files to inspect")
    args = parser.parse_args(argv)

    errors: list[str] = []
    for path in args.paths:
        errors.extend(_check_file(path))

    if not errors:
        return 0

    print("Internal tracker references found:", file=sys.stderr)
    for err in errors:
        print(f"  {err}", file=sys.stderr)
    print(
        "\nRephrase the reference in plain language, or mark the line with "
        "`# allowlist: <reason>` if the mention is intentional.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
