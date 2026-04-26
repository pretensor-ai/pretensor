#!/usr/bin/env python3
"""Fail if any URL in README.md is relative.

PyPI renders the README from the wheel/sdist long_description. Relative
URLs like ``[text](docs/foo.md)`` 404 there because PyPI cannot resolve
them against a repo. Every URL has to be absolute (``http(s)://``), an
in-page fragment (``#anchor``), or a ``mailto:`` link.

Only README.md is checked. CHANGELOG.md, CONTRIBUTING.md, SECURITY.md,
etc. render fine on GitHub but do not reach PyPI, so relative paths
between them are conventional and harmless.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ALLOWED_PREFIXES = ("http://", "https://", "#", "mailto:")
README = Path("README.md")

_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def find_relative_urls(content: str) -> list[tuple[int, str, str]]:
    errors: list[tuple[int, str, str]] = []

    for m in _IMAGE_PATTERN.finditer(content):
        url = m.group(2)
        if not url.startswith(ALLOWED_PREFIXES):
            line = content[: m.start()].count("\n") + 1
            errors.append((line, "image", url))

    # Strip image syntax before scanning for links — otherwise the badge
    # pattern ``[![alt](image-url)](target-url)`` confuses the link regex
    # because the inner ``]`` closes the outer ``[...]`` prematurely.
    cleaned = _IMAGE_PATTERN.sub("__IMG__", content)

    for m in _LINK_PATTERN.finditer(cleaned):
        url = m.group(2)
        if not url.startswith(ALLOWED_PREFIXES):
            line = cleaned[: m.start()].count("\n") + 1
            errors.append((line, "link", url))

    return errors


def main() -> int:
    if not README.exists():
        print(f"error: {README} not found (run from repo root)", file=sys.stderr)
        return 1
    errors = find_relative_urls(README.read_text())
    if errors:
        print(
            f"{README}: {len(errors)} relative URL(s) found.\n"
            "PyPI renders README.md from the wheel/sdist; every URL must "
            "be absolute (http://..., https://...), an in-page anchor "
            "(#section), or a mailto: link.",
            file=sys.stderr,
        )
        for line, kind, url in errors:
            print(f"  L{line}: {kind}: {url}", file=sys.stderr)
        return 1
    print(f"{README}: URL check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
