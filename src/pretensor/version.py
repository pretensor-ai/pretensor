"""Single source for the installed pretensor package version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["package_version"]


def package_version(fallback: str = "unknown") -> str:
    """Return the installed ``pretensor`` distribution version.

    Args:
        fallback: Value returned when pretensor is not installed as a
            distribution (e.g. running from a bare source checkout).
    """
    try:
        return version("pretensor")
    except PackageNotFoundError:
        return fallback
