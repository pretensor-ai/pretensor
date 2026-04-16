"""Load merged visibility rules for a graph state directory."""

from __future__ import annotations

from pathlib import Path

from pretensor.visibility.config import (
    default_visibility_path,
    load_visibility_config,
    merge_profile_into_base,
)
from pretensor.visibility.filter import VisibilityFilter

__all__ = ["load_visibility_filter_for_graph_dir"]


def load_visibility_filter_for_graph_dir(
    graph_dir: Path,
    *,
    visibility_path: Path | None = None,
    profile: str | None = None,
) -> VisibilityFilter:
    """Load ``visibility.yml`` and return a :class:`VisibilityFilter`.

    Args:
        graph_dir: State directory (``--graph-dir`` / ``.pretensor``).
        visibility_path: Optional explicit file; defaults to ``graph_dir/visibility.yml``.
        profile: Optional named profile merged onto the base section.

    Returns:
        Filter instance (allows all tables when no file or empty rules).
    """
    path = visibility_path if visibility_path is not None else default_visibility_path(graph_dir)
    base = load_visibility_config(path)
    merged = merge_profile_into_base(base, profile)
    return VisibilityFilter.from_config(merged)
