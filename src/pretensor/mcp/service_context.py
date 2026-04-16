"""Per-process MCP server context (visibility filter bound to ``graph_dir``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pretensor.config import GraphConfig, PretensorConfig
from pretensor.search.base import BaseSearchIndex
from pretensor.search.index import KeywordSearchIndex
from pretensor.visibility.filter import VisibilityFilter
from pretensor.visibility.runtime import load_visibility_filter_for_graph_dir

__all__ = [
    "DEFAULT_GRAPH_CONFIG",
    "ServerContext",
    "build_server_context",
    "get_effective_graph_config",
    "get_effective_search_index_cls",
    "get_effective_visibility_filter",
    "get_server_context",
    "reset_server_context",
    "set_server_context",
]

DEFAULT_GRAPH_CONFIG = GraphConfig()


@dataclass(frozen=True, slots=True)
class ServerContext:
    """Holds graph workspace path and merged visibility rules for tool handlers."""

    graph_dir: Path
    visibility_filter: VisibilityFilter
    config: PretensorConfig = field(default_factory=PretensorConfig)

    @property
    def graph_config(self) -> GraphConfig:
        """Backward-compatible access to graph tuning config."""
        return self.config.graph

    @property
    def search_index_cls(self) -> type[BaseSearchIndex]:
        """Backward-compatible access to configured search index class."""
        return self.config.search_index_cls


_ctx: ServerContext | None = None


def set_server_context(ctx: ServerContext) -> None:
    """Install context for the current MCP server process (stdio)."""
    global _ctx
    _ctx = ctx


def reset_server_context() -> None:
    """Clear the active server context.

    Intended for test teardown only — production servers never need to call this.
    """
    global _ctx
    _ctx = None


def get_server_context() -> ServerContext:
    """Return the active server context (must be set before handling tools)."""
    if _ctx is None:
        raise RuntimeError("MCP server context not initialized")
    return _ctx


def build_server_context(
    graph_dir: Path,
    *,
    visibility_path: Path | None = None,
    profile: str | None = None,
    config: PretensorConfig | None = None,
) -> ServerContext:
    """Resolve ``graph_dir`` and load visibility for one server lifetime."""
    resolved = graph_dir.resolve()
    vf = load_visibility_filter_for_graph_dir(
        resolved,
        visibility_path=visibility_path,
        profile=profile,
    )
    effective_config = config or PretensorConfig()
    return ServerContext(
        graph_dir=resolved,
        visibility_filter=vf,
        config=effective_config,
    )


def get_effective_visibility_filter(
    explicit: VisibilityFilter | None = None,
) -> VisibilityFilter | None:
    """Return an explicit filter, the server context filter, or None (no MCP server)."""
    if explicit is not None:
        return explicit
    try:
        return get_server_context().visibility_filter
    except RuntimeError:
        return None


def get_effective_graph_config(explicit: GraphConfig | None = None) -> GraphConfig:
    """Return explicit graph config or server context fallback."""
    if explicit is not None:
        return explicit
    try:
        return get_server_context().config.graph
    except RuntimeError:
        return DEFAULT_GRAPH_CONFIG


def get_effective_search_index_cls(
    explicit: type[BaseSearchIndex] | None = None,
) -> type[BaseSearchIndex]:
    """Return an explicit class, context class, or the OSS default."""
    if explicit is not None:
        return explicit
    try:
        return get_server_context().config.search_index_cls
    except RuntimeError:
        return KeywordSearchIndex
