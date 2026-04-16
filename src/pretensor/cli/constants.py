"""Shared CLI literals (state dir, registry filenames, graph layout)."""

from __future__ import annotations

from pathlib import Path

DEFAULT_STATE_DIR = Path(".pretensor")
GRAPHS_SUBDIR = "graphs"
REGISTRY_FILENAME = "registry.json"
UNIFIED_GRAPH_BASENAME = "unified.kuzu"
