"""Path helpers for CLI state directory layout."""

from __future__ import annotations

from pathlib import Path

from pretensor.cli import constants as c
from pretensor.introspection.models.config import DatabaseType
from pretensor.introspection.models.dsn import (
    connection_config_from_url,
    infer_database_type_from_dsn,
)


def default_connection_name(dsn: str) -> str:
    """Use the database name from the URL when present, else ``default``."""
    raw = dsn.strip()
    try:
        if infer_database_type_from_dsn(raw) == DatabaseType.BIGQUERY:
            cfg = connection_config_from_url(raw, "_tmp")
            logical_db = (cfg.database or "").strip()
            if "/" in logical_db:
                return logical_db.split("/", 1)[1]
            return logical_db or "default"
    except ValueError:
        pass
    cfg = connection_config_from_url(raw, "_tmp")
    if cfg.database:
        return cfg.database.replace("/", "_")
    return "default"


def graph_file_for_connection(state_dir: Path, connection_name: str) -> Path:
    safe = connection_name.replace("/", "_")
    return state_dir / c.GRAPHS_SUBDIR / f"{safe}.kuzu"


def unified_graph_path(state_dir: Path) -> Path:
    return state_dir / c.GRAPHS_SUBDIR / c.UNIFIED_GRAPH_BASENAME


def keystore_path(state_dir: Path) -> Path:
    return state_dir / "keystore"
