"""Postgres connector requires the optional ``postgres`` extra (psycopg2)."""

from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any

import pytest

import pretensor.connectors.registry as connector_registry
from pretensor.introspection.models.config import ConnectionConfig, DatabaseType


def test_importing_pretensor_does_not_require_psycopg2() -> None:
    """A bare install must import the package and the connector module."""
    # Importing the connector module must not pull psycopg2 at module load.
    mod = importlib.import_module("pretensor.connectors.postgres")
    assert hasattr(mod, "PostgresConnector")


def test_get_connector_postgres_without_driver_raises_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psycopg2" or name.startswith("psycopg2."):
            raise ImportError("No module named 'psycopg2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("psycopg2", None)

    cfg = ConnectionConfig(
        name="x",
        type=DatabaseType.POSTGRES,
        host="h",
        database="db",
        user="u",
        password="p",
    )
    with pytest.raises(ImportError, match=r"pretensor\[postgres\]"):
        connector_registry.get_connector(cfg)
