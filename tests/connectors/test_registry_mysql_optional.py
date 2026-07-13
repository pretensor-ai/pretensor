"""MySQL connector requires the optional ``PyMySQL`` extra."""

from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest

import pretensor.connectors.registry as connector_registry
from pretensor.introspection.models.config import ConnectionConfig, DatabaseType


def test_get_connector_mysql_without_driver_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pymysql" or name.startswith("pymysql."):
            raise ImportError("No module named 'pymysql'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("pymysql", None)

    cfg = ConnectionConfig(
        name="x",
        type=DatabaseType.MYSQL,
        host="localhost",
        database="sakila",
    )
    with pytest.raises(ImportError, match="pip install"):
        connector_registry.get_connector(cfg)
