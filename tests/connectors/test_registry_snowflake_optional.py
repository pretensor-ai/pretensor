"""Snowflake connector requires the optional ``snowflake-sqlalchemy`` extra."""

from __future__ import annotations

import builtins
import sys
from typing import Any

import pytest

import pretensor.connectors.registry as connector_registry
from pretensor.introspection.models.config import ConnectionConfig, DatabaseType


def test_get_connector_snowflake_without_driver_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "snowflake.sqlalchemy" or name.startswith("snowflake.sqlalchemy."):
            raise ImportError("No module named 'snowflake.sqlalchemy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Ensure a cached import does not bypass our fake (fresh check uses __import__).
    sys.modules.pop("snowflake.sqlalchemy", None)

    cfg = ConnectionConfig(
        name="x",
        type=DatabaseType.SNOWFLAKE,
        host="acct",
        database="DB",
    )
    with pytest.raises(ImportError, match="pip install"):
        connector_registry.get_connector(cfg)
