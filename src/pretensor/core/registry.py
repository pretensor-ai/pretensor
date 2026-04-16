"""JSON registry of indexed databases under ``.pretensor/``."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pretensor.core.dsn_crypto import DSNEncryptor

__all__ = [
    "GraphRegistry",
    "MultiDatabaseRegistry",
    "RegistryEntry",
    "DatabaseNotFoundError",
]

_REGISTRY_VERSION = 1


class DatabaseNotFoundError(KeyError):
    """Raised when a connection name is missing from the registry."""


class RegistryEntry(BaseModel):
    """One indexed connection."""

    model_config = ConfigDict(frozen=True)

    connection_name: str
    database: str
    dsn: str = ""
    dsn_encrypted: str | None = None
    graph_path: str
    unified_graph_path: str | None = None
    last_indexed_at: datetime
    # Persisted by index/add so reindex/detect_changes can parse stored DSNs.
    dialect: Literal["postgres", "mysql", "snowflake", "bigquery"] = "postgres"
    table_count: int | None = None
    description: str | None = None
    # Capability provenance for ``list_databases``: distinguishes "user didn't
    # run this" ("not_attempted") from "ran but found nothing" ("empty").
    dbt_manifest_path: str | None = None
    llm_enrichment_ran: bool = False

    def plaintext_dsn(self, encryptor: DSNEncryptor | None) -> str:
        """Return the connection URL, decrypting when ``dsn_encrypted`` is set."""
        if self.dsn_encrypted:
            if encryptor is None:
                raise RuntimeError(
                    "DSNEncryptor required for encrypted registry entries"
                )
            return encryptor.decrypt(self.dsn_encrypted)
        return self.dsn


class _RegistryFile(BaseModel):
    version: int = _REGISTRY_VERSION
    entries: dict[str, RegistryEntry] = Field(default_factory=dict)


class GraphRegistry:
    """Load and save ``registry.json`` tracking graph files per connection."""

    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path
        self._data = _RegistryFile()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> GraphRegistry:
        """Load from disk if the file exists; otherwise start empty."""
        if not self._path.exists():
            return self
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "entries" not in raw and "version" not in raw:
            raw = {"version": _REGISTRY_VERSION, "entries": raw}
        self._data = _RegistryFile.model_validate(raw)
        return self

    def get(self, connection_name: str) -> RegistryEntry | None:
        return self._data.entries.get(connection_name)

    def require(self, connection_name: str) -> RegistryEntry:
        """Return an entry or raise :class:`DatabaseNotFoundError`."""
        entry = self.get(connection_name)
        if entry is None:
            raise DatabaseNotFoundError(
                f"No registry entry for connection {connection_name!r}. "
                "Use `pretensor list` or `pretensor add`."
            )
        return entry

    def list_entries(self) -> list[RegistryEntry]:
        """Return all entries sorted by connection name."""
        return sorted(self._data.entries.values(), key=lambda e: e.connection_name)

    def remove(self, connection_name: str) -> None:
        """Drop a connection from the in-memory registry (call :meth:`save`)."""
        entries = dict(self._data.entries)
        entries.pop(connection_name, None)
        self._data = _RegistryFile(version=self._data.version, entries=entries)

    def upsert(
        self,
        *,
        connection_name: str,
        database: str,
        dsn: str,
        graph_path: Path,
        indexed_at: datetime | None = None,
        encrypt_dsn: bool = False,
        encryptor: DSNEncryptor | None = None,
        unified_graph_path: Path | None = None,
        dialect: Literal["postgres", "mysql", "snowflake", "bigquery"] = "postgres",
        table_count: int | None = None,
        description: str | None = None,
        dbt_manifest_path: str | None = None,
        llm_enrichment_ran: bool = False,
    ) -> None:
        """Record or replace an entry for ``connection_name``."""
        when = indexed_at or datetime.now(timezone.utc)
        dsn_plain = ""
        dsn_enc: str | None = None
        if encrypt_dsn:
            if encryptor is None:
                raise ValueError("encryptor is required when encrypt_dsn is True")
            dsn_enc = encryptor.encrypt(dsn)
        else:
            dsn_plain = dsn
        uni = str(unified_graph_path.resolve()) if unified_graph_path else None
        entry = RegistryEntry(
            connection_name=connection_name,
            database=database,
            dsn=dsn_plain,
            dsn_encrypted=dsn_enc,
            graph_path=str(graph_path.resolve()),
            unified_graph_path=uni,
            last_indexed_at=when,
            dialect=dialect,
            table_count=table_count,
            description=description,
            dbt_manifest_path=dbt_manifest_path,
            llm_enrichment_ran=llm_enrichment_ran,
        )
        entries = dict(self._data.entries)
        entries[connection_name] = entry
        self._data = _RegistryFile(version=_REGISTRY_VERSION, entries=entries)

    def update_indexed_metadata(
        self,
        connection_name: str,
        *,
        table_count: int,
        indexed_at: datetime | None = None,
    ) -> None:
        """Refresh ``table_count`` and optional timestamp after indexing."""
        cur = self.require(connection_name)
        when = indexed_at or datetime.now(timezone.utc)
        entries = dict(self._data.entries)
        entries[connection_name] = RegistryEntry(
            connection_name=cur.connection_name,
            database=cur.database,
            dsn=cur.dsn,
            dsn_encrypted=cur.dsn_encrypted,
            graph_path=cur.graph_path,
            unified_graph_path=cur.unified_graph_path,
            last_indexed_at=when,
            dialect=cur.dialect,
            table_count=table_count,
            description=cur.description,
            dbt_manifest_path=cur.dbt_manifest_path,
            llm_enrichment_ran=cur.llm_enrichment_ran,
        )
        self._data = _RegistryFile(version=self._data.version, entries=entries)

    def unified_graph_file(self) -> Path | None:
        """Return the shared Kuzu path when all entries agree, else ``None``."""
        entries = self.list_entries()
        if not entries:
            return None
        paths = {e.unified_graph_path or e.graph_path for e in entries}
        if len(paths) != 1:
            return None
        return Path(next(iter(paths)))

    def save(self) -> None:
        """Write registry atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = self._data.model_dump(mode="json")
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)


class MultiDatabaseRegistry(GraphRegistry):
    """Alias for the multi-connection registry (Phase 3 unified graph)."""

    pass
