"""Tests for GraphRegistry."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pretensor.core.dsn_crypto import DSNEncryptor
from pretensor.core.registry import GraphRegistry


def test_registry_roundtrip(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg = GraphRegistry(reg_path)
    reg.upsert(
        connection_name="app",
        database="appdb",
        dsn="postgresql://localhost/appdb",
        graph_path=tmp_path / "app.kuzu",
        indexed_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )
    reg.save()

    loaded = GraphRegistry(reg_path).load()
    entry = loaded.get("app")
    assert entry is not None
    assert entry.database == "appdb"
    assert entry.dsn == "postgresql://localhost/appdb"
    assert entry.graph_path.endswith("app.kuzu")


def test_registry_encrypted_dsn_roundtrip(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    graph = tmp_path / "unified.kuzu"
    enc = DSNEncryptor(tmp_path / "keystore")
    reg = GraphRegistry(reg_path)
    dsn = "postgresql://user:secret@host/db"
    reg.upsert(
        connection_name="app",
        database="appdb",
        dsn=dsn,
        graph_path=graph,
        unified_graph_path=graph,
        encrypt_dsn=True,
        encryptor=enc,
        indexed_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )
    reg.save()
    loaded = GraphRegistry(reg_path).load().get("app")
    assert loaded is not None
    assert loaded.dsn_encrypted
    assert loaded.plaintext_dsn(enc) == dsn
