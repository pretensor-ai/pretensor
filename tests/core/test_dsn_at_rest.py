"""Regression tests for DSN-at-rest hardening.

Covers: default-on encryption (ciphertext in registry.json, not the plaintext
password), 0o600 keystore + registry permissions, and the redaction helper.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from pretensor.core.dsn_crypto import DSNEncryptor
from pretensor.core.registry import GraphRegistry
from pretensor.core.secure_io import atomic_write_text, secure_mkdir
from pretensor.introspection.models.dsn import redact_dsn

_POSIX = os.name == "posix"
_SECRET = "sup3r-s3cret-pw"
_DSN = f"postgresql://admin:{_SECRET}@db.internal:5432/prod"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_default_on_encryption_stores_ciphertext(tmp_path: Path) -> None:
    """Mirrors what `index` now does: encrypt_dsn=True on a fresh state dir."""
    keystore = tmp_path / "keystore"
    enc = DSNEncryptor(keystore)
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="prod",
        database="prod",
        dsn=_DSN,
        graph_path=tmp_path / "graphs" / "prod.kuzu",
        encrypt_dsn=True,
        encryptor=enc,
    )
    reg.save()

    raw = (tmp_path / "registry.json").read_text(encoding="utf-8")
    # The plaintext password must not appear anywhere in the registry file.
    assert _SECRET not in raw
    # The entry round-trips back to the original DSN via the keystore.
    reloaded = GraphRegistry(tmp_path / "registry.json").load()
    entry = reloaded.list_entries()[0]
    assert entry.dsn_encrypted
    assert entry.plaintext_dsn(DSNEncryptor(keystore)) == _DSN


@pytest.mark.skipif(not _POSIX, reason="POSIX file modes only")
def test_keystore_and_registry_are_owner_only(tmp_path: Path) -> None:
    keystore = tmp_path / "keystore"
    enc = DSNEncryptor(keystore)
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="prod",
        database="prod",
        dsn=_DSN,
        graph_path=tmp_path / "graphs" / "prod.kuzu",
        encrypt_dsn=True,
        encryptor=enc,
    )
    reg.save()

    assert _mode(keystore) == 0o600
    assert _mode(tmp_path / "registry.json") == 0o600


@pytest.mark.skipif(not _POSIX, reason="POSIX file modes only")
def test_secure_mkdir_is_0700(tmp_path: Path) -> None:
    d = tmp_path / "state" / "nested"
    secure_mkdir(d)
    assert _mode(d) == 0o700


@pytest.mark.skipif(not _POSIX, reason="POSIX file modes only")
def test_loose_keystore_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    keystore = tmp_path / "keystore"
    DSNEncryptor(keystore)  # creates it 0o600
    os.chmod(keystore, 0o644)  # loosen it
    with caplog.at_level("WARNING"):
        DSNEncryptor(keystore)  # re-open should warn
    assert any("permissions" in r.message for r in caplog.records)


@pytest.mark.skipif(not _POSIX, reason="POSIX file modes only")
def test_atomic_write_text_default_mode(tmp_path: Path) -> None:
    target = tmp_path / "secret.txt"
    atomic_write_text(target, "data")
    assert _mode(target) == 0o600
    assert target.read_text() == "data"


@pytest.mark.parametrize(
    ("dsn", "expected"),
    [
        ("postgresql://u:pw@h:5432/db", "postgresql://u:***@h:5432/db"),
        ("snowflake://bob:pw@acct/DB/SCHEMA?warehouse=W", None),  # password masked
        ("postgresql://h:5432/db", "postgresql://h:5432/db"),  # no userinfo
        ("not-a-dsn", "not-a-dsn"),
    ],
)
def test_redact_dsn(dsn: str, expected: str | None) -> None:
    out = redact_dsn(dsn)
    assert "pw" not in out
    if expected is not None:
        assert out == expected
