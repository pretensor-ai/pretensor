"""Fernet encryption for DSN strings at rest under ``.pretensor/``."""

from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from pretensor.core.secure_io import (
    atomic_write_bytes,
    secure_mkdir,
    warn_if_world_readable,
)
from pretensor.errors import PretensorError

__all__ = ["DSNEncryptor", "DSNDecryptError"]


class DSNDecryptError(PretensorError, RuntimeError):
    """Raised when an encrypted DSN cannot be decrypted."""


class DSNEncryptor:
    """Symmetric encryption for connection strings (key file beside registry)."""

    def __init__(self, key_path: Path) -> None:
        self._key_path = key_path
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        # The directory holding the sole secret that decrypts every DSN is
        # created 0o700, and the key file 0o600, so a local user on a shared
        # host cannot read it. An existing too-loose key triggers a warning.
        secure_mkdir(self._key_path.parent)
        if self._key_path.exists():
            warn_if_world_readable(self._key_path)
            return self._key_path.read_bytes().strip()
        key = Fernet.generate_key()
        atomic_write_bytes(self._key_path, key + b"\n", mode=0o600)
        return key

    def encrypt(self, dsn: str) -> str:
        """Return urlsafe base64 ciphertext."""
        return self._fernet.encrypt(dsn.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        """Recover the original DSN or raise :class:`DSNDecryptError`."""
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise DSNDecryptError("Invalid encrypted DSN or wrong keystore") from exc
