"""Fernet encryption for DSN strings at rest under ``.pretensor/``."""

from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

__all__ = ["DSNEncryptor", "DSNDecryptError"]


class DSNDecryptError(RuntimeError):
    """Raised when an encrypted DSN cannot be decrypted."""


class DSNEncryptor:
    """Symmetric encryption for connection strings (key file beside registry)."""

    def __init__(self, key_path: Path) -> None:
        self._key_path = key_path
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        self._key_path.parent.mkdir(parents=True, exist_ok=True)
        if self._key_path.exists():
            return self._key_path.read_bytes().strip()
        key = Fernet.generate_key()
        self._key_path.write_bytes(key + b"\n")
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
