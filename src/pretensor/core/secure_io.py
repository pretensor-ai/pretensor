"""Helpers for writing secrets and credential stores with tight permissions.

These centralise the POSIX file-permission handling for the keystore and the
connection registry so neither the Fernet key nor the (possibly cleartext)
DSN registry is written world-readable. All helpers are POSIX-oriented and
degrade to a plain write on non-POSIX platforms (e.g. Windows), where file
modes are not enforced the same way.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

__all__ = [
    "atomic_write_bytes",
    "atomic_write_text",
    "secure_mkdir",
    "warn_if_world_readable",
]

logger = logging.getLogger(__name__)

_POSIX = os.name == "posix"


def secure_mkdir(path: Path, *, mode: int = 0o700) -> None:
    """Create ``path`` (and parents) and tighten it to ``mode`` on POSIX.

    ``mkdir``'s mode argument is masked by the process umask, so the mode is
    applied explicitly with ``chmod`` after creation.
    """
    path.mkdir(parents=True, exist_ok=True)
    if _POSIX:
        try:
            os.chmod(path, mode)
        except OSError as exc:  # pragma: no cover - platform/filesystem specific
            logger.debug("could not chmod %s to %o: %s", path, mode, exc)


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    """Atomically write ``data`` to ``path``.

    On POSIX the temp file is created with ``O_EXCL`` at ``mode`` (default
    ``0o600``) so the bytes are never visible at looser permissions, then
    ``os.replace``-d into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        if _POSIX:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC, mode)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        else:  # pragma: no cover - exercised on Windows only
            tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(
    path: Path, text: str, *, mode: int = 0o600, encoding: str = "utf-8"
) -> None:
    """Atomically write ``text`` to ``path`` (see :func:`atomic_write_bytes`)."""
    atomic_write_bytes(path, text.encode(encoding), mode=mode)


def warn_if_world_readable(path: Path, *, allowed_mode: int = 0o600) -> None:
    """Log a warning if ``path`` carries permission bits outside ``allowed_mode``.

    ``allowed_mode`` is a bitmask of the permission bits considered safe (default
    ``0o600`` — owner read/write); any bit set on ``path`` but not in the mask
    triggers the warning.
    """
    if not _POSIX or not path.exists():
        return
    current = stat.S_IMODE(path.stat().st_mode)
    if current & ~allowed_mode & 0o777:
        logger.warning(
            "%s has permissions %o (looser than %o); tighten it with: chmod %o %s",
            path,
            current,
            allowed_mode,
            allowed_mode,
            path,
        )
