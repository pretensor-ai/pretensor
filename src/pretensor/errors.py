"""Root exception hierarchy for the pretensor package.

All internal errors raised by pretensor modules are subclasses of
:class:`PretensorError`. Callers that want to catch any pretensor-specific
failure without enumerating every subclass can use a single ``except
PretensorError`` clause. Subclasses may additionally inherit from a Python
builtin (e.g. ``ValueError``, ``KeyError``) so that existing ``except
<Builtin>`` sites keep working without any changes.
"""

from __future__ import annotations

__all__ = ["PretensorError", "ConnectorError"]


class PretensorError(Exception):
    """Base class for all exceptions raised by the pretensor library.

    Catching this class is sufficient to handle any error that originates
    inside pretensor code (as opposed to errors propagated from third-party
    libraries such as SQLAlchemy or Fernet, which are left unwrapped unless
    pretensor re-raises them explicitly).
    """


class ConnectorError(PretensorError):
    """Raised when a database connector operation fails.

    This is the single shared base for all connector-level errors across
    every supported database backend (PostgreSQL, MySQL, Snowflake, BigQuery,
    etc.). Both ``pretensor.connectors.postgres`` and
    ``pretensor.connectors.mysql`` re-export this class under the name
    ``ConnectorError`` so that code importing from either module continues
    to work unchanged while now sharing the same exception identity.
    """
