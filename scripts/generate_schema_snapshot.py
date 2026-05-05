"""Regenerate a benchmark schema-snapshot YAML from a live Postgres database.

One developer-only helper. Not library code and not invoked by tests — it
exists so fixtures under ``tests/fixtures/schemas/*.yaml`` stay reproducible
from checked-in DDL (``scripts/data/<dataset>/schema.sql``).

Two modes:

* ``--url <dsn>``: connect to an existing Postgres and introspect it.
* ``--ddl <path>``: spawn an ephemeral in-process Postgres via ``pgserver``,
  ``psql``-load the DDL file, introspect, then tear the server down. No
  Docker, no root, no system Postgres required.

In both modes the resulting :class:`SchemaSnapshot` has ``introspected_at``
normalized to ``2024-01-01T00:00:00Z`` so re-running the generator produces
a byte-identical YAML (outside of genuine schema changes).

Usage::

    # Ephemeral DB, load DDL, snapshot to YAML
    uv run --with pgserver python scripts/generate_schema_snapshot.py \\
        --ddl scripts/data/adventureworks/schema.sql \\
        --name adventureworks \\
        --out tests/fixtures/schemas/adventureworks.yaml

    # Existing Postgres
    uv run python scripts/generate_schema_snapshot.py \\
        --url postgresql://user:pass@localhost/mydb \\
        --name mydb \\
        --out tests/fixtures/schemas/mydb.yaml
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from pretensor.connectors.inspect import inspect  # noqa: E402
from pretensor.connectors.postgres import PostgresConnector  # noqa: E402
from pretensor.introspection.models.config import (  # noqa: E402
    ConnectionConfig,
    DatabaseType,
)

FROZEN_INTROSPECTED_AT = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _patch_postgres_connector_for_unix_sockets() -> None:
    """Extend ``PostgresConnector._build_url`` to recognize Unix-socket configs.

    Production ``ConnectionConfig`` only carries ``host``/``port`` (TCP). When
    ``metadata_extra["unix_socket_dir"]`` is present we emit the libpq
    ``postgresql:///db?host=/dir`` form instead. The patch mutates
    ``PostgresConnector._build_url`` for the lifetime of this process, so
    any library code running in the same process would see the patched
    method; this is a dev-only script invoked via ``uv run ...``, so no
    library call site shares that process.
    """
    original = PostgresConnector._build_url

    def _build_url(self):
        cfg = self.config
        sock = cfg.metadata_extra.get("unix_socket_dir")
        if sock:
            user = cfg.user or "postgres"
            password = cfg.password or ""
            database = cfg.database or "postgres"
            cred = f"{user}:{password}@" if password else f"{user}@"
            return f"postgresql+psycopg2://{cred}/{database}?host={sock}"
        return original(self)

    PostgresConnector._build_url = _build_url  # type: ignore[method-assign]


def _config_from_url(name: str, url: str) -> ConnectionConfig:
    parsed = urlparse(url)
    if parsed.scheme not in ("postgresql", "postgres"):
        raise ValueError(f"Only postgres URLs are supported, got {parsed.scheme!r}")
    metadata_extra: dict[str, object] = {}
    # pgserver and libpq both encode a Unix socket dir as `?host=/path/...`.
    query = dict(
        kv.split("=", 1)
        for kv in (parsed.query.split("&") if parsed.query else [])
        if "=" in kv
    )
    if "host" in query and query["host"].startswith("/"):
        metadata_extra["unix_socket_dir"] = query["host"]
    return ConnectionConfig(
        name=name,
        type=DatabaseType.POSTGRES,
        host=parsed.hostname,
        port=parsed.port,
        database=(parsed.path or "/").lstrip("/") or None,
        user=parsed.username,
        password=parsed.password,
        metadata_extra=metadata_extra,
    )


def _run_against_url(name: str, url: str) -> str:
    _patch_postgres_connector_for_unix_sockets()
    config = _config_from_url(name, url)
    snapshot = inspect(config)
    # Pin the timestamp so the YAML diff is stable across re-runs.
    snapshot = snapshot.model_copy(update={"introspected_at": FROZEN_INTROSPECTED_AT})
    return snapshot.to_yaml()


def _run_ephemeral(name: str, ddl_path: Path) -> str:
    try:
        import pgserver  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "pgserver is required for --ddl mode. "
            "Re-run with: uv run --with pgserver python scripts/generate_schema_snapshot.py ..."
        ) from exc

    # Raise on decode errors rather than silently substituting replacement
    # characters — a corrupted DDL byte would otherwise land in the shipped
    # snapshot with no diagnostic.
    ddl = ddl_path.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="pretensor-pgserver-") as data_dir:
        with pgserver.get_server(data_dir, cleanup_mode="stop") as srv:
            # psql executes the DDL including meta-commands like \connect.
            srv.psql(ddl)
            uri = srv.get_uri()
            return _run_against_url(name, uri)


def main() -> None:
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(description=doc.splitlines()[0] if doc else None)
    parser.add_argument(
        "--name", required=True, help="Dataset name (goes into connection_name)."
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="Where to write the YAML."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Postgres URL of an already-running DB.")
    source.add_argument(
        "--ddl", type=Path, help="Path to a DDL-only .sql file; spawns pgserver."
    )
    args = parser.parse_args()

    if args.url is not None:
        yaml_text = _run_against_url(args.name, args.url)
    else:
        yaml_text = _run_ephemeral(args.name, args.ddl)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml_text, encoding="utf-8")
    print(f"wrote {args.out} ({len(yaml_text)} bytes)")


if __name__ == "__main__":
    main()
