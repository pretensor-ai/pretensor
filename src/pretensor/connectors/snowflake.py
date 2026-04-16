"""Snowflake connector using SQLAlchemy (snowflake-sqlalchemy)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from pretensor.connectors.base import (
    BaseConnector,
    ColumnInfo,
    ColumnStats,
    ForeignKeyInfo,
    TableGrant,
    TableInfo,
)
from pretensor.connectors.lineage_sqlglot import dml_write_targets, table_refs_from_sql
from pretensor.connectors.models import ViewDependency
from pretensor.introspection.models.config import ConnectionConfig, SchemaFilter

logger = logging.getLogger(__name__)

_FQ_TABLE_QUOTED_RE = re.compile(
    r'^"(?P<db>[^"]+)"\."(?P<schema>[^"]+)"\."(?P<table>[^"]+)"$'
)
_STALE_ACCESS_DAYS = 90


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_snowflake_table_fqn(fq: str, expected_db: str) -> tuple[str, str] | None:
    """Return ``(schema, table)`` when FQN targets ``expected_db``."""
    m = _FQ_TABLE_QUOTED_RE.match(fq.strip())
    if m:
        if m.group("db").upper() != expected_db.upper():
            return None
        return m.group("schema"), m.group("table")
    parts = fq.split(".")
    if len(parts) == 3:
        db, sn, tn = (p.strip().strip('"') for p in parts)
        if db.upper() != expected_db.upper():
            return None
        return sn, tn
    return None


class SnowflakeConnectorError(Exception):
    """Raised when Snowflake connector operations fail."""


def _sf_table_type(raw: str | None) -> str:
    if not raw:
        return "table"
    u = raw.upper()
    if u == "BASE TABLE":
        return "table"
    if u == "VIEW":
        return "view"
    if u == "MATERIALIZED VIEW":
        return "materialized_view"
    if u == "EXTERNAL TABLE":
        return "foreign_table"
    return raw.lower().replace(" ", "_")


def _quote_ident(name: str) -> str:
    """Quote a Snowflake identifier with double-quote escaping."""
    return f'"{name.replace(chr(34), chr(34) * 2)}"'


class SnowflakeConnector(BaseConnector):
    """Introspect Snowflake via INFORMATION_SCHEMA and SHOW commands."""

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._engine: Engine | None = None
        self._pk_cache: dict[tuple[str, str], set[str]] | None = None

    def _snowflake_url(self) -> str:
        cfg = self.config
        account = (cfg.host or "").strip()
        if not account:
            raise SnowflakeConnectorError("Snowflake account (DSN host) is required")

        user = cfg.user or ""
        password = cfg.password or ""
        database = (cfg.database or "").strip()
        if not database:
            raise SnowflakeConnectorError(
                "Snowflake database name is required in the DSN path"
            )

        extra = cfg.metadata_extra or {}
        schema = (extra.get("snowflake_schema") or "").strip()
        warehouse = (extra.get("warehouse") or "").strip()
        role = (extra.get("role") or "").strip()

        path = f"/{database}"
        if schema:
            path += f"/{schema}"

        query_parts: list[str] = []
        if warehouse:
            from urllib.parse import quote

            query_parts.append(f"warehouse={quote(warehouse, safe='')}")
        if role:
            from urllib.parse import quote

            query_parts.append(f"role={quote(role, safe='')}")

        q = ("?" + "&".join(query_parts)) if query_parts else ""
        return f"snowflake://{user}:{password}@{account}{path}{q}"

    def connect(self) -> None:
        url = self._snowflake_url()
        try:
            self._engine = create_engine(url, pool_pre_ping=True)
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(
                "Connected to Snowflake account %s db %s",
                self.config.host,
                self.config.database,
            )
        except Exception as exc:
            raise SnowflakeConnectorError(
                f"Failed to connect to Snowflake: {exc}"
            ) from exc

    def disconnect(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise SnowflakeConnectorError("Not connected.")
        return self._engine

    def _qualified_information_schema(self) -> str:
        db = (self.config.database or "").replace('"', '""')
        return f'"{db}".INFORMATION_SCHEMA'

    def get_tables(self, schema_filter: SchemaFilter | None = None) -> list[TableInfo]:
        sf = schema_filter or self.config.schema_filter
        ischema = self._qualified_information_schema()

        query = text(f"""\
            SELECT
                table_schema,
                table_name,
                table_type,
                comment,
                row_count
            FROM {ischema}.TABLES
            WHERE table_schema NOT IN ('INFORMATION_SCHEMA')
            ORDER BY table_schema, table_name
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()

        out: list[TableInfo] = []
        for row in rows:
            schema_name = str(row["table_schema"])
            if sf.include and schema_name not in sf.include:
                continue
            if sf.exclude and schema_name in sf.exclude:
                continue
            rc = row["row_count"]
            out.append(
                TableInfo(
                    name=str(row["table_name"]),
                    schema_name=schema_name,
                    row_count=int(rc) if rc is not None else None,
                    comment=str(row["comment"]) if row["comment"] else None,
                    table_type=_sf_table_type(
                        str(row["table_type"]) if row["table_type"] else None
                    ),
                )
            )
        return out

    def get_columns(self, table_name: str, schema_name: str) -> list[ColumnInfo]:
        ischema = self._qualified_information_schema()
        query = text(f"""\
            SELECT
                column_name,
                ordinal_position,
                data_type,
                is_nullable,
                column_default,
                comment
            FROM {ischema}.COLUMNS
            WHERE table_schema = :schema_name
              AND table_name = :table_name
            ORDER BY ordinal_position
        """)

        pk_cols = self._primary_key_columns(schema_name, table_name)
        checks_by_col = self._load_check_constraints_for_table(schema_name, table_name)

        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    query,
                    {"schema_name": schema_name, "table_name": table_name},
                )
                .mappings()
                .all()
            )

        columns: list[ColumnInfo] = []
        for row in rows:
            cname = str(row["column_name"])
            ord_raw = row["ordinal_position"]
            columns.append(
                ColumnInfo(
                    name=cname,
                    data_type=str(row["data_type"] or ""),
                    nullable=str(row["is_nullable"] or "").upper() == "YES",
                    is_primary_key=cname in pk_cols,
                    default_value=str(row["column_default"])
                    if row["column_default"] is not None
                    else None,
                    comment=str(row["comment"]) if row["comment"] else None,
                    is_indexed=False,
                    check_constraints=list(checks_by_col.get(cname, [])),
                    ordinal_position=int(ord_raw) if ord_raw is not None else None,
                )
            )
        return columns

    def _load_primary_keys_for_schema(self, schema_name: str) -> None:
        """Populate ``_pk_cache`` via ``SHOW PRIMARY KEYS IN SCHEMA``.

        Uses ``exec_driver_sql`` because SHOW commands don't support bind
        params.  Result columns by index: 0=created_on, 1=database_name,
        2=schema_name, 3=table_name, 4=column_name, 5=key_sequence,
        6=constraint_name.
        """
        db = _quote_ident(self.config.database or "")
        schema = _quote_ident(schema_name)
        sql = f"SHOW PRIMARY KEYS IN SCHEMA {db}.{schema}"
        cache: dict[tuple[str, str], set[str]] = {}
        try:
            with self.engine.connect() as conn:
                rows = conn.exec_driver_sql(sql).all()
            for row in rows:
                sn, tn, col = str(row[2]), str(row[3]), str(row[4])
                cache.setdefault((sn, tn), set()).add(col)
            logger.debug(
                "Loaded %d PK entries for %s.%s",
                sum(len(v) for v in cache.values()),
                self.config.database,
                schema_name,
            )
        except Exception:
            logger.warning(
                "SHOW PRIMARY KEYS failed for %s.%s; PK info unavailable",
                self.config.database,
                schema_name,
                exc_info=True,
            )
        self._pk_cache = cache

    def _primary_key_columns(self, schema_name: str, table_name: str) -> set[str]:
        if self._pk_cache is None:
            self._load_primary_keys_for_schema(schema_name)
        assert self._pk_cache is not None  # populated by _load above
        return self._pk_cache.get((schema_name, table_name), set())

    def _load_check_constraints_for_table(
        self, schema_name: str, table_name: str
    ) -> dict[str, list[str]]:
        # Snowflake doesn't enforce CHECK constraints and exposes no SHOW
        # equivalent — return empty unconditionally.
        return {}

    def get_foreign_keys(self) -> list[ForeignKeyInfo]:
        show_fks, show_succeeded = self._foreign_keys_from_show()
        if show_fks:
            return show_fks
        if show_succeeded:
            logger.info(
                "SHOW IMPORTED KEYS returned no rows for %s; trying INFORMATION_SCHEMA fallback",
                self.config.database,
            )
        else:
            logger.info(
                "SHOW IMPORTED KEYS failed for %s; trying INFORMATION_SCHEMA fallback",
                self.config.database,
            )
        return self._foreign_keys_from_information_schema()

    def _foreign_keys_from_show(self) -> tuple[list[ForeignKeyInfo], bool]:
        """Fetch FKs via ``SHOW IMPORTED KEYS IN DATABASE``.

        Result columns by index: 0=created_on, 1=pk_database_name,
        2=pk_schema_name, 3=pk_table_name, 4=pk_column_name,
        5=fk_database_name, 6=fk_schema_name, 7=fk_table_name,
        8=fk_column_name, 9=key_sequence, 10=update_rule,
        11=delete_rule, 12=fk_name, …

        Returns:
            Tuple of ``(foreign_keys, show_succeeded)`` where
            ``show_succeeded`` is False only if SHOW itself errored.
        """
        db = _quote_ident(self.config.database or "")
        sql = f"SHOW IMPORTED KEYS IN DATABASE {db}"
        try:
            with self.engine.connect() as conn:
                rows = conn.exec_driver_sql(sql).all()
        except Exception:
            logger.warning(
                "SHOW IMPORTED KEYS failed for %s",
                self.config.database,
                exc_info=True,
            )
            return [], False
        if not rows:
            return [], True

        # Sort by fk_schema(6), fk_table(7), fk_name(12), key_sequence(9)
        # so composite FK columns are ordered correctly for downstream grouping.
        sorted_rows = sorted(
            rows,
            key=lambda r: (str(r[6]), str(r[7]), str(r[12]), _safe_int(r[9])),
        )
        fks = [
            ForeignKeyInfo(
                constraint_name=str(row[12]),
                source_schema=str(row[6]),
                source_table=str(row[7]),
                source_column=str(row[8]),
                target_schema=str(row[2]),
                target_table=str(row[3]),
                target_column=str(row[4]),
            )
            for row in sorted_rows
        ]
        return fks, True

    def _foreign_keys_from_information_schema(self) -> list[ForeignKeyInfo]:
        ischema = self._qualified_information_schema()
        query = text(f"""\
            SELECT
                tc.constraint_name,
                kcu_src.table_schema AS source_schema,
                kcu_src.table_name AS source_table,
                kcu_src.column_name AS source_column,
                kcu_ref.table_schema AS target_schema,
                kcu_ref.table_name AS target_table,
                kcu_ref.column_name AS target_column,
                kcu_src.ordinal_position AS source_ordinal_position
            FROM {ischema}.TABLE_CONSTRAINTS tc
            JOIN {ischema}.KEY_COLUMN_USAGE kcu_src
              ON tc.constraint_name = kcu_src.constraint_name
             AND tc.constraint_schema = kcu_src.constraint_schema
             AND tc.table_schema = kcu_src.table_schema
             AND tc.table_name = kcu_src.table_name
            JOIN {ischema}.REFERENTIAL_CONSTRAINTS rc
              ON tc.constraint_name = rc.constraint_name
             AND tc.constraint_schema = rc.constraint_schema
            JOIN {ischema}.KEY_COLUMN_USAGE kcu_ref
              ON rc.unique_constraint_name = kcu_ref.constraint_name
             AND rc.unique_constraint_schema = kcu_ref.constraint_schema
             AND kcu_src.position_in_unique_constraint = kcu_ref.ordinal_position
            WHERE tc.constraint_type = 'FOREIGN KEY'
        """)
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(query).mappings().all()
        except SQLAlchemyError:
            logger.warning(
                "INFORMATION_SCHEMA FK fallback failed for %s",
                self.config.database,
                exc_info=True,
            )
            return []
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                str(row["source_schema"]),
                str(row["source_table"]),
                str(row["constraint_name"]),
                _safe_int(row["source_ordinal_position"]),
            ),
        )
        return [
            ForeignKeyInfo(
                constraint_name=str(row["constraint_name"]),
                source_schema=str(row["source_schema"]),
                source_table=str(row["source_table"]),
                source_column=str(row["source_column"]),
                target_schema=str(row["target_schema"]),
                target_table=str(row["target_table"]),
                target_column=str(row["target_column"]),
            )
            for row in sorted_rows
        ]

    def get_table_row_count(self, table_name: str, schema_name: str) -> int:
        ischema = self._qualified_information_schema()
        q = text(f"""\
            SELECT row_count
            FROM {ischema}.TABLES
            WHERE table_schema = :sn AND table_name = :tn
        """)
        with self.engine.connect() as conn:
            row = conn.execute(q, {"sn": schema_name, "tn": table_name}).first()
        if row is None or row[0] is None:
            return 0
        return int(row[0])

    def get_column_stats(
        self, table_name: str, column_name: str, schema_name: str
    ) -> ColumnStats:
        return ColumnStats()

    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        return [dict(row) for row in rows]

    def get_table_grants(
        self, schema_filter: SchemaFilter | None = None
    ) -> list[TableGrant]:
        """Return direct SELECT grants on tables from OBJECT_PRIVILEGES.

        Uses ``GRANTEE``, ``OBJECT_SCHEMA``, and ``OBJECT_NAME`` (aliased in SQL).
        Recursive expansion of nested roles (``SHOW GRANTS TO ROLE``) is not
        implemented; only privileges visible in INFORMATION_SCHEMA are returned.
        """
        sf = schema_filter or self.config.schema_filter
        ischema = self._qualified_information_schema()
        query = text(f"""\
            SELECT
                GRANTEE AS grantee_name,
                OBJECT_SCHEMA AS table_schema,
                OBJECT_NAME AS table_name
            FROM {ischema}.OBJECT_PRIVILEGES
            WHERE PRIVILEGE_TYPE = 'SELECT'
              AND OBJECT_TYPE = 'TABLE'
        """)
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(query).mappings().all()
        except SQLAlchemyError as exc:
            logger.warning(
                "Could not read SELECT grants from information_schema.object_privileges: %s",
                exc,
            )
            return []

        seen: set[tuple[str, str, str]] = set()
        out: list[TableGrant] = []
        for row in rows:
            ge = row["grantee_name"]
            ts = row["table_schema"]
            tn = row["table_name"]
            if ge is None or ts is None or tn is None:
                continue
            grantee = str(ge)
            schema_name = str(ts)
            table_name = str(tn)
            if not self._schema_ok(schema_name, sf):
                continue
            key = (grantee, schema_name, table_name)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                TableGrant(
                    grantee=grantee,
                    schema_name=schema_name,
                    table_name=table_name,
                )
            )
        out.sort(key=lambda tg: (tg.schema_name, tg.table_name, tg.grantee))
        return out

    def _schema_ok(self, sn: str, schema_filter: SchemaFilter) -> bool:
        if schema_filter.include and sn not in schema_filter.include:
            return False
        if schema_filter.exclude and sn in schema_filter.exclude:
            return False
        return sn.upper() != "INFORMATION_SCHEMA"

    def _sf_lineage_type(self, domain: str | None) -> str | None:
        if not domain:
            return None
        u = str(domain).strip().upper().replace(" ", "_")
        if u == "MATERIALIZED_VIEW" or u == "MATERIALIZEDVIEW":
            return "MATERIALIZED_VIEW"
        if u in ("VIEW", "TRIGGER", "TASK", "STREAM"):
            return u
        return None

    def load_view_dependencies(self, schema_filter: SchemaFilter) -> list[ViewDependency]:
        """Lineage from OBJECT_DEPENDENCIES, view SQL, SHOW STREAMS, and SHOW TASKS."""
        deps: list[ViewDependency] = []
        obj_dep_keys: set[tuple[str, str, str, str]] = set()
        deps.extend(
            self._snowflake_object_dependencies(schema_filter, obj_dep_keys)
        )
        deps.extend(
            self._snowflake_view_sql_fallback(schema_filter, obj_dep_keys)
        )
        deps.extend(self._snowflake_streams_lineage(schema_filter))
        deps.extend(self._snowflake_tasks_lineage(schema_filter))
        return deps

    def _snowflake_object_dependencies(
        self,
        schema_filter: SchemaFilter,
        obj_dep_keys: set[tuple[str, str, str, str]],
    ) -> list[ViewDependency]:
        out: list[ViewDependency] = []
        ischema = self._qualified_information_schema()
        q = text(f"""\
            SELECT
                referencing_object_schema,
                referencing_object_name,
                referencing_object_domain,
                referenced_object_schema,
                referenced_object_name
            FROM {ischema}.OBJECT_DEPENDENCIES
            WHERE referenced_object_domain = 'TABLE'
        """)
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(q).mappings().all()
        except Exception as exc:
            logger.warning("Snowflake OBJECT_DEPENDENCIES unavailable: %s", exc)
            return out
        for row in rows:
            ref_schema = str(row["referenced_object_schema"] or "")
            ref_name = str(row["referenced_object_name"] or "")
            tgt_schema = str(row["referencing_object_schema"] or "")
            tgt_name = str(row["referencing_object_name"] or "")
            dom = row["referencing_object_domain"]
            ltype = self._sf_lineage_type(str(dom) if dom is not None else None)
            if ltype is None:
                continue
            if not self._schema_ok(ref_schema, schema_filter):
                continue
            if not self._schema_ok(tgt_schema, schema_filter):
                continue
            obj = f"{tgt_schema}.{tgt_name}"
            out.append(
                ViewDependency(
                    source_schema=ref_schema,
                    source_table=ref_name,
                    target_schema=tgt_schema,
                    target_table=tgt_name,
                    lineage_type=ltype,
                    object_name=obj,
                    confidence=1.0,
                )
            )
            obj_dep_keys.add((tgt_schema, tgt_name, ref_schema, ref_name))
        return out

    def _snowflake_view_sql_fallback(
        self,
        schema_filter: SchemaFilter,
        obj_dep_keys: set[tuple[str, str, str, str]],
    ) -> list[ViewDependency]:
        out: list[ViewDependency] = []
        ischema = self._qualified_information_schema()
        q = text(f"""\
            SELECT table_schema, table_name, view_definition
            FROM {ischema}.VIEWS
            WHERE table_schema NOT IN ('INFORMATION_SCHEMA')
        """)
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(q).mappings().all()
        except Exception as exc:
            logger.warning("Snowflake VIEWS lineage fallback skipped: %s", exc)
            return out
        for row in rows:
            ts = str(row["table_schema"] or "")
            if not self._schema_ok(ts, schema_filter):
                continue
            tname = str(row["table_name"] or "")
            vd = row["view_definition"]
            if vd is None:
                continue
            sql_text = str(vd)
            for src_schema, src_table in table_refs_from_sql(
                sql_text, dialect="snowflake", default_schema=ts
            ):
                if not self._schema_ok(src_schema, schema_filter):
                    continue
                if src_schema == ts and src_table == tname:
                    continue
                key = (ts, tname, src_schema, src_table)
                if key in obj_dep_keys:
                    continue
                out.append(
                    ViewDependency(
                        source_schema=src_schema,
                        source_table=src_table,
                        target_schema=ts,
                        target_table=tname,
                        lineage_type="VIEW",
                        object_name=f"{ts}.{tname}",
                        confidence=0.85,
                    )
                )
        return out

    def _snowflake_streams_lineage(self, schema_filter: SchemaFilter) -> list[ViewDependency]:
        out: list[ViewDependency] = []
        db = (self.config.database or "").replace('"', '""')
        if not db:
            return out
        q = text(f'SHOW STREAMS IN DATABASE "{db}"')
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(q).mappings().all()
        except Exception as exc:
            logger.warning("Snowflake SHOW STREAMS unavailable: %s", exc)
            return out
        for row in rows:
            m = dict(row)
            name = m.get("name") or m.get("NAME")
            schema_key = m.get("schema_name") or m.get("SCHEMA_NAME")
            table_key = m.get("table_name") or m.get("TABLE_NAME")
            if name is None or schema_key is None or table_key is None:
                continue
            tgt_schema = str(schema_key)
            tgt_name = str(name)
            src_table = str(table_key)
            src_schema = str(m.get("source_schema") or m.get("SOURCE_SCHEMA") or tgt_schema)
            if not self._schema_ok(tgt_schema, schema_filter):
                continue
            if not self._schema_ok(src_schema, schema_filter):
                continue
            out.append(
                ViewDependency(
                    source_schema=src_schema,
                    source_table=src_table,
                    target_schema=tgt_schema,
                    target_table=tgt_name,
                    lineage_type="STREAM",
                    object_name=f"{tgt_schema}.{tgt_name}",
                    confidence=1.0,
                )
            )
        return out

    def _snowflake_tasks_lineage(self, schema_filter: SchemaFilter) -> list[ViewDependency]:
        out: list[ViewDependency] = []
        db = (self.config.database or "").replace('"', '""')
        if not db:
            return out
        q = text(f'SHOW TASKS IN DATABASE "{db}"')
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(q).mappings().all()
        except Exception as exc:
            logger.warning("Snowflake SHOW TASKS unavailable: %s", exc)
            return out
        for row in rows:
            m = dict(row)
            name = m.get("name") or m.get("NAME")
            schema_key = m.get("schema_name") or m.get("SCHEMA_NAME")
            definition = m.get("definition") or m.get("DEFINITION")
            if name is None or schema_key is None or definition is None:
                continue
            ts = str(schema_key)
            if not self._schema_ok(ts, schema_filter):
                continue
            tname = str(name)
            sql_text = str(definition)
            reads = table_refs_from_sql(
                sql_text, dialect="snowflake", default_schema=ts
            )
            writes = dml_write_targets(
                sql_text, dialect="snowflake", default_schema=ts
            )
            if not writes:
                continue
            for ws, wt in writes:
                if not self._schema_ok(ws, schema_filter):
                    continue
                for rs, rt in reads:
                    if not self._schema_ok(rs, schema_filter):
                        continue
                    if rs == ws and rt == wt:
                        continue
                    out.append(
                        ViewDependency(
                            source_schema=rs,
                            source_table=rt,
                            target_schema=ws,
                            target_table=wt,
                            lineage_type="TASK",
                            object_name=f"{ts}.{tname}",
                            confidence=0.85,
                        )
                    )
        return out

    def load_deep_catalog(
        self, schema_filter: SchemaFilter
    ) -> tuple[
        dict[tuple[str, str], dict[str, Any]],
        dict[tuple[str, str, str], dict[str, Any]],
    ]:
        """TABLES extras and optional ACCOUNT_USAGE access aggregates (best-effort)."""
        table_extra: dict[tuple[str, str], dict[str, Any]] = {}
        column_extra: dict[tuple[str, str, str], dict[str, Any]] = {}

        def _schema_ok(sn: str) -> bool:
            if schema_filter.include and sn not in schema_filter.include:
                return False
            if schema_filter.exclude and sn in schema_filter.exclude:
                return False
            return True

        ischema = self._qualified_information_schema()
        db_name = (self.config.database or "").replace('"', '""')

        ext_sql = text(f"""\
            SELECT table_schema, table_name, bytes, clustering_key
            FROM {ischema}.TABLES
            WHERE table_schema NOT IN ('INFORMATION_SCHEMA')
        """)
        try:
            with self.engine.connect() as conn:
                for row in conn.execute(ext_sql).mappings().all():
                    sn = str(row["table_schema"])
                    tn = str(row["table_name"])
                    if not _schema_ok(sn):
                        continue
                    key = (sn, tn)
                    entry: dict[str, Any] = {}
                    b = row.get("bytes")
                    if b is not None:
                        entry["table_bytes"] = int(b)
                    ck = row.get("clustering_key")
                    if ck:
                        entry["clustering_key"] = str(ck)
                    if entry:
                        table_extra.setdefault(key, {}).update(entry)
        except Exception as exc:
            logger.warning("Snowflake TABLES extended fields skipped: %s", exc)

        read_sql = text("""\
            SELECT
                f.value:objectName::string AS fq_name,
                COUNT(*) AS read_count,
                MAX(q.query_start_time) AS last_read
            FROM snowflake.account_usage.access_history q,
            LATERAL FLATTEN(input => q.base_objects_accessed) f
            WHERE q.query_start_time >= DATEADD(day, -365, CURRENT_TIMESTAMP())
              AND f.value:objectDomain::string = 'Table'
            GROUP BY 1
        """)
        write_sql = text("""\
            SELECT
                f.value:objectName::string AS fq_name,
                COUNT(*) AS write_count
            FROM snowflake.account_usage.access_history q,
            LATERAL FLATTEN(input => q.objects_modified) f
            WHERE q.query_start_time >= DATEADD(day, -365, CURRENT_TIMESTAMP())
              AND f.value:objectDomain::string IN ('Table', 'Materialized View')
            GROUP BY 1
        """)

        read_map: dict[str, tuple[int, datetime | None]] = {}
        write_map: dict[str, int] = {}
        try:
            with self.engine.connect() as conn:
                for row in conn.execute(read_sql).mappings().all():
                    fq = row["fq_name"]
                    if fq is None:
                        continue
                    read_map[str(fq)] = (
                        int(row["read_count"] or 0),
                        row["last_read"],
                    )
                for row in conn.execute(write_sql).mappings().all():
                    fq = row["fq_name"]
                    if fq is None:
                        continue
                    write_map[str(fq)] = int(row["write_count"] or 0)
        except Exception as exc:
            logger.warning("Snowflake ACCOUNT_USAGE access history skipped: %s", exc)

        now = datetime.now(timezone.utc)
        all_fq = set(read_map) | set(write_map)
        for fq in all_fq:
            parsed = _parse_snowflake_table_fqn(fq, db_name)
            if parsed is None:
                continue
            sn, tn = parsed
            if not _schema_ok(sn):
                continue
            rc, last_read = read_map.get(fq, (0, None))
            wc = write_map.get(fq, 0)
            key = (sn, tn)
            days_since: int | None = None
            if last_read is not None:
                lr = last_read
                if lr.tzinfo is None:
                    lr = lr.replace(tzinfo=timezone.utc)
                days_since = max(0, int((now - lr.astimezone(timezone.utc)).days))
            # A table absent from read_map was never accessed in the 365-day window,
            # which is a stronger unused signal than one accessed >90 days ago.
            fq_in_read_map = fq in read_map
            potentially_unused = not fq_in_read_map or (
                days_since is not None and days_since >= _STALE_ACCESS_DAYS
            )
            table_extra.setdefault(key, {}).update(
                {
                    "access_read_count": rc,
                    "access_write_count": wc,
                    "days_since_last_access": days_since,
                    "potentially_unused": potentially_unused,
                }
            )

        return table_extra, column_extra
