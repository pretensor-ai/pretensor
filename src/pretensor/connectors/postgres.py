"""PostgreSQL connector.

All Postgres-specific imports and SQL live here. Nothing outside this
package should reference psycopg2 or sqlalchemy.dialects.postgresql.
"""

from __future__ import annotations

import logging
import os
import time
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
from pretensor.connectors.pg_array_parse import parse_pg_array_literal
from pretensor.introspection.models.config import ConnectionConfig, SchemaFilter

logger = logging.getLogger(__name__)

_PROFILE_INDEX = os.environ.get("PRETENSOR_PROFILE_INDEX", "").lower() not in (
    "",
    "0",
    "false",
    "no",
)

LOW_CARDINALITY_THRESHOLD = 50
PG_STATS_MOST_COMMON_CAP = 20

# Wall-clock cap for live `SELECT COUNT(*)` against a view (catalog stats are NULL
# for views in pg_stat_user_tables). Capped server-side via SET LOCAL statement_timeout.
VIEW_COUNT_TIMEOUT_MS = 2000


def _pg_transitive_members(children: dict[str, set[str]], root: str) -> set[str]:
    """All roles (including ``root``) that are direct or indirect members of ``root``.

    ``children[r]`` is the set of direct members of role ``r`` (PostgreSQL ``pg_auth_members``).
    """
    out: set[str] = {root}
    stack = [root]
    while stack:
        parent = stack.pop()
        for mbr in children.get(parent, ()):
            if mbr not in out:
                out.add(mbr)
                stack.append(mbr)
    return out


class ConnectorError(Exception):
    """Raised when a connector operation fails."""


def _map_pg_table_type(relkind: str | None, info_table_type: str) -> str:
    """Normalize to classifier-friendly labels used in the graph."""
    if relkind == "r":
        return "table"
    if relkind == "p":
        return "partitioned_table"
    if relkind == "v":
        return "view"
    if relkind == "m":
        return "materialized_view"
    if relkind == "f":
        return "foreign_table"
    if info_table_type == "VIEW":
        return "view"
    if info_table_type == "FOREIGN TABLE":
        return "foreign_table"
    if info_table_type == "BASE TABLE":
        return "table"
    return "table"


class PostgresConnector(BaseConnector):
    """Postgres implementation of the database connector interface."""

    def __init__(self, config: ConnectionConfig) -> None:
        super().__init__(config)
        self._engine: Engine | None = None
        self._indexed_columns: set[tuple[str, str, str]] = set()
        self._check_by_column: dict[tuple[str, str, str], list[str]] = {}

    def _build_url(self) -> str:
        cfg = self.config
        user = cfg.user or "postgres"
        password = cfg.password or ""
        host = cfg.host or "localhost"
        port = cfg.port or 5432
        database = cfg.database or "postgres"
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    def connect(self) -> None:
        self._indexed_columns = set()
        self._check_by_column = {}
        url = self._build_url()
        try:
            self._engine = create_engine(
                url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 10},
            )
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(
                "Connected to %s:%s/%s",
                self.config.host,
                self.config.port,
                self.config.database,
            )
        except Exception as exc:
            raise ConnectorError(
                f"Failed to connect to Postgres at {self.config.host}:{self.config.port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("Disconnected from %s", self.config.database)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise ConnectorError("Not connected. Call connect() first.")
        return self._engine

    def _refresh_column_metadata_cache(self, schema_filter: SchemaFilter) -> None:
        """Load indexed-column and check-constraint maps for visible schemas."""
        idx_sql = text("""\
            SELECT DISTINCT n.nspname AS table_schema,
                   c.relname AS table_name,
                   a.attname AS column_name
            FROM pg_catalog.pg_index i
            JOIN pg_catalog.pg_class c ON c.oid = i.indrelid
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            CROSS JOIN LATERAL unnest(i.indkey) AS u(attnum)
            JOIN pg_catalog.pg_attribute a
                ON a.attrelid = c.oid AND a.attnum = u.attnum AND NOT a.attisdropped
            WHERE i.indisvalid
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
        """)
        chk_sql = text("""\
            SELECT n.nspname AS table_schema,
                   c.relname AS table_name,
                   a.attname AS column_name,
                   pg_catalog.pg_get_constraintdef(con.oid, true) AS check_def
            FROM pg_catalog.pg_constraint con
            JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            CROSS JOIN LATERAL unnest(con.conkey) AS ck(attnum)
            JOIN pg_catalog.pg_attribute a
                ON a.attrelid = c.oid AND a.attnum = ck.attnum AND NOT a.attisdropped
            WHERE con.contype = 'c'
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
        """)

        self._indexed_columns = set()
        self._check_by_column = {}

        with self.engine.connect() as conn:
            for row in conn.execute(idx_sql).mappings().all():
                sn, tn, cn = row["table_schema"], row["table_name"], row["column_name"]
                if schema_filter.include and sn not in schema_filter.include:
                    continue
                if schema_filter.exclude and sn in schema_filter.exclude:
                    continue
                self._indexed_columns.add((sn, tn, cn))

            for row in conn.execute(chk_sql).mappings().all():
                sn, tn, cn = row["table_schema"], row["table_name"], row["column_name"]
                if schema_filter.include and sn not in schema_filter.include:
                    continue
                if schema_filter.exclude and sn in schema_filter.exclude:
                    continue
                key = (sn, tn, cn)
                expr = row["check_def"]
                if expr:
                    self._check_by_column.setdefault(key, []).append(str(expr))

    def get_tables(self, schema_filter: SchemaFilter | None = None) -> list[TableInfo]:
        sf = schema_filter or self.config.schema_filter
        self._refresh_column_metadata_cache(sf)

        query = text("""\
            SELECT
                t.table_schema,
                t.table_name,
                t.table_type AS info_table_type,
                c.relkind::text AS relkind,
                obj_description(
                    (quote_ident(t.table_schema) || '.' || quote_ident(t.table_name))::regclass
                ) AS table_comment,
                s.n_live_tup AS approx_row_count
            FROM information_schema.tables t
            LEFT JOIN pg_catalog.pg_namespace n ON n.nspname = t.table_schema
            LEFT JOIN pg_catalog.pg_class c
                ON c.relnamespace = n.oid AND c.relname = t.table_name
            LEFT JOIN pg_stat_user_tables s
                ON s.schemaname = t.table_schema
                AND s.relname = t.table_name
            WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
                AND t.table_type IN ('BASE TABLE', 'VIEW', 'FOREIGN TABLE')
            ORDER BY t.table_schema, t.table_name
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()

        tables: list[TableInfo] = []
        for row in rows:
            schema_name = row["table_schema"]
            if sf.include and schema_name not in sf.include:
                continue
            if sf.exclude and schema_name in sf.exclude:
                continue
            relkind = row["relkind"]
            info_tt = row["info_table_type"] or "BASE TABLE"
            mapped_type = _map_pg_table_type(relkind, info_tt)
            approx = row["approx_row_count"]
            if approx is not None:
                row_count: int | None = int(approx)
                row_count_source: str | None = "stat"
            elif mapped_type == "view":
                row_count, row_count_source = self._count_view_rows(
                    schema_name, row["table_name"]
                )
            else:
                row_count = None
                row_count_source = None
            tables.append(
                TableInfo(
                    name=row["table_name"],
                    schema_name=schema_name,
                    row_count=row_count,
                    row_count_source=row_count_source,
                    comment=row["table_comment"],
                    table_type=mapped_type,
                )
            )
        return tables

    def _count_view_rows(
        self, schema_name: str, view_name: str
    ) -> tuple[int, str]:
        """Run ``SELECT COUNT(*)`` on a view with a 2s statement timeout.

        Returns ``(count, "view_count")`` on success, ``(-1, "view_timeout")``
        on timeout or any error — the indexer must not fail because one view is
        slow or broken.
        """
        # Quote identifiers to defeat any oddly-named schemas/views.
        qualified = (
            f'"{schema_name.replace(chr(34), chr(34) * 2)}"'
            f'."{view_name.replace(chr(34), chr(34) * 2)}"'
        )
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(f"SET LOCAL statement_timeout = {VIEW_COUNT_TIMEOUT_MS}")
                )
                result = conn.execute(text(f"SELECT COUNT(*) AS n FROM {qualified}"))
                row = result.mappings().first()
                if row is None or row["n"] is None:
                    return -1, "view_timeout"
                return int(row["n"]), "view_count"
        except SQLAlchemyError as exc:
            logger.debug(
                "View row-count failed for %s.%s: %s",
                schema_name,
                view_name,
                exc,
            )
            return -1, "view_timeout"

    def get_columns(self, table_name: str, schema_name: str) -> list[ColumnInfo]:
        query = text("""\
            SELECT
                c.column_name,
                c.ordinal_position,
                c.data_type,
                c.udt_name,
                c.is_nullable,
                c.column_default,
                col_description(
                    (quote_ident(c.table_schema) || '.' || quote_ident(c.table_name))::regclass,
                    c.ordinal_position
                ) AS column_comment,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END AS is_pk
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT kcu.column_name, kcu.table_schema, kcu.table_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
            ) pk
                ON pk.column_name = c.column_name
                AND pk.table_schema = c.table_schema
                AND pk.table_name = c.table_name
            WHERE c.table_name = :table_name
                AND c.table_schema = :schema_name
            ORDER BY c.ordinal_position
        """)

        with self.engine.connect() as conn:
            rows = (
                conn.execute(
                    query,
                    {"table_name": table_name, "schema_name": schema_name},
                )
                .mappings()
                .all()
            )

        columns: list[ColumnInfo] = []
        for row in rows:
            col_name = row["column_name"]
            data_type = (
                row["udt_name"]
                if row["data_type"] == "USER-DEFINED"
                else row["data_type"]
            )
            ck = (schema_name, table_name, col_name)
            ord_pos = row["ordinal_position"]
            columns.append(
                ColumnInfo(
                    name=col_name,
                    data_type=data_type,
                    nullable=row["is_nullable"] == "YES",
                    is_primary_key=bool(row["is_pk"]),
                    default_value=row["column_default"],
                    comment=row["column_comment"],
                    is_indexed=ck in self._indexed_columns,
                    check_constraints=list(self._check_by_column.get(ck, [])),
                    ordinal_position=int(ord_pos) if ord_pos is not None else None,
                )
            )
        return columns

    def get_foreign_keys(self) -> list[ForeignKeyInfo]:
        # Use key_column_usage on both sides to correctly pair composite FK
        # columns.  The source side (kcu_src) carries ordinal_position and
        # position_in_unique_constraint; the target side (kcu_ref) is the
        # referenced unique/PK constraint whose ordinal_position matches
        # position_in_unique_constraint.  This avoids the Cartesian product
        # that occurs when joining through constraint_column_usage which
        # lacks ordinal_position.
        query = text("""\
            SELECT
                tc.constraint_name,
                kcu_src.table_schema  AS source_schema,
                kcu_src.table_name    AS source_table,
                kcu_src.column_name   AS source_column,
                kcu_ref.table_schema  AS target_schema,
                kcu_ref.table_name    AS target_table,
                kcu_ref.column_name   AS target_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu_src
                ON tc.constraint_name = kcu_src.constraint_name
                AND tc.table_schema   = kcu_src.table_schema
            JOIN information_schema.referential_constraints rc
                ON tc.constraint_name  = rc.constraint_name
                AND tc.constraint_schema = rc.constraint_schema
            JOIN information_schema.key_column_usage kcu_ref
                ON rc.unique_constraint_name  = kcu_ref.constraint_name
                AND rc.unique_constraint_schema = kcu_ref.constraint_schema
                AND kcu_src.position_in_unique_constraint = kcu_ref.ordinal_position
            WHERE tc.constraint_type = 'FOREIGN KEY'
            ORDER BY kcu_src.table_schema, kcu_src.table_name,
                     tc.constraint_name, kcu_src.ordinal_position
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()

        return [
            ForeignKeyInfo(
                constraint_name=row["constraint_name"],
                source_schema=row["source_schema"],
                source_table=row["source_table"],
                source_column=row["source_column"],
                target_schema=row["target_schema"],
                target_table=row["target_table"],
                target_column=row["target_column"],
            )
            for row in rows
        ]

    def get_table_row_count(self, table_name: str, schema_name: str) -> int:
        query = text("""\
            SELECT n_live_tup AS approx_count
            FROM pg_stat_user_tables
            WHERE schemaname = :schema_name
                AND relname = :table_name
        """)

        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    query,
                    {"table_name": table_name, "schema_name": schema_name},
                )
                .mappings()
                .first()
            )

        if row is None:
            return 0
        return int(row["approx_count"])

    def get_column_stats(
        self, table_name: str, column_name: str, schema_name: str
    ) -> ColumnStats:
        fqn = f"{_quote_ident(schema_name)}.{_quote_ident(table_name)}"
        col = _quote_ident(column_name)

        stats_query = text(f"""\
            SELECT
                COUNT(DISTINCT {col})                       AS distinct_count,
                MIN({col})::text                            AS min_val,
                MAX({col})::text                            AS max_val,
                ROUND(100.0 * SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) / GREATEST(COUNT(*), 1), 2)
                                                            AS null_pct,
                COUNT(*)                                    AS total_rows
            FROM {fqn}
        """)

        with self.engine.connect() as conn:
            row = conn.execute(stats_query).mappings().first()

        if row is None:
            return ColumnStats()

        distinct_count = int(row["distinct_count"])
        sample_values: list[str] | None = None

        if distinct_count <= LOW_CARDINALITY_THRESHOLD:
            sample_query = text(
                f"SELECT DISTINCT {col}::text AS val FROM {fqn} WHERE {col} IS NOT NULL ORDER BY val LIMIT {LOW_CARDINALITY_THRESHOLD}"
            )
            with self.engine.connect() as conn:
                sample_rows = conn.execute(sample_query).mappings().all()
            sample_values = [r["val"] for r in sample_rows]

        return ColumnStats(
            distinct_count=distinct_count,
            min_value=row["min_val"],
            max_value=row["max_val"],
            null_percentage=float(row["null_pct"])
            if row["null_pct"] is not None
            else 0.0,
            sample_distinct_values=sample_values,
        )

    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        return [dict(row) for row in rows]

    def _schema_visible(self, schema_name: str, schema_filter: SchemaFilter) -> bool:
        if schema_filter.include and schema_name not in schema_filter.include:
            return False
        if schema_filter.exclude and schema_name in schema_filter.exclude:
            return False
        return schema_name not in ("pg_catalog", "information_schema")

    def get_table_grants(
        self, schema_filter: SchemaFilter | None = None
    ) -> list[TableGrant]:
        """Return SELECT grants with grantees expanded along PostgreSQL role membership."""
        sf = schema_filter or self.config.schema_filter

        select_grants_sql = text("""\
            SELECT grantee, table_schema, table_name
            FROM information_schema.table_privileges
            WHERE privilege_type = 'SELECT'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
        """)
        membership_sql = text("""\
            SELECT m.rolname AS member_name, r.rolname AS role_name
            FROM pg_catalog.pg_auth_members am
            JOIN pg_catalog.pg_roles m ON m.oid = am.member
            JOIN pg_catalog.pg_roles r ON r.oid = am.roleid
        """)

        try:
            with self.engine.connect() as conn:
                priv_rows = conn.execute(select_grants_sql).mappings().all()
        except SQLAlchemyError as exc:
            logger.warning(
                "Could not read SELECT grants from information_schema.table_privileges: %s",
                exc,
            )
            return []

        direct_keys: set[tuple[str, str, str]] = set()
        direct: list[tuple[str, str, str]] = []
        for row in priv_rows:
            ge = row["grantee"]
            ts = row["table_schema"]
            tn = row["table_name"]
            if ge is None or ts is None or tn is None:
                continue
            grantee = str(ge)
            schema_name = str(ts)
            table_name = str(tn)
            if not self._schema_visible(schema_name, sf):
                continue
            if grantee == "PUBLIC":
                continue
            dk = (grantee, schema_name, table_name)
            if dk in direct_keys:
                continue
            direct_keys.add(dk)
            direct.append(dk)

        children: dict[str, set[str]] = {}
        try:
            with self.engine.connect() as conn:
                for row in conn.execute(membership_sql).mappings().all():
                    mname = row["member_name"]
                    rname = row["role_name"]
                    if mname is None or rname is None:
                        continue
                    parent = str(rname)
                    member = str(mname)
                    children.setdefault(parent, set()).add(member)
        except SQLAlchemyError as exc:
            # Falling back to direct grants only — an admin role without
            # pg_catalog access still gets useful output instead of an empty
            # visibility.yml. ``_pg_transitive_members`` with an empty children
            # map yields ``{grantee}``, so the downstream loop preserves each
            # direct grant one-to-one.
            logger.warning(
                "Could not read role membership from pg_auth_members "
                "(returning direct grants only, no inherited-role expansion): %s",
                exc,
            )
            children = {}

        seen: set[tuple[str, str, str]] = set()
        out: list[TableGrant] = []
        for grantee, schema_name, table_name in direct:
            for eff in _pg_transitive_members(children, grantee):
                key = (eff, schema_name, table_name)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    TableGrant(
                        grantee=eff,
                        schema_name=schema_name,
                        table_name=table_name,
                    )
                )
        out.sort(key=lambda tg: (tg.schema_name, tg.table_name, tg.grantee))
        return out

    def load_view_dependencies(self, schema_filter: SchemaFilter) -> list[ViewDependency]:
        """Lineage from views, materialized views, and cross-table triggers."""
        out: list[ViewDependency] = []
        out.extend(self._load_pg_view_sql_lineage(schema_filter))
        out.extend(self._load_pg_trigger_lineage(schema_filter))
        return out

    def _load_pg_view_sql_lineage(self, schema_filter: SchemaFilter) -> list[ViewDependency]:
        deps: list[ViewDependency] = []
        views_sql = text("""\
            SELECT schemaname, viewname, definition
            FROM pg_catalog.pg_views
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        """)
        mat_sql = text("""\
            SELECT schemaname, matviewname, definition
            FROM pg_catalog.pg_matviews
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        """)
        with self.engine.connect() as conn:
            view_rows = conn.execute(views_sql).mappings().all()
            mat_rows = conn.execute(mat_sql).mappings().all()
        for row in view_rows:
            ts = str(row["schemaname"])
            if not self._schema_visible(ts, schema_filter):
                continue
            tname = str(row["viewname"])
            definition = row["definition"]
            if definition is None:
                continue
            sql_text = str(definition)
            for src_schema, src_table in table_refs_from_sql(
                sql_text, dialect="postgres", default_schema=ts
            ):
                if not self._schema_visible(src_schema, schema_filter):
                    continue
                if src_schema == ts and src_table == tname:
                    continue
                deps.append(
                    ViewDependency(
                        source_schema=src_schema,
                        source_table=src_table,
                        target_schema=ts,
                        target_table=tname,
                        lineage_type="VIEW",
                        object_name=f"{ts}.{tname}",
                        confidence=1.0,
                    )
                )
        for row in mat_rows:
            ts = str(row["schemaname"])
            if not self._schema_visible(ts, schema_filter):
                continue
            tname = str(row["matviewname"])
            definition = row["definition"]
            if definition is None:
                continue
            sql_text = str(definition)
            for src_schema, src_table in table_refs_from_sql(
                sql_text, dialect="postgres", default_schema=ts
            ):
                if not self._schema_visible(src_schema, schema_filter):
                    continue
                if src_schema == ts and src_table == tname:
                    continue
                deps.append(
                    ViewDependency(
                        source_schema=src_schema,
                        source_table=src_table,
                        target_schema=ts,
                        target_table=tname,
                        lineage_type="MATERIALIZED_VIEW",
                        object_name=f"{ts}.{tname}",
                        confidence=1.0,
                    )
                )
        return deps

    def _load_pg_trigger_lineage(self, schema_filter: SchemaFilter) -> list[ViewDependency]:
        deps: list[ViewDependency] = []
        trig_sql = text("""\
            SELECT t.tgname AS trigger_name,
                   n.nspname AS source_schema,
                   c.relname AS source_table,
                   pg_get_functiondef(p.oid) AS func_def,
                   l.lanname AS func_language
            FROM pg_catalog.pg_trigger t
            JOIN pg_catalog.pg_class c ON t.tgrelid = c.oid
            JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
            JOIN pg_catalog.pg_proc p ON t.tgfoid = p.oid
            JOIN pg_catalog.pg_language l ON p.prolang = l.oid
            WHERE NOT t.tgisinternal
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(trig_sql).mappings().all()
        for row in rows:
            source_schema = str(row["source_schema"])
            if not self._schema_visible(source_schema, schema_filter):
                continue
            source_table = str(row["source_table"])
            tgname = str(row["trigger_name"])
            func_def = row["func_def"]
            if func_def is None:
                continue
            body = str(func_def)
            if "EXECUTE" in body.upper():
                continue
            lang = str(row["func_language"] or "").lower()
            if lang != "sql":
                # sqlglot cannot parse pg_get_functiondef output for plpgsql/c/
                # internal function bodies; it falls back to a Command node
                # which yields zero write targets and emits a noisy warning
                # per trigger row. Skip non-SQL languages entirely.
                continue
            conf = 0.85
            targets: set[tuple[str, str]] = set()
            for ts, tt in dml_write_targets(
                body, dialect="postgres", default_schema=source_schema
            ):
                targets.add((ts, tt))
            for tgt_schema, tgt_table in targets:
                if not self._schema_visible(tgt_schema, schema_filter):
                    continue
                if tgt_schema == source_schema and tgt_table == source_table:
                    continue
                deps.append(
                    ViewDependency(
                        source_schema=source_schema,
                        source_table=source_table,
                        target_schema=tgt_schema,
                        target_table=tgt_table,
                        lineage_type="TRIGGER",
                        object_name=f"{source_schema}.{source_table}:{tgname}",
                        confidence=conf,
                    )
                )
        return deps

    def load_deep_catalog(
        self, schema_filter: SchemaFilter
    ) -> tuple[
        dict[tuple[str, str], dict[str, Any]],
        dict[tuple[str, str, str], dict[str, Any]],
    ]:
        """Usage stats, partitions, grants (``SchemaTable``) and ``pg_stats`` (columns)."""
        table_extra: dict[tuple[str, str], dict[str, Any]] = {}
        column_extra: dict[tuple[str, str, str], dict[str, Any]] = {}

        usage_sql = text("""\
            SELECT schemaname, relname,
                   seq_scan, idx_scan, n_tup_ins, n_tup_upd, n_tup_del
            FROM pg_stat_user_tables
        """)
        part_sql = text("""\
            SELECT n.nspname AS table_schema,
                   c.relname AS table_name,
                   a.attname AS partition_column
            FROM pg_catalog.pg_partitioned_table pt
            JOIN pg_catalog.pg_class c ON c.oid = pt.partrelid
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_catalog.pg_attribute a
              ON a.attrelid = c.oid
             AND a.attnum = pt.partattrs[1]  -- first key column only; multi-column keys are truncated
             AND NOT a.attisdropped
        """)
        grant_sql = text("""\
            SELECT table_schema, table_name, grantee, privilege_type
            FROM information_schema.role_table_grants
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        """)
        stats_sql = text("""\
            SELECT schemaname, tablename, attname,
                   most_common_vals::text AS mcv_text,
                   histogram_bounds::text AS hb_text,
                   correlation,
                   n_distinct,
                   null_frac
            FROM pg_stats
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        """)
        index_sql = text("""\
            WITH idx_cols AS (
                SELECT n.nspname AS schemaname,
                       t.relname AS tablename,
                       a.attname AS attname,
                       am.amname AS index_type,
                       ix.indisunique AS index_is_unique,
                       ROW_NUMBER() OVER (
                           PARTITION BY t.oid, a.attnum
                           ORDER BY ix.indisunique DESC,
                                    (am.amname = 'btree') DESC
                       ) AS rn
                FROM pg_index ix
                JOIN pg_class t ON t.oid = ix.indrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                JOIN pg_class ic ON ic.oid = ix.indexrelid
                JOIN pg_am am ON am.oid = ic.relam
                JOIN pg_attribute a
                  ON a.attrelid = t.oid
                 AND a.attnum = ANY(ix.indkey)
                 AND a.attnum > 0
                 AND NOT a.attisdropped
                WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
            )
            SELECT schemaname, tablename, attname, index_type, index_is_unique
            FROM idx_cols
            WHERE rn = 1
        """)

        def _schema_ok(sn: str) -> bool:
            if schema_filter.include and sn not in schema_filter.include:
                return False
            if schema_filter.exclude and sn in schema_filter.exclude:
                return False
            return True

        with self.engine.connect() as conn:
            _t = time.perf_counter() if _PROFILE_INDEX else 0.0
            usage_rows = conn.execute(usage_sql).mappings().all()
            if _PROFILE_INDEX:
                print(
                    f"[profile] load_deep_catalog.usage_sql: {(time.perf_counter() - _t) * 1000:.0f}ms ({len(usage_rows)} rows)",
                    flush=True,
                )
            for row in usage_rows:
                sn = str(row["schemaname"])
                tn = str(row["relname"])
                if not _schema_ok(sn):
                    continue
                key = (sn, tn)
                table_extra.setdefault(key, {})
                table_extra[key].update(
                    {
                        "seq_scan_count": int(row["seq_scan"] or 0),
                        "idx_scan_count": int(row["idx_scan"] or 0),
                        "insert_count": int(row["n_tup_ins"] or 0),
                        "update_count": int(row["n_tup_upd"] or 0),
                        "delete_count": int(row["n_tup_del"] or 0),
                    }
                )

            _t = time.perf_counter() if _PROFILE_INDEX else 0.0
            part_rows = conn.execute(part_sql).mappings().all()
            if _PROFILE_INDEX:
                print(
                    f"[profile] load_deep_catalog.part_sql: {(time.perf_counter() - _t) * 1000:.0f}ms ({len(part_rows)} rows)",
                    flush=True,
                )
            for row in part_rows:
                sn = str(row["table_schema"])
                tn = str(row["table_name"])
                if not _schema_ok(sn):
                    continue
                key = (sn, tn)
                table_extra.setdefault(key, {})
                table_extra[key]["is_partitioned"] = True
                pc = row["partition_column"]
                if pc is not None:
                    table_extra[key]["partition_key"] = str(pc)

            grants_by_table: dict[tuple[str, str], list[dict[str, str]]] = {}
            _t = time.perf_counter() if _PROFILE_INDEX else 0.0
            grant_rows = conn.execute(grant_sql).mappings().all()
            if _PROFILE_INDEX:
                print(
                    f"[profile] load_deep_catalog.grant_sql: {(time.perf_counter() - _t) * 1000:.0f}ms ({len(grant_rows)} rows)",
                    flush=True,
                )
            for row in grant_rows:
                sn = str(row["table_schema"])
                tn = str(row["table_name"])
                if not _schema_ok(sn):
                    continue
                ge = row["grantee"]
                priv = row["privilege_type"]
                if ge is None or priv is None:
                    continue
                gkey = (sn, tn)
                grants_by_table.setdefault(gkey, []).append(
                    {"grantee": str(ge), "privilege": str(priv)}
                )
            for gkey, grants in grants_by_table.items():
                table_extra.setdefault(gkey, {})
                table_extra[gkey]["grants"] = grants

            _t = time.perf_counter() if _PROFILE_INDEX else 0.0
            stats_rows = conn.execute(stats_sql).mappings().all()
            if _PROFILE_INDEX:
                print(
                    f"[profile] load_deep_catalog.stats_sql: {(time.perf_counter() - _t) * 1000:.0f}ms ({len(stats_rows)} rows)",
                    flush=True,
                )
            for row in stats_rows:
                sn = str(row["schemaname"])
                tn = str(row["tablename"])
                cn = str(row["attname"])
                if not _schema_ok(sn):
                    continue
                ckey = (sn, tn, cn)
                mcv_raw = row["mcv_text"]
                hb_raw = row["hb_text"]
                corr = row["correlation"]
                entry: dict[str, Any] = {}
                if mcv_raw:
                    parsed = parse_pg_array_literal(str(mcv_raw))
                    if parsed:
                        entry["most_common_values"] = parsed[:PG_STATS_MOST_COMMON_CAP]
                if hb_raw:
                    hb_parsed = parse_pg_array_literal(str(hb_raw))
                    if hb_parsed:
                        entry["histogram_bounds"] = hb_parsed
                if corr is not None:
                    entry["stats_correlation"] = float(corr)
                n_distinct = row["n_distinct"]
                if n_distinct is not None:
                    entry["column_cardinality"] = int(round(float(n_distinct)))
                null_frac = row["null_frac"]
                if null_frac is not None:
                    entry["null_percentage"] = float(null_frac) * 100.0
                if entry:
                    column_extra[ckey] = entry

            _t = time.perf_counter() if _PROFILE_INDEX else 0.0
            index_rows = conn.execute(index_sql).mappings().all()
            if _PROFILE_INDEX:
                print(
                    f"[profile] load_deep_catalog.index_sql: {(time.perf_counter() - _t) * 1000:.0f}ms ({len(index_rows)} rows)",
                    flush=True,
                )
            for row in index_rows:
                sn = str(row["schemaname"])
                tn = str(row["tablename"])
                cn = str(row["attname"])
                if not _schema_ok(sn):
                    continue
                ckey = (sn, tn, cn)
                column_extra.setdefault(ckey, {})
                column_extra[ckey]["index_type"] = str(row["index_type"])
                column_extra[ckey]["index_is_unique"] = bool(row["index_is_unique"])

        return table_extra, column_extra


def _quote_ident(identifier: str) -> str:
    """Quote a SQL identifier to prevent injection. Double any internal quotes."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'
