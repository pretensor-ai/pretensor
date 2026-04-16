"""Compile user-authored YAML metrics to validated SQL (OSS).

The :class:`MetricSqlCompiler` turns a :class:`Metric` from a
:class:`SemanticLayer` YAML (``src/pretensor/introspection/models/semantic.py``)
into an ANSI-quoted SQL string and runs it through the existing
:class:`pretensor.validation.query_validator.QueryValidator`. See
``docs/specs/semantic/spec.md`` for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from pretensor.core.store import KuzuStore
from pretensor.introspection.models.semantic import (
    Entity,
    Metric,
    MetricType,
)
from pretensor.introspection.models.semantic import (
    SemanticLayer as SemanticLayerModel,
)
from pretensor.validation.query_validator import QueryValidator, ValidationResult

__all__ = ["CompiledMetric", "MetricCompileError", "MetricSqlCompiler"]


_DEFAULT_DIALECT = "postgres"


def _pg_ident(part: str) -> str:
    """ANSI / PostgreSQL double-quote identifier escaping."""
    return '"' + part.replace('"', '""') + '"'


def _split_schema_table(qualified: str) -> tuple[str | None, str]:
    """Split ``schema.table`` into ``(schema, table)`` — ``(None, table)`` when unqualified."""
    if "." in qualified:
        schema, _, table = qualified.partition(".")
        return schema, table
    return None, qualified


@dataclass
class CompiledMetric:
    """Outcome of compiling a single metric."""

    metric: str
    entity: str
    sql: str
    dialect: str
    warnings: list[str] = field(default_factory=list)
    validation: ValidationResult = field(default_factory=lambda: ValidationResult(valid=True))


class MetricCompileError(ValueError):
    """Raised when a metric cannot be compiled (missing field, unknown table, parse error)."""


class MetricSqlCompiler:
    """Compile YAML-authored :class:`Metric` objects to SQL.

    The compiler resolves the owning :class:`Entity` source table through the
    same Kuzu lookup the validator uses so callers cannot compile metrics
    against tables that are not in the indexed graph.
    """

    def __init__(
        self,
        store: KuzuStore,
        *,
        connection_name: str,
        database_key: str,
        dialect: str = _DEFAULT_DIALECT,
    ) -> None:
        self._store = store
        self._connection_name = connection_name
        self._database_key = database_key
        self._dialect = dialect
        self._validator = QueryValidator(
            store,
            connection_name=connection_name,
            database_key=database_key,
            dialect=dialect,
        )

    def compile(
        self,
        layer: SemanticLayerModel,
        metric_name: str,
    ) -> CompiledMetric:
        """Compile ``metric_name`` from ``layer`` to SQL and validate.

        Raises:
            MetricCompileError: when the metric is not found, its entity is
                missing, its field is unknown, its expression is unparseable,
                or the owning table cannot be resolved to an indexed table.
        """
        entity, metric = self._find_metric(layer, metric_name)
        warnings: list[str] = []

        if metric.type is MetricType.DERIVED:
            sql = self._compile_derived(metric, warnings)
        else:
            sql = self._compile_aggregate(entity, metric)

        validation = self._validator.validate(sql)
        return CompiledMetric(
            metric=metric.name,
            entity=entity.name,
            sql=sql,
            dialect=self._dialect,
            warnings=warnings,
            validation=validation,
        )

    # ---- internals ---------------------------------------------------------

    def _find_metric(
        self, layer: SemanticLayerModel, metric_name: str
    ) -> tuple[Entity, Metric]:
        for domain in layer.domains:
            for entity in domain.entities:
                for metric in entity.metrics:
                    if metric.name == metric_name:
                        return entity, metric
        raise MetricCompileError(
            f"metric {metric_name!r} not found in semantic layer "
            f"for connection {layer.connection_name!r}"
        )

    def _resolve_entity_table(self, entity: Entity) -> tuple[str, str]:
        schema, table = _split_schema_table(entity.source_table)
        if schema is not None:
            rows = self._store.query_all_rows(
                """
                MATCH (t:SchemaTable)
                WHERE t.connection_name = $cn AND t.database = $db
                  AND t.schema_name = $sn AND t.table_name = $tn
                RETURN t.schema_name, t.table_name
                LIMIT 1
                """,
                {
                    "cn": self._connection_name,
                    "db": self._database_key,
                    "sn": schema,
                    "tn": table,
                },
            )
        else:
            rows = self._store.query_all_rows(
                """
                MATCH (t:SchemaTable)
                WHERE t.connection_name = $cn AND t.database = $db
                  AND t.table_name = $tbl
                RETURN t.schema_name, t.table_name
                """,
                {
                    "cn": self._connection_name,
                    "db": self._database_key,
                    "tbl": table,
                },
            )
            if len(rows) > 1:
                raise MetricCompileError(
                    f"entity {entity.name!r} source table {entity.source_table!r} "
                    f"is ambiguous across schemas; qualify it as `schema.table`"
                )
        if not rows:
            raise MetricCompileError(
                f"entity {entity.name!r} source table {entity.source_table!r} "
                f"is not indexed for connection {self._connection_name!r}"
            )
        sn, tn = rows[0]
        return str(sn), str(tn)

    def _column_exists(
        self, schema_name: str, table_name: str, column_name: str
    ) -> bool:
        rows = self._store.query_all_rows(
            """
            MATCH (t:SchemaTable)-[:HAS_COLUMN]->(c:SchemaColumn)
            WHERE t.connection_name = $cn AND t.database = $db
              AND t.schema_name = $sn AND t.table_name = $tn
              AND c.column_name = $col
            RETURN 1 LIMIT 1
            """,
            {
                "cn": self._connection_name,
                "db": self._database_key,
                "sn": schema_name,
                "tn": table_name,
                "col": column_name,
            },
        )
        return bool(rows)

    def _compile_aggregate(self, entity: Entity, metric: Metric) -> str:
        if metric.field is None:
            raise MetricCompileError(
                f"metric {metric.name!r} of type {metric.type.value!r} "
                f"requires `field`"
            )
        schema_name, table_name = self._resolve_entity_table(entity)
        if not self._column_exists(schema_name, table_name, metric.field):
            raise MetricCompileError(
                f"metric {metric.name!r} field {metric.field!r} not found on "
                f"{schema_name}.{table_name} in the indexed graph"
            )

        qschema = _pg_ident(schema_name)
        qtable = _pg_ident(table_name)
        qcol = _pg_ident(metric.field)
        qalias = _pg_ident(metric.name)

        if metric.type is MetricType.COUNT:
            agg = f"COUNT({qcol})"
        elif metric.type is MetricType.COUNT_DISTINCT:
            agg = f"COUNT(DISTINCT {qcol})"
        elif metric.type is MetricType.SUM:
            agg = f"SUM({qcol})"
        elif metric.type is MetricType.AVERAGE:
            agg = f"AVG({qcol})"
        else:  # pragma: no cover — DERIVED routed elsewhere
            raise MetricCompileError(
                f"unsupported aggregate type {metric.type.value!r}"
            )

        sql = f"SELECT {agg} AS {qalias} FROM {qschema}.{qtable}"
        if metric.filters:
            joined = " AND ".join(f"({f})" for f in metric.filters)
            sql = f"{sql} WHERE {joined}"
        return sql

    def _compile_derived(self, metric: Metric, warnings: list[str]) -> str:
        if not metric.expression:
            raise MetricCompileError(
                f"metric {metric.name!r} of type 'derived' requires `expression`"
            )
        try:
            parsed = sqlglot.parse_one(metric.expression, dialect=self._dialect)
        except ParseError as exc:
            raise MetricCompileError(
                f"metric {metric.name!r} expression failed to parse: {exc}"
            ) from exc

        tables: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for table in parsed.find_all(exp.Table):
            name = table.name
            db = table.args.get("db")
            schema = db.name if isinstance(db, exp.Identifier) else None
            qual = f"{schema}.{name}" if schema else name
            resolved = self._resolve_qualified(qual)
            if resolved is None:
                raise MetricCompileError(
                    f"metric {metric.name!r} expression references unknown "
                    f"table {qual!r}"
                )
            if resolved not in seen:
                seen.add(resolved)
                tables.append(resolved)

        # Warn about table pairs without a graph-backed join edge.
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                a, b = tables[i], tables[j]
                if not self._tables_joinable(a, b):
                    warnings.append(
                        f"no FK_REFERENCES or INFERRED_JOIN in graph between "
                        f"`{a[0]}.{a[1]}` and `{b[0]}.{b[1]}`"
                    )
        return metric.expression.strip()

    def _resolve_qualified(self, qualified: str) -> tuple[str, str] | None:
        schema, table = _split_schema_table(qualified)
        if schema is not None:
            rows = self._store.query_all_rows(
                """
                MATCH (t:SchemaTable)
                WHERE t.connection_name = $cn AND t.database = $db
                  AND t.schema_name = $sn AND t.table_name = $tn
                RETURN t.schema_name, t.table_name
                LIMIT 1
                """,
                {
                    "cn": self._connection_name,
                    "db": self._database_key,
                    "sn": schema,
                    "tn": table,
                },
            )
        else:
            rows = self._store.query_all_rows(
                """
                MATCH (t:SchemaTable)
                WHERE t.connection_name = $cn AND t.database = $db
                  AND t.table_name = $tbl
                RETURN t.schema_name, t.table_name
                """,
                {
                    "cn": self._connection_name,
                    "db": self._database_key,
                    "tbl": table,
                },
            )
            if len(rows) > 1:
                return None
        if not rows:
            return None
        sn, tn = rows[0]
        return (str(sn), str(tn))

    def _tables_joinable(
        self, left: tuple[str, str], right: tuple[str, str]
    ) -> bool:
        rows = self._store.query_all_rows(
            """
            MATCH (a:SchemaTable), (b:SchemaTable)
            WHERE a.connection_name = $cn AND a.database = $db
              AND b.connection_name = $cn AND b.database = $db
              AND a.schema_name = $lsa AND a.table_name = $lta
              AND b.schema_name = $rsb AND b.table_name = $rtb
              AND (
                (a)-[:FK_REFERENCES]->(b) OR (b)-[:FK_REFERENCES]->(a)
                OR (a)-[:INFERRED_JOIN]->(b) OR (b)-[:INFERRED_JOIN]->(a)
              )
            RETURN count(*) AS n
            """,
            {
                "cn": self._connection_name,
                "db": self._database_key,
                "lsa": left[0],
                "lta": left[1],
                "rsb": right[0],
                "rtb": right[1],
            },
        )
        if not rows:
            return False
        n = rows[0][0]
        return int(n) > 0 if n is not None else False
