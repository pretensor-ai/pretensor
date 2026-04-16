"""Parse SQL with sqlglot and validate table/column/join references against Kuzu."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from pretensor.core.store import KuzuStore

__all__ = [
    "JoinWarning",
    "QueryValidator",
    "ValidationResult",
]

_DEFAULT_DIALECT = "postgres"
_FUZZY_CUTOFF = 0.5


@dataclass
class JoinWarning:
    """A join whose endpoints are not linked by FK or inferred join in the graph."""

    message: str
    left_table: str
    right_table: str


@dataclass
class ValidationResult:
    """Outcome of semantic validation for a single SQL string."""

    valid: bool
    syntax_errors: list[str] = field(default_factory=list)
    missing_tables: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    invalid_joins: list[JoinWarning] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class QueryValidator:
    """Validate SQL against ``SchemaTable`` / ``SchemaColumn`` / join edges in Kuzu."""

    def __init__(
        self,
        store: KuzuStore,
        *,
        connection_name: str,
        database_key: str,
        dialect: str = _DEFAULT_DIALECT,
    ) -> None:
        """Args:
        store: Open Kuzu store for one indexed connection.
        connection_name: ``SchemaTable.connection_name`` filter (registry connection).
        database_key: ``SchemaTable.database`` filter (logical DB; same as MCP ``db`` key).
        dialect: sqlglot dialect for parsing (default PostgreSQL).
        """
        self._store = store
        self._connection_name = connection_name
        self._database_key = database_key
        self._dialect = dialect

    def validate(self, sql: str) -> ValidationResult:
        """Parse SQL, then check tables, columns, and join relationships in the graph."""
        try:
            parsed = sqlglot.parse_one(sql, dialect=self._dialect)
        except ParseError as exc:
            return ValidationResult(valid=False, syntax_errors=[str(exc)])

        syntax_errors: list[str] = []
        missing_tables: list[str] = []
        missing_columns: list[str] = []
        invalid_joins: list[JoinWarning] = []
        suggestions: list[str] = []

        alias_map = _build_alias_map(parsed)
        tables_in_query = _tables_referenced_in_query(parsed)

        physical_by_qual: dict[str, tuple[str, str]] = {}
        for qual in sorted(tables_in_query):
            phys = self._resolve_physical_table(qual)
            if phys is None:
                missing_tables.append(qual)
                suggestions.extend(self._suggest_tables(qual))
            else:
                physical_by_qual[qual] = phys

        if missing_tables:
            return ValidationResult(
                valid=False,
                syntax_errors=syntax_errors,
                missing_tables=missing_tables,
                missing_columns=missing_columns,
                invalid_joins=invalid_joins,
                suggestions=suggestions,
            )

        for column_ref in _iter_column_refs(parsed):
            table_qual = _column_table_qualifier(
                column_ref, alias_map, physical_by_qual
            )
            if table_qual is None:
                continue
            if table_qual not in physical_by_qual:
                continue
            schema_name, table_name = physical_by_qual[table_qual]
            col_name = column_ref.name
            if not self._column_exists(schema_name, table_name, col_name):
                key = f"{schema_name}.{table_name}.{col_name}"
                missing_columns.append(key)
                hint = self._suggest_column(schema_name, table_name, col_name)
                if hint:
                    suggestions.append(
                        f"column `{schema_name}.{table_name}.{col_name}` not found in graph; "
                        f"did you mean `{hint}`?"
                    )

        seen_bad_pairs: set[tuple[str, str]] = set()
        for join in _iter_joins(parsed):
            on_expr = join.args.get("on")
            if on_expr is None:
                continue
            physical_tables: set[tuple[str, str]] = set()
            for col in on_expr.find_all(exp.Column):
                tq = _column_table_qualifier(col, alias_map, physical_by_qual)
                if tq is None or tq not in physical_by_qual:
                    continue
                physical_tables.add(physical_by_qual[tq])
            if len(physical_tables) < 2:
                continue
            pairs = list(physical_tables)
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    a, b = pairs[i], pairs[j]
                    if self._tables_joinable(a, b):
                        continue
                    left_qual = f"{a[0]}.{a[1]}"
                    right_qual = f"{b[0]}.{b[1]}"
                    ordered = sorted((left_qual, right_qual))
                    pair_key: tuple[str, str] = (ordered[0], ordered[1])
                    if pair_key in seen_bad_pairs:
                        continue
                    seen_bad_pairs.add(pair_key)
                    left_s, left_t = a
                    right_s, right_t = b
                    invalid_joins.append(
                        JoinWarning(
                            message=(
                                f"no FK_REFERENCES or INFERRED_JOIN in graph between "
                                f"`{left_s}.{left_t}` and `{right_s}.{right_t}`"
                            ),
                            left_table=f"{left_s}.{left_t}",
                            right_table=f"{right_s}.{right_t}",
                        )
                    )
                    alt = self._join_suggestion(a, b)
                    if alt:
                        suggestions.append(alt)

        has_semantic_issue = bool(missing_columns or invalid_joins)
        return ValidationResult(
            valid=not has_semantic_issue,
            syntax_errors=syntax_errors,
            missing_tables=missing_tables,
            missing_columns=missing_columns,
            invalid_joins=invalid_joins,
            suggestions=suggestions,
        )

    def _resolve_physical_table(self, qualified: str) -> tuple[str, str] | None:
        """Map ``schema.table`` or bare ``table`` to ``(schema_name, table_name)`` in Kuzu."""
        if "." in qualified:
            schema_name, _, table_name = qualified.partition(".")
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
                    "sn": schema_name,
                    "tn": table_name,
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
                    "tbl": qualified,
                },
            )
            if len(rows) > 1:
                return None
        if not rows:
            return None
        sn, tn = rows[0]
        return (str(sn), str(tn))

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

    def _tables_joinable(
        self,
        left: tuple[str, str],
        right: tuple[str, str],
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

    def _suggest_tables(self, qualified: str) -> list[str]:
        if "." in qualified:
            _, _, tbl = qualified.partition(".")
        else:
            tbl = qualified
        rows = self._store.query_all_rows(
            """
            MATCH (t:SchemaTable)
            WHERE t.connection_name = $cn AND t.database = $db
            RETURN t.schema_name, t.table_name
            """,
            {"cn": self._connection_name, "db": self._database_key},
        )
        names = [f"{sn}.{tn}" for sn, tn in rows]
        matches = get_close_matches(
            tbl, [r[1] for r in rows], n=3, cutoff=_FUZZY_CUTOFF
        )
        out: list[str] = []
        for m in matches:
            for sn, tn in rows:
                if str(tn) == m:
                    out.append(
                        f"table `{qualified}` not in graph; did you mean `{sn}.{tn}`?"
                    )
                    break
        if not out and names:
            out.append(
                f"table `{qualified}` not in graph; known tables include "
                f"{', '.join(names[:5])}{'…' if len(names) > 5 else ''}"
            )
        return out

    def _suggest_column(
        self, schema_name: str, table_name: str, column_name: str
    ) -> str | None:
        rows = self._store.query_all_rows(
            """
            MATCH (t:SchemaTable)-[:HAS_COLUMN]->(c:SchemaColumn)
            WHERE t.connection_name = $cn AND t.database = $db
              AND t.schema_name = $sn AND t.table_name = $tn
            RETURN c.column_name
            """,
            {
                "cn": self._connection_name,
                "db": self._database_key,
                "sn": schema_name,
                "tn": table_name,
            },
        )
        names = [str(r[0]) for r in rows]
        matches = get_close_matches(column_name, names, n=1, cutoff=_FUZZY_CUTOFF)
        if not matches:
            return None
        return f"{schema_name}.{table_name}.{matches[0]}"

    def _join_suggestion(
        self, left: tuple[str, str], right: tuple[str, str]
    ) -> str | None:
        """If a single-hop path exists via an intermediate table, mention it."""
        rows = self._store.query_all_rows(
            """
            MATCH (a:SchemaTable), (b:SchemaTable), (mid:SchemaTable)
            WHERE a.connection_name = $cn AND a.database = $db
              AND b.connection_name = $cn AND b.database = $db
              AND mid.connection_name = $cn AND mid.database = $db
              AND a.schema_name = $lsa AND a.table_name = $lta
              AND b.schema_name = $rsb AND b.table_name = $rtb
              AND a <> mid AND b <> mid AND a <> b
              AND (
                ((a)-[:FK_REFERENCES|INFERRED_JOIN]-(mid)) AND
                ((mid)-[:FK_REFERENCES|INFERRED_JOIN]-(b))
              )
            RETURN mid.schema_name, mid.table_name
            LIMIT 1
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
            return None
        ms, mt = rows[0]
        return (
            f"no direct graph edge between `{left[0]}.{left[1]}` and `{right[0]}.{right[1]}`; "
            f"try joining through `{ms}.{mt}` if that matches your model"
        )


def _build_alias_map(root: Any) -> dict[str, str]:
    """Map SQL alias → qualified ``schema.table`` or ``table`` string."""
    mapping: dict[str, str] = {}
    for table in root.find_all(exp.Table):
        qual = _table_qualifier(table)
        alias = _table_alias(table)
        if alias:
            mapping[alias.lower()] = qual
        mapping[qual.lower()] = qual
    return mapping


def _table_qualifier(table: exp.Table) -> str:
    table_name = table.name
    db = table.args.get("db")
    schema_name = db.name if isinstance(db, exp.Identifier) else None
    if schema_name:
        return f"{schema_name}.{table_name}"
    return table_name


def _table_alias(table: exp.Table) -> str | None:
    alias = table.args.get("alias")
    if alias is None:
        return None
    if isinstance(alias, exp.TableAlias):
        ident = alias.this
        if isinstance(ident, exp.Identifier):
            return ident.name
    return None


def _tables_referenced_in_query(parsed: Any) -> set[str]:
    seen: set[str] = set()
    for table in parsed.find_all(exp.Table):
        seen.add(_table_qualifier(table))
    return seen


def _iter_joins(parsed: Any) -> list[exp.Join]:
    joins: list[exp.Join] = []
    select = parsed.find(exp.Select)
    if not select:
        return joins
    for j in select.args.get("joins") or []:
        if isinstance(j, exp.Join):
            joins.append(j)
    return joins


def _iter_column_refs(parsed: Any) -> list[exp.Column]:
    return [c for c in parsed.find_all(exp.Column) if isinstance(c, exp.Column)]


def _column_table_qualifier(
    column: exp.Column,
    alias_map: dict[str, str],
    physical_by_qual: dict[str, tuple[str, str]],
) -> str | None:
    table_ref = column.table
    if table_ref is None:
        return None
    if isinstance(table_ref, exp.Identifier):
        key = table_ref.name.lower()
    elif isinstance(table_ref, str):
        key = table_ref.lower()
    else:
        return None
    if key not in alias_map:
        return None
    raw = alias_map[key]
    # Normalize to a key present in physical_by_qual (case-insensitive match).
    for q in physical_by_qual:
        if q.lower() == raw.lower():
            return q
    return raw
