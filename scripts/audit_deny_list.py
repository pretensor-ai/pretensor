"""Audit candidate entries for `_BORING_SHARED_NAMES`.

Parses the AdventureWorks and Pagila DDL fixtures, counts how many tables
share each candidate column name, and flags names that already appear in
real ``FOREIGN KEY`` / ``REFERENCES`` clauses on the Pagila side so we do
not suppress legitimate FK-like inferences.

Noise metric per column ``c``:
    pairs(c) = n * (n - 1)    where n = tables carrying ``c``

This is the upper bound on ``heuristic_same_name`` candidates produced
before the type-family veto and per-pair cap — the inferred-edge noise
count used to justify deny-list additions.

Run:
    uv run python scripts/audit_deny_list.py
"""

from __future__ import annotations

import re
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests/e2e/fixtures/sql"
AW = FIXTURES / "adventureworks_ddl.sql"
PAGILA = FIXTURES / "pagila_ddl.sql"

# Audit-tier candidates plus rating/length (included for regression
# sanity-check — already shipped in an earlier commit).
CANDIDATES = [
    "rating",
    "length",
    "count",
    "total",
    "quantity",
    "qty",
    "amount",
    "price",
    "weight",
    "height",
    "width",
    "score",
    "rank",
    "position",
    "order",
    "level",
]

_CREATE_TABLE = re.compile(
    r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?([\w.]+)\s*\((.*?)\n\s*\)\s*;",
    re.DOTALL | re.IGNORECASE,
)
# column-line: leading identifier, possibly quoted, optionally followed by type
_COLUMN_LINE = re.compile(r"^\s*([A-Za-z_][\w]*)\s+\S", re.MULTILINE)
_FK_INLINE = re.compile(r"REFERENCES\s+[\w.]+\s*\(([\w, ]+)\)", re.IGNORECASE)


def _parse_tables(sql: str) -> dict[str, set[str]]:
    """Return {qualified_table_name: set(column_names_lowercased)}."""
    out: dict[str, set[str]] = {}
    for match in _CREATE_TABLE.finditer(sql):
        name = match.group(1).lower()
        body = match.group(2)
        cols: set[str] = set()
        for line in body.splitlines():
            stripped = line.strip().rstrip(",")
            if not stripped or stripped.upper().startswith(
                ("PRIMARY KEY", "FOREIGN KEY", "CONSTRAINT", "UNIQUE", "CHECK")
            ):
                continue
            m = _COLUMN_LINE.match(line)
            if m:
                cols.add(m.group(1).lower())
        if cols:
            out[name] = cols
    return out


def _fk_columns(sql: str) -> set[str]:
    """Column names that appear as FK sources (inline ``col ... REFERENCES`` or
    table-level ``FOREIGN KEY (col...)``)."""
    fk_cols: set[str] = set()

    # table-level: FOREIGN KEY (a, b) REFERENCES ...
    for m in re.finditer(
        r"FOREIGN KEY\s*\(([\w, ]+)\)\s*REFERENCES", sql, re.IGNORECASE
    ):
        for c in m.group(1).split(","):
            fk_cols.add(c.strip().lower())

    # inline: "<col>  INT  ...  REFERENCES target(pk)" — grab the column name
    # at the start of each column line that also contains REFERENCES.
    for line in sql.splitlines():
        if "REFERENCES" in line.upper() and not line.strip().upper().startswith(
            "FOREIGN KEY"
        ):
            m = _COLUMN_LINE.match(line)
            if m:
                fk_cols.add(m.group(1).lower())

    return fk_cols


def _count(tables: dict[str, set[str]], name: str) -> tuple[int, list[str]]:
    hosts = [t for t, cols in tables.items() if name in cols]
    return len(hosts), hosts


def main() -> None:
    aw_tables = _parse_tables(AW.read_text())
    pagila_tables = _parse_tables(PAGILA.read_text())
    pagila_fk = _fk_columns(PAGILA.read_text())
    aw_fk = _fk_columns(AW.read_text())

    print("# Inferred-edge deny-list audit")
    print(f"# AdventureWorks tables parsed: {len(aw_tables)}")
    print(f"# Pagila tables parsed: {len(pagila_tables)}")
    print()
    print(
        "| column | aw_n | aw_pairs | pagila_n | legit_fk? | recommend |"
    )
    print("|---|---:|---:|---:|---|---|")

    for col in CANDIDATES:
        aw_n, aw_hosts = _count(aw_tables, col)
        pg_n, pg_hosts = _count(pagila_tables, col)
        aw_pairs = aw_n * (aw_n - 1)
        is_fk = col in pagila_fk or col in aw_fk
        # Rule: include iff noise >= 2 and not a real FK column in either fixture.
        include = aw_pairs >= 2 and not is_fk
        decision = "ADD" if include else ("SKIP (FK)" if is_fk else "skip (no noise)")
        print(
            f"| `{col}` | {aw_n} | {aw_pairs} | {pg_n} | "
            f"{'yes' if is_fk else 'no'} | {decision} |"
        )

    # Detail block for the names that get added: show which tables contributed.
    print()
    print("## Detail (hosts per candidate)")
    for col in CANDIDATES:
        _, aw_hosts = _count(aw_tables, col)
        _, pg_hosts = _count(pagila_tables, col)
        if aw_hosts or pg_hosts:
            print(f"- `{col}`: AW={sorted(aw_hosts)} | Pagila={sorted(pg_hosts)}")


if __name__ == "__main__":
    main()
