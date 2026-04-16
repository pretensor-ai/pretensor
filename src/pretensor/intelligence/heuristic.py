"""Deterministic relationship candidates from column and table naming patterns."""

from __future__ import annotations

from collections import defaultdict

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.ids import table_node_id
from pretensor.graph_models.relationship import RelationshipCandidate
from pretensor.intelligence.scoring import JoinKey, RelationshipScorer

__all__ = ["HeuristicScorer", "discover_heuristic_candidates"]

_CONFIDENCE_HIGH = 0.85
_CONFIDENCE_MEDIUM = 0.55
_CONFIDENCE_LOW = 0.25

_SUFFIX_ID = "_id"
_SUFFIX_FK = "_fk"

_GIN_BRIN_INDEX = frozenset({"gin", "brin"})

# ---------------------------------------------------------------------------
# Type-family mapping for cross-dialect compatibility checks.
# Unknown types (not in this map) receive no veto — avoids false negatives
# for exotic or custom types.
# ---------------------------------------------------------------------------
_TYPE_FAMILY: dict[str, str] = {
    # numeric
    "int": "numeric",
    "integer": "numeric",
    "bigint": "numeric",
    "smallint": "numeric",
    "tinyint": "numeric",
    "serial": "numeric",
    "bigserial": "numeric",
    "number": "numeric",
    "numeric": "numeric",
    "decimal": "numeric",
    "float": "numeric",
    "double": "numeric",
    "real": "numeric",
    "money": "numeric",
    # string
    "varchar": "string",
    "text": "string",
    "char": "string",
    "character": "string",
    "string": "string",
    "nvarchar": "string",
    "nchar": "string",
    "ntext": "string",
    "clob": "string",
    "bpchar": "string",
    "character varying": "string",
    # temporal
    "date": "temporal",
    "timestamp": "temporal",
    "datetime": "temporal",
    "timestamptz": "temporal",
    "timestamp without time zone": "temporal",
    "timestamp with time zone": "temporal",
    "time": "temporal",
    "interval": "temporal",
    # boolean
    "bool": "boolean",
    "boolean": "boolean",
    "bit": "boolean",
    # uuid
    "uuid": "uuid",
    "uniqueidentifier": "uuid",
    # binary
    "bytea": "binary",
    "blob": "binary",
    "binary": "binary",
    "varbinary": "binary",
    "image": "binary",
}


def _type_family(data_type: str) -> str | None:
    """Normalize a dialect-specific type name to a family, or ``None`` if unknown."""
    base = data_type.strip().lower().split("(")[0].strip()
    return _TYPE_FAMILY.get(base)


def _types_compatible(src: Column | None, dst: Column | None) -> bool:
    """Reject cross-family inferred joins (e.g. numeric id vs. varchar email).

    Unknown types pass through — vetoing on missing info would drop valid
    candidates for exotic dialects. Applied to every heuristic kind, not just
    same_name, so ``productid (int) → emailaddress (varchar)``-style fan-outs
    never reach the graph.
    """
    if src is None or dst is None:
        return True
    sf = _type_family(src.data_type)
    df = _type_family(dst.data_type)
    if sf is None or df is None:
        return True
    return sf == df
_NULL_PCT_HIGH = 80.0
_LOW_CARD_MAX = 50
_LARGE_TABLE_ROWS = 10_000

# Column names that are audit/metadata fields rather than join keys. Filtering
# by *name* (not by how many tables a column appears on) keeps legitimate
# wide-shared columns — e.g. a ``region_code`` denormalized across many tables
# — emitting candidates while suppressing the AdventureWorks pathology where
# ``ModifiedDate`` / ``rowguid`` live on ~60 tables. Comparison is
# case-insensitive (see lookup site below). The downstream join-path DFS has
# its own perf safety net via ``_DFS_VISIT_BUDGET`` in
# ``intelligence/join_paths/on_demand.py``.
_BORING_SHARED_NAMES = frozenset(
    {
        # Audit / metadata (alphabetical)
        "created_at",
        "deleted_at",
        "id",
        "last_update",
        "lastupdate",
        "modified_at",
        "modifieddate",
        "rowguid",
        "updated_at",
        "version",
        # Generic descriptors — near-zero selectivity as join keys (alphabetical)
        "category",
        "class",
        "code",
        "color",
        "comments",
        "description",
        "due_date",
        "duedate",
        "end_date",
        "enddate",
        "flag",
        "group",
        "label",
        "name",
        "notes",
        "size",
        "start_date",
        "startdate",
        "status",
        "style",
        "title",
        "type",
        # Dimensional attributes — high cardinality but no cross-table join meaning.
        # Dropping these (rather than down-weighting) avoids polluting
        # ``context`` output with e.g. film.rating = category.rating.
        "active",
        "length",
        "rating",
        # Generic numerics — shared column names with no cross-table join
        # semantics. The type-family veto (numeric ↔ numeric) cannot distinguish
        # e.g. ``productreview.rating`` from ``product.rating`` as unrelated
        # concepts, and the per-pair cap (3) only caps, does not eliminate, the
        # noise.
        "amount",
        "count",
        "height",
        "level",
        "order",
        "position",
        "price",
        "qty",
        "quantity",
        "rank",
        "score",
        "total",
        "weight",
        "width",
    }
)


def _table_key(t: Table) -> tuple[str, str]:
    return (t.schema_name, t.name)


def _column_lookup(
    snapshot: SchemaSnapshot,
) -> dict[tuple[str, str], dict[str, Column]]:
    """Map (schema, table_name) -> column_name -> Column."""
    out: dict[tuple[str, str], dict[str, Column]] = {}
    for t in snapshot.tables:
        out[_table_key(t)] = {c.name: c for c in t.columns}
    return out


def _plural_forms(stem: str) -> list[str]:
    """Cheap English plural guesses for table-name matching."""
    s = stem.lower()
    if not s:
        return []
    forms = {s}
    forms.add(s + "s")
    if s.endswith("y") and len(s) > 1:
        forms.add(s[:-1] + "ies")
    if s.endswith("ch") or s.endswith("sh") or s.endswith("s") or s.endswith("x"):
        forms.add(s + "es")
    return list(forms)


def _find_referenced_table(
    snapshot: SchemaSnapshot,
    stem: str,
    exclude: tuple[str, str],
) -> Table | None:
    """Pick a single target table whose name matches ``stem`` plural guesses."""
    forms = set(_plural_forms(stem))
    matches = [
        t
        for t in snapshot.tables
        if _table_key(t) != exclude and t.name.lower() in forms
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer exact stem match, then shortest name (e.g. customer over customers_backup).
        exact = [t for t in matches if t.name.lower() == stem.lower()]
        if len(exact) == 1:
            return exact[0]
        return min(matches, key=lambda t: len(t.name))
    return None


def _prefix_match_tables(
    snapshot: SchemaSnapshot,
    prefix: str,
    exclude: tuple[str, str],
) -> Table | None:
    """Tables whose name starts with ``prefix`` (abbreviation heuristic)."""
    p = prefix.lower()
    if len(p) < 2:
        return None
    matches = [
        t
        for t in snapshot.tables
        if _table_key(t) != exclude and t.name.lower().startswith(p)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return min(matches, key=lambda t: len(t.name))
    return None


def _target_has_id_column(
    columns_by_table: dict[tuple[str, str], dict[str, Column]], t: Table
) -> bool:
    cols = columns_by_table.get(_table_key(t), {})
    if "id" in cols:
        return True
    for name, col in cols.items():
        if col.is_primary_key:
            return True
    return False


def _pick_target_column(
    columns_by_table: dict[tuple[str, str], dict[str, Column]], t: Table
) -> str | None:
    cols = columns_by_table.get(_table_key(t), {})
    if "id" in cols:
        return "id"
    for name, col in cols.items():
        if col.is_primary_key:
            return name
    return None


def _candidate_id(
    connection_name: str,
    src_schema: str,
    src_table: str,
    src_col: str,
    dst_schema: str,
    dst_table: str,
    dst_col: str,
    kind: str,
) -> str:
    return (
        f"{kind}:{connection_name}:{src_schema}.{src_table}.{src_col}"
        f"->{dst_schema}.{dst_table}.{dst_col}"
    )


def _column_at(
    columns_by_table: dict[tuple[str, str], dict[str, Column]],
    table: Table,
    col_name: str,
) -> Column | None:
    return columns_by_table.get(_table_key(table), {}).get(col_name)


def _is_low_cardinality_large_table(src_table: Table, src_col: Column | None) -> bool:
    """True when catalog says few distinct values on a large source table."""
    if src_table.row_count is None or src_table.row_count <= _LARGE_TABLE_ROWS:
        return False
    if src_col is None or src_col.column_cardinality is None:
        return False
    c = src_col.column_cardinality
    return c >= 0 and c < _LOW_CARD_MAX


def _apply_catalog_signals(
    *,
    src_table: Table,
    dst_table: Table,
    src_col_name: str,
    dst_col_name: str,
    kind: str,
    base_confidence: float,
    columns_by_table: dict[tuple[str, str], dict[str, Column]],
) -> float | None:
    """Adjust confidence using SchemaColumn catalog fields.

    Order: vetoes (drop candidate), then penalties and boosts, then clamp to [0, 1].
    """
    src_c = _column_at(columns_by_table, src_table, src_col_name)
    dst_c = _column_at(columns_by_table, dst_table, dst_col_name)

    src_index = (src_c.index_type or "").lower() if src_c else ""
    if src_index in _GIN_BRIN_INDEX and kind == "heuristic_same_name":
        return None
    if (
        _is_low_cardinality_large_table(src_table, src_c)
        and kind == "heuristic_same_name"
    ):
        return None
    if not _types_compatible(src_c, dst_c):
        return None

    conf = base_confidence
    if src_index in _GIN_BRIN_INDEX and kind in ("heuristic_fk", "heuristic_id"):
        conf -= 0.30
    if kind == "heuristic_id" and _is_low_cardinality_large_table(src_table, src_c):
        conf -= 0.20
    if dst_c is not None and dst_c.index_is_unique is True:
        conf += 0.15
    for col in (src_c, dst_c):
        if (
            col is not None
            and col.null_percentage is not None
            and col.null_percentage > _NULL_PCT_HIGH
        ):
            conf -= 0.15

    return max(0.0, min(1.0, conf))


def discover_heuristic_candidates(
    snapshot: SchemaSnapshot,
) -> list[RelationshipCandidate]:
    """Return heuristic join hypotheses for tables in ``snapshot``.

    Rules: ``*_id`` / ``*_fk`` → referenced table ``.id`` (high),
    abbreviated prefix (medium), same column name across tables (low).

    Catalog signals adjust confidence when ``Column`` / ``Table`` fields
    from introspection are present (cardinality, index type, uniqueness, null rate,
    row counts).
    """
    conn = snapshot.connection_name
    columns_by_table = _column_lookup(snapshot)
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    out: list[RelationshipCandidate] = []

    def add(
        src_table: Table,
        src_col: str,
        dst_table: Table,
        dst_col: str,
        confidence: float,
        reasoning: str,
        kind: str,
    ) -> None:
        adjusted = _apply_catalog_signals(
            src_table=src_table,
            dst_table=dst_table,
            src_col_name=src_col,
            dst_col_name=dst_col,
            kind=kind,
            base_confidence=confidence,
            columns_by_table=columns_by_table,
        )
        if adjusted is None:
            return
        confidence = adjusted
        src_id = table_node_id(conn, src_table.schema_name, src_table.name)
        dst_id = table_node_id(conn, dst_table.schema_name, dst_table.name)
        key = (src_id, dst_id, src_col, dst_col, kind)
        if key in seen_keys:
            return
        seen_keys.add(key)
        out.append(
            RelationshipCandidate(
                candidate_id=_candidate_id(
                    conn,
                    src_table.schema_name,
                    src_table.name,
                    src_col,
                    dst_table.schema_name,
                    dst_table.name,
                    dst_col,
                    kind,
                ),
                source_node_id=src_id,
                target_node_id=dst_id,
                source_column=src_col,
                target_column=dst_col,
                source="heuristic",
                reasoning=reasoning,
                confidence=confidence,
                status="suggested",
            )
        )

    for table in snapshot.tables:
        exclude_key = _table_key(table)
        cols = columns_by_table.get(exclude_key, {})
        for col_name, col in cols.items():
            lower = col_name.lower()

            if lower.endswith(_SUFFIX_FK):
                stem = col_name[: -len(_SUFFIX_FK)]
                ref = _find_referenced_table(snapshot, stem, exclude_key)
                if ref is None:
                    ref = _prefix_match_tables(snapshot, stem, exclude_key)
                    conf = _CONFIDENCE_MEDIUM
                    reason = f"column '{col_name}' ends with _fk; matched table prefix '{stem}'"
                else:
                    conf = _CONFIDENCE_HIGH
                    reason = f"column '{col_name}' ends with _fk → {ref.name}.id"
                if ref and _target_has_id_column(columns_by_table, ref):
                    tc = _pick_target_column(columns_by_table, ref)
                    if tc:
                        add(table, col_name, ref, tc, conf, reason, "heuristic_fk")

            elif lower.endswith(_SUFFIX_ID) and lower != "id":
                stem = col_name[: -len(_SUFFIX_ID)]
                if not stem:
                    continue
                ref = _find_referenced_table(snapshot, stem, exclude_key)
                if ref is None:
                    ref = _prefix_match_tables(snapshot, stem, exclude_key)
                    conf = _CONFIDENCE_MEDIUM
                    reason = (
                        f"column '{col_name}' ends with _id; matched table name "
                        f"starting with '{stem}'"
                    )
                else:
                    conf = _CONFIDENCE_HIGH
                    reason = f"column '{col_name}' → {ref.name}.id (naming convention)"
                if ref and _target_has_id_column(columns_by_table, ref):
                    tc = _pick_target_column(columns_by_table, ref)
                    if tc:
                        add(table, col_name, ref, tc, conf, reason, "heuristic_id")

    # Same column name across distinct tables (low confidence).
    # Collect candidates first, then cap per table pair.
    _SAME_NAME_CAP_PER_PAIR = 3

    by_name: dict[str, list[Table]] = defaultdict(list)
    for t in snapshot.tables:
        for c in t.columns:
            if c.name.lower() in _BORING_SHARED_NAMES:
                continue
            by_name[c.name].append(t)

    # Each entry: (src, col, dst, col, selectivity, reason)
    raw_same: list[tuple[Table, str, Table, str, float, str]] = []

    for col_name, tables in by_name.items():
        if len(tables) < 2:
            continue
        uniq_tables: list[Table] = []
        seen_tbl: set[tuple[str, str]] = set()
        for t in tables:
            k = _table_key(t)
            if k not in seen_tbl:
                seen_tbl.add(k)
                uniq_tables.append(t)
        if len(uniq_tables) < 2:
            continue
        selectivity = 1.0 / len(uniq_tables)
        for i, t1 in enumerate(uniq_tables):
            for t2 in uniq_tables[i + 1 :]:
                reason = (
                    f"shared column name '{col_name}' on {t1.name} and {t2.name} "
                    "(weak join signal)"
                )
                raw_same.append((t1, col_name, t2, col_name, selectivity, reason))
                raw_same.append((t2, col_name, t1, col_name, selectivity, reason))

    # Group by directed (src, dst) pair and keep top-N by selectivity.
    pair_groups: dict[
        tuple[tuple[str, str], tuple[str, str]],
        list[tuple[Table, str, Table, str, float, str]],
    ] = defaultdict(list)
    for entry in raw_same:
        pk = (_table_key(entry[0]), _table_key(entry[2]))
        pair_groups[pk].append(entry)

    for entries in pair_groups.values():
        entries.sort(key=lambda e: (-e[4], e[1]))
        for src_t, src_col, dst_t, dst_col, _sel, reason in entries[
            :_SAME_NAME_CAP_PER_PAIR
        ]:
            add(src_t, src_col, dst_t, dst_col, _CONFIDENCE_LOW, reason, "heuristic_same_name")

    return out


class HeuristicScorer(RelationshipScorer):
    """Default OSS scorer that runs deterministic naming heuristics."""

    def name(self) -> str:
        return "heuristic"

    def score(
        self,
        snapshot: SchemaSnapshot,
        explicit_fk_keys: set[JoinKey],
    ) -> list[RelationshipCandidate]:
        """Return heuristic candidates excluding explicit FK edges."""
        raw = discover_heuristic_candidates(snapshot)
        return [c for c in raw if _candidate_join_key(c) not in explicit_fk_keys]


def _candidate_join_key(candidate: RelationshipCandidate) -> JoinKey:
    return (
        candidate.source_node_id,
        candidate.target_node_id,
        candidate.source_column,
        candidate.target_column,
    )
