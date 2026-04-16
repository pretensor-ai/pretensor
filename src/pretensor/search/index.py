"""SQLite FTS5 keyword search index over SchemaTable and Entity nodes."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from pretensor.core.registry import GraphRegistry, RegistryEntry
from pretensor.core.store import KuzuStore
from pretensor.search.base import BaseSearchIndex, SearchResult

__all__ = ["KeywordSearchIndex", "SearchIndex", "SearchResult"]

_INDEX_SCHEMA_VERSION = "2"
_IDENTIFIER_TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


def _graph_path_for_entry(entry: RegistryEntry) -> Path:
    return Path(entry.unified_graph_path or entry.graph_path)


def _camel_to_space(value: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)


def _dedupe_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _identifier_tokens(value: str) -> list[str]:
    pieces = [value, _camel_to_space(value)]
    out: list[str] = []
    for piece in pieces:
        for token in _IDENTIFIER_TOKEN_SPLIT_RE.split(piece.lower()):
            if token:
                out.append(token)
    return _dedupe_non_empty(out)


def _identifier_terms(schema_name: str, table_name: str) -> str:
    fq_name = f"{schema_name}.{table_name}"
    ordered = _dedupe_non_empty(
        [
            fq_name,
            schema_name,
            table_name,
            *_identifier_tokens(schema_name),
            *_identifier_tokens(table_name),
        ]
    )
    return " ".join(ordered)


def _trim_for_tie_breaker(raw: str, *, max_tokens: int = 16) -> str:
    tokens = raw.split()
    if not tokens:
        return ""
    return " ".join(tokens[:max_tokens])


def _sanitize_fts_query(raw: str) -> str:
    """Build a safe FTS5 MATCH expression from free text (token OR token)."""
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in raw)
    tokens = [t for t in cleaned.lower().split() if t]
    if not tokens:
        return '""'
    return " OR ".join(tokens)


def _open_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


class KeywordSearchIndex(BaseSearchIndex):
    """BM25-ranked keyword search backed by SQLite FTS5."""

    def __init__(self, index_path: Path) -> None:
        self._path = index_path

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def default_path(graph_dir: Path) -> Path:
        """Default FTS index location under the graph state directory."""
        return graph_dir / "search_metadata.fts.sqlite"

    @classmethod
    def needs_rebuild(cls, registry: GraphRegistry, index_path: Path) -> bool:
        """True when the index is missing or older than any registered Kuzu file."""
        if not index_path.exists():
            return True
        if not cls._has_current_schema(index_path):
            return True
        entries = registry.list_entries()
        if not entries:
            return False
        index_mtime = index_path.stat().st_mtime
        for entry in entries:
            gp = _graph_path_for_entry(entry)
            if gp.exists() and gp.stat().st_mtime > index_mtime:
                return True
        return False

    @classmethod
    def _has_current_schema(cls, index_path: Path) -> bool:
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(str(index_path))
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?",
                ("schema_version",),
            ).fetchone()
            return bool(row) and str(row[0]) == _INDEX_SCHEMA_VERSION
        except sqlite3.Error:
            return False
        finally:
            if conn is not None:
                conn.close()

    @classmethod
    def build(cls, registry: GraphRegistry, index_path: Path) -> KeywordSearchIndex:
        """Rebuild the FTS index from all graphs in the registry."""
        index = cls(index_path)
        conn = _open_sqlite(index_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
                DROP TABLE IF EXISTS graph_fts;
                CREATE VIRTUAL TABLE graph_fts USING fts5(
                    node_type UNINDEXED,
                    name,
                    description,
                    table_text,
                    cluster_context,
                    database_name UNINDEXED,
                    connection_name UNINDEXED,
                    tokenize = 'porter unicode61'
                );
                """
            )
            for entry in registry.list_entries():
                gp = _graph_path_for_entry(entry)
                if not gp.exists():
                    continue
                store = KuzuStore(gp)
                try:
                    index._ingest_connection(store, entry, conn)
                finally:
                    store.close()
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", _INDEX_SCHEMA_VERSION),
            )
            conn.commit()
        finally:
            conn.close()
        return index

    @classmethod
    def load_or_build(
        cls, registry: GraphRegistry, index_path: Path
    ) -> KeywordSearchIndex:
        """Return a search index, rebuilding automatically when graphs are newer."""
        if cls.needs_rebuild(registry, index_path):
            return cls.build(registry, index_path)
        return cls(index_path)

    def _ingest_connection(
        self,
        store: KuzuStore,
        entry: RegistryEntry,
        conn: sqlite3.Connection,
    ) -> None:
        table_signals_rows = store.query_all_rows(
            """
            MATCH (t:SchemaTable {connection_name: $cn})-[:HAS_COLUMN]->(c:SchemaColumn)
            RETURN t.schema_name, t.table_name, c.column_name, c.most_common_values_json
            """,
            {"cn": entry.connection_name},
        )
        column_names_by_table: dict[tuple[str, str], list[str]] = {}
        mcv_by_table: dict[tuple[str, str], list[str]] = {}
        for sn, tn, column_name, mcv_json in table_signals_rows:
            key = (str(sn), str(tn))
            column_text = str(column_name).strip()
            if column_text:
                column_names_by_table.setdefault(key, []).append(column_text)
            if not mcv_json:
                continue
            try:
                parsed = json.loads(str(mcv_json))
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, list):
                continue
            for v in parsed:
                s = str(v).strip()
                if s:
                    mcv_by_table.setdefault(key, []).append(s)

        entity_text_by_table: dict[tuple[str, str], list[str]] = {}
        entity_rows = store.query_all_rows(
            """
            MATCH (e:Entity {connection_name: $cn})-[:REPRESENTS]->(t:SchemaTable {connection_name: $cn})
            RETURN t.schema_name, t.table_name, e.name, e.description
            """,
            {"cn": entry.connection_name},
        )
        for sn, tn, entity_name, entity_desc in entity_rows:
            key = (str(sn), str(tn))
            if entity_name:
                entity_text_by_table.setdefault(key, []).append(str(entity_name))
            if entity_desc and str(entity_desc).strip():
                entity_text_by_table.setdefault(key, []).append(str(entity_desc).strip())

        cluster_text_by_table: dict[tuple[str, str], list[str]] = {}
        cluster_rows = store.query_all_rows(
            """
            MATCH (t:SchemaTable {connection_name: $cn})-[:IN_CLUSTER]->(c:Cluster)
            RETURN t.schema_name, t.table_name, c.label, c.description
            """,
            {"cn": entry.connection_name},
        )
        for sn, tn, cluster_label, cluster_desc in cluster_rows:
            key = (str(sn), str(tn))
            if cluster_label:
                cluster_text_by_table.setdefault(key, []).append(str(cluster_label))
            if cluster_desc and str(cluster_desc).strip():
                trimmed = _trim_for_tie_breaker(str(cluster_desc).strip())
                if trimmed:
                    cluster_text_by_table.setdefault(key, []).append(trimmed)

        rows = store.query_all_rows(
            """
            MATCH (t:SchemaTable {connection_name: $cn})
            RETURN t.schema_name, t.table_name, t.comment, t.description,
                   t.entity_type, t.database,
                   COALESCE(t.tags, CAST([] AS STRING[])) AS tags
            """,
            {"cn": entry.connection_name},
        )
        for (
            schema_name,
            table_name,
            comment,
            desc_col,
            entity_type,
            database,
            tags_arr,
        ) in rows:
            key = (str(schema_name), str(table_name))
            parts: list[str] = []
            if comment:
                parts.append(str(comment))
            if desc_col and str(desc_col).strip():
                parts.append(f"description: {str(desc_col).strip()}")
            if isinstance(tags_arr, (list, tuple)) and tags_arr:
                parts.append(
                    "tags: " + " ".join(str(x) for x in tags_arr if str(x).strip())
                )
            if entity_type:
                parts.append(f"entity_type: {entity_type}")
            mcv_extra = _dedupe_non_empty(mcv_by_table.get(key, []))
            if mcv_extra:
                parts.append("common_values: " + " ".join(mcv_extra))
            desc = " ".join(parts)

            table_text_parts: list[str] = []
            table_text_parts.append(
                "identifier: "
                + _identifier_terms(str(schema_name), str(table_name))
            )
            column_terms = _dedupe_non_empty(column_names_by_table.get(key, []))
            if column_terms:
                column_tokens: list[str] = []
                for col in column_terms:
                    column_tokens.extend(_identifier_tokens(col))
                table_text_parts.append(
                    "columns: "
                    + " ".join(_dedupe_non_empty([*column_terms, *column_tokens]))
                )
            entity_terms = _dedupe_non_empty(entity_text_by_table.get(key, []))
            if entity_terms:
                table_text_parts.append("entity: " + " ".join(entity_terms))
            table_text = " ".join(table_text_parts)

            cluster_terms = _dedupe_non_empty(cluster_text_by_table.get(key, []))
            cluster_context = ""
            if cluster_terms:
                cluster_context = "cluster: " + " ".join(cluster_terms)

            fq = f"{schema_name}.{table_name}"
            conn.execute(
                """
                INSERT INTO graph_fts(
                    node_type, name, description, table_text, cluster_context,
                    database_name, connection_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "SchemaTable",
                    fq,
                    desc,
                    table_text,
                    cluster_context,
                    str(database or entry.database),
                    entry.connection_name,
                ),
            )

        erows = store.query_all_rows(
            """
            MATCH (e:Entity {connection_name: $cn})
            RETURN e.name, e.description, e.database
            """,
            {"cn": entry.connection_name},
        )
        for name, description, database in erows:
            conn.execute(
                """
                INSERT INTO graph_fts(
                    node_type, name, description, table_text, cluster_context,
                    database_name, connection_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "Entity",
                    str(name),
                    str(description or ""),
                    "",
                    "",
                    str(database or entry.database),
                    entry.connection_name,
                ),
            )

    def search(
        self,
        q: str,
        *,
        db: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Run BM25 search; optional ``db`` filters by connection or logical database name."""
        if not self._path.exists():
            return []
        conn = sqlite3.connect(str(self._path))
        try:
            match_expr = _sanitize_fts_query(q)
            sql = """
                SELECT node_type, name, description, database_name, connection_name,
                       bm25(graph_fts, 0.0, 1.0, 1.0, 1.4, 0.2, 0.0, 0.0) AS score
                FROM graph_fts
                WHERE graph_fts MATCH ?
            """
            params: list[object] = [match_expr]
            if db is not None:
                sql += " AND (connection_name = ? OR database_name = ?)"
                params.extend([db, db])
            # bm25() is lower for better matches in SQLite FTS5
            sql += " ORDER BY score ASC LIMIT ?"
            params.append(limit)
            cur = conn.execute(sql, params)
            out: list[SearchResult] = []
            for row in cur.fetchall():
                out.append(
                    SearchResult(
                        node_type=str(row[0]),
                        name=str(row[1]),
                        database_name=str(row[3]),
                        connection_name=str(row[4]),
                        description=str(row[2] or ""),
                        score=float(row[5]),
                    )
                )
            return out
        finally:
            conn.close()

    def similar(
        self,
        name: str,
        *,
        db: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return nodes related to *name* via keyword proximity in the FTS index.

        Uses the node's own name as the search query so that nodes sharing
        significant tokens (schema prefix, entity type, common words) rank
        highly.  Results matching *name* exactly are excluded.

        Args:
            name: Fully-qualified node name, e.g. ``schema.table``.
            db: Optional connection or logical database name filter.
            limit: Maximum number of results to return.

        Returns:
            Related nodes ranked by BM25 proximity.
        """
        candidates = self.search(name, db=db, limit=limit + 1)
        return [r for r in candidates if r.name != name][:limit]

    def index_graph(self, registry: GraphRegistry) -> None:
        """Rebuild the FTS index from all graphs in *registry*.

        Args:
            registry: Loaded registry describing available graph files.
        """
        self.build(registry, self._path)


# Backward-compatible alias — existing imports of ``SearchIndex`` keep working.
SearchIndex = KeywordSearchIndex
