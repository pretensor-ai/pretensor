"""MCP ``cypher`` tool: read-only Kuzu Cypher with timeout and JSON-safe rows."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from pretensor.core.store import KuzuStore
from pretensor.visibility.filter import VisibilityFilter

from ..service_context import get_effective_visibility_filter
from ..service_registry import (
    graph_path_for_entry,
    load_registry,
    resolve_registry_entry,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CYPHER_TIMEOUT_SECONDS",
    "assert_read_only_cypher",
    "cypher_payload",
    "json_safe_cypher_value",
]

DEFAULT_CYPHER_TIMEOUT_SECONDS = 5.0
_MAX_TIMEOUT_SECONDS = 120.0

# Mask quoted literals before scanning so clauses inside strings do not false-positive.
_STRING_LITERAL_RE = re.compile(
    r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"",
)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_FORBIDDEN_CLAUSE_RE = re.compile(
    r"\b(CREATE|DELETE|MERGE|SET)\b",
    flags=re.IGNORECASE,
)
_PROPERTY_NOT_FOUND_RE = re.compile(
    r"Cannot find property (\w+) for (\w+)", re.IGNORECASE
)
_NODE_ALIAS_RE = re.compile(r"\((\w+):(\w+)")  # (alias:Label)
_RESERVED_TOKEN_RE = re.compile(
    r"mismatched input '(\w+)'.*?expecting", re.IGNORECASE | re.DOTALL
)
# Common Kuzu reserved words that users often try to use as identifiers.
# Not exhaustive; matched case-insensitively.
_KUZU_RESERVED = frozenset(
    {
        "asc",
        "desc",
        "order",
        "group",
        "by",
        "limit",
        "skip",
        "match",
        "return",
        "where",
        "with",
        "union",
        "call",
        "as",
        "in",
        "and",
        "or",
        "not",
        "null",
        "true",
        "false",
        "optional",
    }
)


def json_safe_cypher_value(value: Any) -> Any:
    """Recursively coerce a Kuzu cell value to JSON-serializable form."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe_cypher_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_cypher_value(v) for v in value]
    return str(value)


def _materialize_read_only_rows(graph_path: Path, q: str) -> list[dict[str, Any]]:
    """Open DB, run Cypher, return row dicts (runs in a worker thread for timeouts)."""
    store = KuzuStore(graph_path)
    try:
        store.ensure_schema()
        raw = store.execute(q)
        if isinstance(raw, list):
            raise TypeError("Unexpected multi-statement Cypher result")
        dict_result = raw.rows_as_dict()
        rows: list[dict[str, Any]] = []
        while dict_result.has_next():
            row_any: Any = dict_result.get_next()
            if not isinstance(row_any, dict):
                raise TypeError("Unexpected row shape from graph engine")
            row: dict[str, Any] = row_any
            rows.append({str(k): json_safe_cypher_value(v) for k, v in row.items()})
        return rows
    finally:
        store.close()


def assert_read_only_cypher(query: str) -> None:
    """Raise ValueError if ``query`` may perform graph mutations.

    Rejects CREATE, DELETE, MERGE, and SET as top-level clauses (after comments
    and string literals are stripped). Results are post-filtered for hidden tables.
    """
    stripped = _strip_comments_and_mask_strings(query)
    if _FORBIDDEN_CLAUSE_RE.search(stripped):
        raise ValueError(
            "Only read queries are allowed; CREATE, DELETE, MERGE, and SET are not permitted."
        )


def _strip_comments_and_mask_strings(query: str) -> str:
    text = _BLOCK_COMMENT_RE.sub(" ", query)
    parts: list[str] = []
    for line in text.splitlines():
        if "--" in line:
            line = line.split("--", 1)[0]
        if "//" in line:
            line = line.split("//", 1)[0]
        parts.append(line)
    joined = "\n".join(parts)
    return _STRING_LITERAL_RE.sub("''", joined)


def _redact_cypher_value(
    value: Any,
    vf: VisibilityFilter,
    connection_name: str,
) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        parts = value.split("::")
        # SchemaTable node IDs use the format "connection::schema::table".
        # Entity nodes share the same 3-part format but use the literal schema
        # name "entity" by convention; metric node IDs contain "metric" in the
        # table part.  Skipping these avoids false-positive redactions.
        if len(parts) == 3 and parts[1] not in ("entity",) and "metric" not in parts[2]:
            if not vf.is_schema_table_node_id_visible(value):
                return None
            return value
        if "." in value and "::" not in value:
            sn, _, tn = value.partition(".")
            if sn and tn and not vf.is_table_visible(connection_name, sn, tn):
                return None
        return value
    if isinstance(value, dict):
        return {
            str(k): _redact_cypher_value(v, vf, connection_name)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_cypher_value(v, vf, connection_name) for v in value]
    return str(value)


def filter_cypher_rows_for_visibility(
    rows: list[dict[str, Any]],
    *,
    visibility_filter: VisibilityFilter,
    connection_name: str,
) -> list[dict[str, Any]]:
    """Drop or redact cells that reference hidden ``SchemaTable`` nodes."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if any(
            k.lower() in ("node_id", "table_node_id", "from_table_id", "to_table_id")
            and isinstance(v, str)
            and not visibility_filter.is_schema_table_node_id_visible(v)
            for k, v in row.items()
        ):
            continue
        redacted = {
            k: _redact_cypher_value(v, visibility_filter, connection_name)
            for k, v in row.items()
        }
        out.append(redacted)
    return out


def _parse_node_aliases(query: str) -> dict[str, str]:
    """Return ``{alias: label}`` from MATCH patterns like ``(c:SchemaColumn)``."""
    return dict(_NODE_ALIAS_RE.findall(query))


def _get_node_properties(graph_path: Path, label: str) -> list[str]:
    """Return property names for a Kuzu node table, or empty list on error."""
    store = KuzuStore(graph_path)
    try:
        store.ensure_schema()
        result = store.execute(f"CALL table_info('{label}') RETURN name")
        if isinstance(result, list):
            return []
        rows = result.rows_as_dict()
        props: list[str] = []
        while rows.has_next():
            row: Any = rows.get_next()
            if isinstance(row, dict) and "name" in row:
                props.append(str(row["name"]))
        return props
    except Exception:
        return []
    finally:
        store.close()


def _enrich_kuzu_error(exc: Exception, query: str, graph_path: Path) -> str:
    """Wrap raw Kuzu parser/runtime errors with actionable hints when possible."""
    err_str = str(exc)
    reserved = _RESERVED_TOKEN_RE.search(err_str)
    if reserved and reserved.group(1).lower() in _KUZU_RESERVED:
        token = reserved.group(1)
        return (
            f"Cypher error: {token!r} is a reserved Kuzu keyword used where an "
            f"identifier was expected. Rename the alias (e.g. {token}_) or "
            f"backtick it (`{token}`)."
        )
    match = _PROPERTY_NOT_FOUND_RE.search(err_str)
    if not match:
        return f"Cypher error: {err_str}"
    prop_name, alias = match.group(1), match.group(2)
    aliases = _parse_node_aliases(query)
    label = aliases.get(alias)
    hint = f"Property {prop_name!r} does not exist on alias {alias!r}."
    if label:
        props = _get_node_properties(graph_path, label)
        if props:
            hint += f" Available properties on {label}: {', '.join(sorted(props))}"
    return f"Cypher error: {err_str}. Hint: {hint}"


def cypher_payload(
    graph_dir: Path,
    *,
    query: str,
    database: str | None = None,
    timeout_seconds: float = DEFAULT_CYPHER_TIMEOUT_SECONDS,
    visibility_filter: VisibilityFilter | None = None,
) -> dict[str, Any]:
    """Run read-only Cypher on the indexed graph for ``database``; return row dicts.

    Args:
        graph_dir: Graph workspace directory (registry + Kuzu files).
        query: Raw Cypher string.
        database: Connection name or logical database. Optional only when a
            single graph is indexed (deprecation shim); required otherwise.
        timeout_seconds: Wall-clock limit for executing the query (default 5s).

    Returns:
        ``{"rows": [...]}`` on success, or ``{"error": "..."}``.
    """
    q = query.strip()
    if not q:
        return {"error": "Missing or empty `query`"}
    db = (database or "").strip()
    if timeout_seconds <= 0:
        return {"error": "`timeout_seconds` must be positive"}
    if timeout_seconds > _MAX_TIMEOUT_SECONDS:
        return {
            "error": f"`timeout_seconds` must be at most {_MAX_TIMEOUT_SECONDS:.0f}s"
        }

    try:
        assert_read_only_cypher(q)
    except ValueError as exc:
        return {"error": str(exc)}

    reg = load_registry(graph_dir)
    auto_picked_warning: str | None = None
    if not db:
        entries = reg.list_entries()
        if len(entries) == 1:
            entry = entries[0]
            db = str(entry.connection_name)
            auto_picked_warning = (
                f"`database` was omitted; auto-selected the sole indexed graph "
                f"{db!r}. This shim will be removed — pass `database` explicitly."
            )
            logger.warning(
                "cypher: auto-selected sole indexed graph %r (deprecation shim)",
                db,
            )
        else:
            names = ", ".join(str(e.connection_name) for e in entries)
            return {
                "error": (
                    "Missing `database`: multiple graphs are indexed, pass one of "
                    f"[{names}]." if entries
                    else "Missing `database`: no graphs are indexed."
                )
            }
    else:
        resolved = resolve_registry_entry(reg, db)
        if resolved is None:
            return {"error": f"Unknown database: {db!r}"}
        entry = resolved

    graph_path = graph_path_for_entry(entry)
    if not graph_path.exists():
        return {"error": f"Graph file not found for database: {db!r}"}

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_materialize_read_only_rows, graph_path, q)
            rows = future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        return {"error": f"Query timed out after {timeout_seconds:g}s"}
    except Exception as exc:
        return {"error": _enrich_kuzu_error(exc, q, graph_path)}
    vf = visibility_filter or get_effective_visibility_filter()
    if vf is not None:
        rows = filter_cypher_rows_for_visibility(
            rows,
            visibility_filter=vf,
            connection_name=str(entry.connection_name),
        )
    result: dict[str, Any] = {"rows": rows}
    if auto_picked_warning is not None:
        result["warning"] = auto_picked_warning
    return result
