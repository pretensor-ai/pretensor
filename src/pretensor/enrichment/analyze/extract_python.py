"""Stdlib ``ast`` extractor for SQL-bearing string literals in Python source."""

from __future__ import annotations

import ast
import io
import logging
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SqlCandidate", "extract_sql_candidates"]

logger = logging.getLogger(__name__)

_SQL_IDENTIFIER_NAMES: frozenset[str] = frozenset({"query", "sql", "stmt", "statement"})
_SQL_IDENTIFIER_SUFFIXES: tuple[str, ...] = ("_query", "_sql", "_stmt", "_statement")
_SQL_IDENTIFIER_PREFIXES: tuple[str, ...] = ("query_", "sql_", "stmt_", "statement_")
_SQL_METHOD_NAMES: frozenset[str] = frozenset(
    {"execute", "executemany", "query", "raw", "exec_driver_sql"}
)
_NOQA_MARKER = "noqa: pretensor-analyze"


def _is_sql_identifier(name: str) -> bool:
    """True for variable names that conventionally hold SQL text.

    Exact names (``sql``, ``query``, …) plus affixed forms (``merge_sql``,
    ``orders_query``, ``sql_fetch_users``). The is-this-SQL classifier still
    gates every candidate, so a ``graphql_query`` holding non-SQL text is
    rejected downstream rather than here.
    """
    lowered = name.lower()
    return (
        lowered in _SQL_IDENTIFIER_NAMES
        or lowered.endswith(_SQL_IDENTIFIER_SUFFIXES)
        or lowered.startswith(_SQL_IDENTIFIER_PREFIXES)
    )


_SQL_FSTRING_PREFIX_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|MERGE|WITH)\b", re.IGNORECASE
)


@dataclass
class SqlCandidate:
    """A SQL string literal candidate extracted from Python source."""

    text: str
    confidence_bucket: (
        str  # "high" | "medium" | "low" — updated by pipeline for fstrings
    )
    line_start: int
    line_end: int
    file_path: Path
    symbol: str
    kind: str  # extraction context: "assignment" | "call" | "fstring" | "sql_file"


def extract_sql_candidates(file_path: Path) -> list[SqlCandidate]:
    """Extract SQL-bearing string literals from a Python source file.

    Uses syntactic context only — whether the string is assigned to a SQL-named
    variable or passed as the first argument to a SQL-execution method. SQL
    content classification is handled separately by ``classify.py``.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("extract_python: cannot read %s: %s", file_path, exc)
        return []

    noqa_lines = _collect_noqa_lines(source)

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        logger.debug("extract_python: syntax error in %s: %s", file_path, exc)
        return []

    visitor = _SqlVisitor(file_path, noqa_lines)
    visitor.visit(tree)
    return visitor.candidates


def _collect_noqa_lines(source: str) -> set[int]:
    """Return the set of line numbers (1-indexed) carrying a pretensor-analyze noqa marker."""
    noqa: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok_type, tok_val, (srow, _scol), *_ in tokens:
            if tok_type == tokenize.COMMENT and _NOQA_MARKER in tok_val:
                noqa.add(srow)
    except tokenize.TokenError:
        pass
    return noqa


class _SqlVisitor(ast.NodeVisitor):
    """AST visitor that collects SQL-bearing string literals by syntactic context."""

    def __init__(self, file_path: Path, noqa_lines: set[int]) -> None:
        self.file_path = file_path
        self.noqa_lines = noqa_lines
        self.candidates: list[SqlCandidate] = []

    def _is_noqa(self, line_start: int) -> bool:
        return line_start in self.noqa_lines or (line_start - 1) in self.noqa_lines

    def _emit_constant(
        self,
        node: ast.Constant,
        *,
        symbol: str,
        kind: str,
        confidence_bucket: str = "high",
    ) -> None:
        if not isinstance(node.value, str):
            return
        text = node.value
        if not text.strip():
            return
        line_start = node.lineno
        line_end = getattr(node, "end_lineno", node.lineno)
        if self._is_noqa(line_start):
            return
        self.candidates.append(
            SqlCandidate(
                text=text,
                confidence_bucket=confidence_bucket,
                line_start=line_start,
                line_end=line_end,
                file_path=self.file_path,
                symbol=symbol,
                kind=kind,
            )
        )

    def _emit_fstring(self, node: ast.JoinedStr, *, symbol: str) -> None:
        """Emit a candidate for an f-string whose first element is a SQL-keyword prefix."""
        if not node.values:
            return
        first = node.values[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            return
        prefix = first.value
        if not _SQL_FSTRING_PREFIX_RE.match(prefix):
            return
        line_start = node.lineno
        line_end = getattr(node, "end_lineno", node.lineno)
        if self._is_noqa(line_start):
            return
        self.candidates.append(
            SqlCandidate(
                text=prefix,
                confidence_bucket="low",
                line_start=line_start,
                line_end=line_end,
                file_path=self.file_path,
                symbol=symbol,
                kind="fstring",
            )
        )

    def _handle_value(self, value: ast.expr, *, symbol: str, kind: str) -> None:
        if isinstance(value, ast.Constant):
            self._emit_constant(value, symbol=symbol, kind=kind)
        elif isinstance(value, ast.JoinedStr):
            self._emit_fstring(value, symbol=symbol)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and _is_sql_identifier(target.id):
                self._handle_value(node.value, symbol=target.id, kind="assignment")
                break
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            isinstance(node.target, ast.Name)
            and _is_sql_identifier(node.target.id)
            and node.value is not None
        ):
            self._handle_value(node.value, symbol=node.target.id, kind="assignment")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr.lower() in _SQL_METHOD_NAMES
            and node.args
        ):
            self._handle_value(node.args[0], symbol=node.func.attr, kind="call")
        self.generic_visit(node)
