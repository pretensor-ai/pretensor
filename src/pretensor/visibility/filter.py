"""Runtime visibility checks for tables and columns (fnmatch globs)."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from pretensor.visibility.config import VisibilityConfig

__all__ = ["VisibilityFilter"]


@dataclass(frozen=True, slots=True)
class VisibilityFilter:
    """Evaluates table/column visibility for one merged :class:`VisibilityConfig`."""

    _config: VisibilityConfig

    @classmethod
    def from_config(cls, config: VisibilityConfig) -> VisibilityFilter:
        return cls(_config=config)

    def is_table_visible(
        self,
        connection_name: str,
        schema_name: str,
        table_name: str,
        *,
        table_type: str | None = None,
    ) -> bool:
        """True if the physical table may appear in MCP responses and traversal."""
        if table_type and self._table_type_hidden(table_type):
            return False
        if self._schema_denied(schema_name):
            return False
        if self._allowed_schemas_active() and not self._schema_allowed(schema_name):
            return False
        if self._allowed_tables_active() and not self._table_allowed(
            connection_name, schema_name, table_name
        ):
            return False
        return not self._table_pattern_matches(connection_name, schema_name, table_name)

    def visible_columns(
        self,
        connection_name: str,
        schema_name: str,
        table_name: str,
        column_names: list[str],
    ) -> list[str]:
        """Return column names that are not hidden for this table."""
        return [
            c
            for c in column_names
            if not self._column_hidden(schema_name, table_name, c)
        ]

    def is_schema_table_node_id_visible(self, node_id: str) -> bool:
        """Interpret ``connection::schema::table`` node ids (SchemaTable)."""
        parts = node_id.split("::")
        if len(parts) != 3:
            return True
        connection_name, schema_name, table_name = parts
        return self.is_table_visible(connection_name, schema_name, table_name)

    def _allowed_schemas_active(self) -> bool:
        return bool(self._config.allowed_schemas)

    def _schema_allowed(self, schema_name: str) -> bool:
        for pat in self._config.allowed_schemas:
            if fnmatch.fnmatchcase(schema_name, pat):
                return True
        return False

    def _allowed_tables_active(self) -> bool:
        return bool(self._config.allowed_tables)

    def _table_allowed(
        self, connection_name: str, schema_name: str, table_name: str
    ) -> bool:
        qualified = f"{schema_name}.{table_name}"
        conn_qualified = f"{connection_name}::{schema_name}.{table_name}"
        for pat in self._config.allowed_tables:
            if not pat.strip():
                continue
            if "::" in pat:
                if fnmatch.fnmatchcase(conn_qualified, pat):
                    return True
                continue
            if "." in pat:
                if fnmatch.fnmatchcase(qualified, pat):
                    return True
                continue
            if fnmatch.fnmatchcase(table_name, pat):
                return True
        return False

    def _schema_denied(self, schema_name: str) -> bool:
        for pat in self._config.hidden_schemas:
            if fnmatch.fnmatchcase(schema_name, pat):
                return True
        return False

    def _table_pattern_matches(
        self, connection_name: str, schema_name: str, table_name: str
    ) -> bool:
        qualified = f"{schema_name}.{table_name}"
        conn_qualified = f"{connection_name}::{schema_name}.{table_name}"
        for pat in self._config.hidden_tables:
            if not pat.strip():
                continue
            if "::" in pat:
                if fnmatch.fnmatchcase(conn_qualified, pat):
                    return True
                continue
            if "." in pat:
                if fnmatch.fnmatchcase(qualified, pat):
                    return True
                continue
            if fnmatch.fnmatchcase(table_name, pat):
                return True
        return False

    def _table_type_hidden(self, table_type: str) -> bool:
        tt = table_type.strip().lower()
        return tt in {t.strip().lower() for t in self._config.hidden_table_types}

    def _column_hidden(self, schema_name: str, table_name: str, column_name: str) -> bool:
        qualified_3 = f"{schema_name}.{table_name}.{column_name}"
        qualified_2 = f"{table_name}.{column_name}"
        for pat in self._config.hidden_columns:
            if not pat.strip():
                continue
            if pat.count(".") >= 2:
                if fnmatch.fnmatchcase(qualified_3, pat):
                    return True
            elif "." in pat:
                if fnmatch.fnmatchcase(qualified_2, pat):
                    return True
            else:
                if fnmatch.fnmatchcase(column_name, pat):
                    return True
        return False
