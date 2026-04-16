"""Generate a compact Claude/Cursor skill markdown from a Kuzu graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pretensor.core.registry import RegistryEntry
from pretensor.core.store import KuzuStore
from pretensor.mcp.server import _build_oss_registry

__all__ = ["SkillGenerator", "write_skill_files_for_index"]


def _render_tool_lines() -> list[str]:
    # Source the canonical tool list from the MCP registry so the skill body
    # never drifts from `pretensor serve`. Handlers are closures that this code
    # never invokes; the placeholder Path is only captured, never read.
    registry = _build_oss_registry(Path("."))
    out: list[str] = []
    for tool in registry.list_tools():
        desc = (tool.description or "").strip().split("\n", 1)[0]
        first_sentence = desc.split(". ", 1)[0].rstrip(".")
        if first_sentence:
            out.append(f"- `{tool.name}` — {first_sentence}")
        else:
            out.append(f"- `{tool.name}`")
    return out


def write_skill_files_for_index(
    *,
    body: str,
    connection_name: str,
    cwd: Path | None = None,
    skills_target: str = "claude",
) -> list[Path]:
    """Write skill markdown to one or more paths (``--skills-target``)."""
    root = (cwd or Path.cwd()).resolve()
    safe = connection_name.replace("/", "_")
    key = skills_target.strip().lower() if skills_target.strip() else "claude"
    out: list[Path] = []

    def _write_claude() -> Path:
        out_dir = root / ".claude" / "skills" / f"pretensor-{safe}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "SKILL.md"
        path.write_text(body, encoding="utf-8")
        return path

    def _write_cursor() -> Path:
        out_dir = root / ".cursor" / "rules"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"pretensor-{safe}.mdc"
        path.write_text(body, encoding="utf-8")
        return path

    if key in ("claude", ""):
        out.append(_write_claude())
    elif key == "cursor":
        out.append(_write_cursor())
    elif key == "all":
        out.append(_write_claude())
        out.append(_write_cursor())
    else:
        custom = Path(skills_target).expanduser()
        if not custom.is_absolute():
            custom = root / custom
        custom.parent.mkdir(parents=True, exist_ok=True)
        custom.write_text(body, encoding="utf-8")
        out.append(custom)
    return out


@dataclass(frozen=True, slots=True)
class SkillGenerator:
    """Build `.claude/skills/pretensor-{db_name}/SKILL.md` after indexing."""

    store: KuzuStore
    connection_name: str
    database: str
    last_indexed_at: datetime

    def render(self) -> str:
        """Return markdown body for the skill file."""
        table_count, row_sum = self._table_stats()
        entities = self._entity_map_lines()
        top_joins = self._top_joins()

        safe = self.connection_name.replace("/", "_")
        skill_name = f"pretensor-{safe}"
        skill_description = (
            f"Pretensor graph for the `{self.database}` database "
            f"(connection `{self.connection_name}`, {table_count} tables). "
            "Use to look up tables, joins, and MCP query tools for this database."
        )

        lines = [
            "---",
            f"name: {skill_name}",
            f"description: {skill_description}",
            "---",
            "",
            f"# Pretensor graph: {self.database}",
            "",
            "## Database overview",
            "",
            f"- **Connection name:** `{self.connection_name}`",
            f"- **Logical database:** `{self.database}`",
            f"- **Tables:** {table_count}",
            f"- **Approx. row count (sum over tables):** {row_sum}",
            f"- **Last indexed:** {self.last_indexed_at.isoformat()}",
            "",
            "## Key entities",
            "",
        ]
        if entities:
            lines.extend(entities)
        else:
            lines.append(
                "_No Entity nodes yet — run indexing with entity extraction enabled._"
            )
        lines.extend(
            [
                "",
                "## MCP tools (Pretensor)",
                "",
                "Use the **pretensor** MCP server (`pretensor serve`):",
                "",
                *_render_tool_lines(),
                "",
                "Resources: `pretensor://databases`, `pretensor://db/{name}/overview`.",
                "",
                "## Common patterns",
                "",
                "_Placeholder — join-path and query patterns land in a later epic._",
                "",
                "## Top relationships (sample)",
                "",
            ]
        )
        if top_joins:
            lines.extend(top_joins)
        else:
            lines.append("_No join edges recorded._")
        lines.append("")
        return "\n".join(lines)

    def _table_stats(self) -> tuple[int, int]:
        rows = self.store.query_all_rows(
            "MATCH (t:SchemaTable) RETURN count(*), sum(t.row_count)"
        )
        if not rows:
            return 0, 0
        c, s = rows[0]
        tc = int(c) if c is not None else 0
        rs = int(s) if s is not None else 0
        return tc, rs

    def _entity_map_lines(self) -> list[str]:
        pairs = self.store.query_all_rows(
            """
            MATCH (e:Entity)-[:REPRESENTS]->(t:SchemaTable)
            RETURN e.name, t.schema_name, t.table_name
            ORDER BY e.name, t.schema_name, t.table_name
            """
        )
        by_entity: dict[str, list[str]] = {}
        for name, schema_name, table_name in pairs:
            key = str(name)
            qn = f"{schema_name}.{table_name}"
            by_entity.setdefault(key, []).append(qn)
        names_only = self.store.query_all_rows(
            "MATCH (e:Entity) RETURN e.name ORDER BY e.name"
        )
        out: list[str] = []
        for (n,) in names_only:
            key = str(n)
            tlist = sorted(set(by_entity.get(key, [])))
            if tlist:
                joined = ", ".join(tlist)
                out.append(f"- **{key}** — {joined}")
            else:
                out.append(f"- **{key}**")
        return out

    def _top_joins(self) -> list[str]:
        fk = self.store.query_all_rows(
            """
            MATCH (a:SchemaTable)-[:FK_REFERENCES]->(b:SchemaTable)
            RETURN concat(a.schema_name, '.', a.table_name),
                   concat(b.schema_name, '.', b.table_name)
            LIMIT 8
            """
        )
        inf = self.store.query_all_rows(
            """
            MATCH (a:SchemaTable)-[r:INFERRED_JOIN]->(b:SchemaTable)
            RETURN concat(a.schema_name, '.', a.table_name),
                   concat(b.schema_name, '.', b.table_name),
                   r.confidence
            LIMIT 8
            """
        )
        lines: list[str] = []
        for src, dst in fk:
            lines.append(f"- `{src}` —[FK_REFERENCES]→ `{dst}`")
        for src, dst, conf in inf:
            conf_s = f" (confidence {float(conf):.2f})" if conf is not None else ""
            lines.append(f"- `{src}` —[INFERRED_JOIN]→ `{dst}`{conf_s}")
        return lines[:12]

    @classmethod
    def write_for_index(
        cls,
        *,
        store: KuzuStore,
        entry: RegistryEntry,
        cwd: Path | None = None,
        skills_target: str = "claude",
    ) -> list[Path]:
        """Write skill file(s) per ``skills_target`` (see ``write_skill_files_for_index``)."""
        body = cls(
            store=store,
            connection_name=entry.connection_name,
            database=entry.database,
            last_indexed_at=entry.last_indexed_at,
        ).render()
        return write_skill_files_for_index(
            body=body,
            connection_name=entry.connection_name,
            cwd=cwd,
            skills_target=skills_target,
        )
