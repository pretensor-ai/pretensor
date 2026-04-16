"""Parse PostgreSQL array literal text from catalog views (e.g. ``pg_stats``)."""

from __future__ import annotations

__all__ = ["parse_pg_array_literal"]


def parse_pg_array_literal(raw: str | None) -> list[str]:
    """Turn ``{a,b,"c,d"}``-style literals into string tokens (best-effort)."""
    if raw is None:
        return []
    s = raw.strip()
    if not s or s == "{}":
        return []
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        s = s[1:-1]
    if not s:
        return []
    out: list[str] = []
    buf: list[str] = []
    in_quote = False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_quote:
            if ch == "\\" and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == '"':
                in_quote = False
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue
        if ch == '"':
            in_quote = True
            i += 1
            continue
        if ch == ",":
            token = "".join(buf).strip()
            if token:
                out.append(token)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out
