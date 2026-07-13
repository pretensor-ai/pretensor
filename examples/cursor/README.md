# Cursor example

Connect a Pretensor index to [Cursor](https://cursor.com) so the agent can answer schema questions, trace join paths, and explore your database's knowledge graph from the editor.

## Prerequisites

- Pretensor installed and an index built — follow [Pagila](../pagila/) or [Snowflake TPCH](../snowflake-tpch/) first
- Cursor installed

## Add the MCP server

Cursor reads MCP servers from two places:

| File | Scope |
|------|-------|
| `~/.cursor/mcp.json` | All projects |
| `<project>/.cursor/mcp.json` | One project |

Copy [`mcp.json`](mcp.json) to either location and replace the path:

```json
{
  "mcpServers": {
    "pretensor": {
      "command": "pretensor",
      "args": ["serve", "--graph-dir", "/absolute/path/to/.pretensor"]
    }
  }
}
```

> **Tip:** `pretensor serve --config-only` prints exactly this block with the path already filled in.

## Verify

Open **Cursor Settings → MCP**. `pretensor` should appear with a green dot and a list of tools. Then open the chat in Agent mode and try:

```
What tables are in the pagila database?
```

```
Show me the join path from customer to film.
```

## Troubleshooting

- **Red dot / "failed to start"** — Cursor launches MCP servers with your login shell's `PATH`. If Pretensor is installed in a virtualenv, replace `"command": "pretensor"` with the absolute path from `which pretensor`.
- **No databases listed** — `--graph-dir` must be an absolute path to the `.pretensor/` directory created by `pretensor index`.
- **Edited the file but nothing changed** — toggle the server off and on in Cursor Settings → MCP, or restart Cursor.
