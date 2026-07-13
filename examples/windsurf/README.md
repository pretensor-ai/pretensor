# Windsurf example

Connect a Pretensor index to [Windsurf](https://windsurf.com) so Cascade can answer schema questions, trace join paths, and explore your database's knowledge graph from the editor.

## Prerequisites

- Pretensor installed and an index built — follow [Pagila](../pagila/) or [Snowflake TPCH](../snowflake-tpch/) first
- Windsurf installed

## Add the MCP server

Windsurf reads MCP servers from a single global file:

```
~/.codeium/windsurf/mcp_config.json
```

Copy [`mcp_config.json`](mcp_config.json) there (or merge the `pretensor` entry into your existing file) and replace the path:

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

You can also manage servers from the UI: open the Cascade panel, click the hammer/plugins icon, then **Configure** to edit the same file. Click **Refresh** after saving.

## Verify

In the Cascade panel's MCP/plugins list, `pretensor` should show as running with its tools. Then ask Cascade:

```
What tables are in the pagila database?
```

```
Show me the join path from customer to film.
```

## Troubleshooting

- **Server won't start** — Windsurf launches MCP servers with your login shell's `PATH`. If Pretensor is installed in a virtualenv, replace `"command": "pretensor"` with the absolute path from `which pretensor`.
- **No databases listed** — `--graph-dir` must be an absolute path to the `.pretensor/` directory created by `pretensor index`.
- **Edited the file but nothing changed** — click **Refresh** in the Cascade plugins panel, or restart Windsurf.
