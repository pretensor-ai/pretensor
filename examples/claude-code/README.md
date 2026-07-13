# Claude Code example

Connect a Pretensor index to [Claude Code](https://docs.anthropic.com/en/docs/claude-code) so Claude can answer schema questions, trace join paths, and explore your database's knowledge graph from the terminal.

## Prerequisites

- Pretensor installed and an index built — follow [Pagila](../pagila/) or [Snowflake TPCH](../snowflake-tpch/) first
- Claude Code installed: `npm install -g @anthropic-ai/claude-code`

## Option A — one command (recommended)

From the directory where you ran `pretensor index` (the one containing `.pretensor/`):

```bash
claude mcp add pretensor -- pretensor serve --graph-dir "$(pwd)/.pretensor"
```

By default the server is registered for the current project only. Add `--scope user` to make it available in every project, or `--scope project` to write a `.mcp.json` you can commit and share with your team.

## Option B — config file

Copy [`mcp.json`](mcp.json) to your project root as `.mcp.json` and replace the path:

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

Start `claude` and run `/mcp` — `pretensor` should be listed with its tools. Then try:

```
What tables are in the pagila database?
```

```
Show me the join path from customer to film.
```

## Troubleshooting

- **`pretensor` not found** — Claude Code launches MCP servers with your login shell's `PATH`. If Pretensor is installed in a virtualenv, replace `"command": "pretensor"` with the absolute path from `which pretensor`.
- **No databases listed** — `--graph-dir` must be an absolute path to the `.pretensor/` directory created by `pretensor index`. Relative paths resolve against Claude Code's working directory, not yours.
