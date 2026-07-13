# Examples

Ready-to-run setups — each takes less than five minutes from a fresh install.

**Databases** — get a Pretensor index built:

| Example | What it covers |
|---------|----------------|
| [Pagila](pagila/) | Local Docker + Pagila sample database; works with `pretensor quickstart` |
| [Snowflake TPCH](snowflake-tpch/) | Snowflake built-in TPCH sample data walkthrough |

**IDEs** — connect the index to your AI tool:

| Example | What it covers |
|---------|----------------|
| [Claude Code](claude-code/) | `claude mcp add` one-liner + `.mcp.json` snippet |
| [Cursor](cursor/) | `~/.cursor/mcp.json` snippet |
| [Windsurf](windsurf/) | `~/.codeium/windsurf/mcp_config.json` snippet |

## Which example should I start with?

- **No database yet?** → Start with [Pagila](pagila/). It spins up a local Postgres in one command.
- **Already have Snowflake?** → [Snowflake TPCH](snowflake-tpch/) uses the sample data that every Snowflake account includes.

Then connect the IDE you use: [Claude Code](claude-code/), [Cursor](cursor/), or [Windsurf](windsurf/).
