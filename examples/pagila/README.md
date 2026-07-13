# Pagila example

[Pagila](https://github.com/devrimgunduz/pagila) is a sample DVD rental PostgreSQL database. It is bundled with Pretensor and used as the default quickstart target — you can go from a fresh install to a working MCP server in under five minutes without any credentials or cloud accounts.

## Prerequisites

- Python 3.11 or 3.12
- [Docker](https://docs.docker.com/get-docker/) (Desktop or Engine)
- Pretensor installed: `pip install pretensor`

## Option A — one command (recommended)

```bash
pretensor quickstart
```

This starts a throwaway Postgres container pre-loaded with Pagila on `localhost:55432`, indexes it into `./.pretensor/`, and prints the `mcpServers` JSON snippet to paste into your IDE. When you are done:

```bash
pretensor quickstart --down
```

## Option B — Docker Compose

Use this if you want the full upstream Pagila dataset, a database that outlives the quickstart container, or more control.

**Start Pagila** (from this directory):

```bash
curl -O https://raw.githubusercontent.com/devrimgunduz/pagila/master/pagila-schema.sql
curl -O https://raw.githubusercontent.com/devrimgunduz/pagila/master/pagila-data.sql
docker compose up -d
```

The [compose file](docker-compose.yml) mounts both SQL files into Postgres's init directory, so the database comes up pre-loaded on `localhost:5433` (5433 to avoid clashing with a Postgres you may already run on 5432).

**Index and get the config snippet:**

```bash
pretensor index "postgresql://postgres:postgres@localhost:5433/pagila" --name pagila
pretensor serve --config-only
```

> If you used `pretensor quickstart` (port 55432), the DSN is
> `postgresql://postgres:postgres@localhost:55432/pagila`.

When you are done: `docker compose down -v`.

## Connect to your IDE

Paste the `mcpServers` JSON output into your IDE's MCP config:

- [Claude Code](../claude-code/)
- [Cursor](../cursor/)
- [Windsurf](../windsurf/)

## Sample prompts

Once connected, try these to get a feel for what Pretensor surfaces:

```
What tables are in the pagila database?
```

```
Show me the join path from customer to film.
```

```
Which tables are in the same cluster as payment? What do they represent?
```

```
What columns does the rental table have, and what foreign keys does it carry?
```

The quickstart bundle ships a trimmed Pagila (8 tables across the `public` and `staff` schemas); the full upstream Pagila loaded by the compose file has 15 tables plus payment partitions, all in `public`. Either way the schema is small enough that every query returns in milliseconds — a clean signal that the MCP server is working before you point it at a real warehouse.
