# Quickstart — Pretensor OSS

This guide takes you from zero to a running MCP server connected to a real database, then shows the current OSS testing, visibility, and graph-inspection flows.

- [0. One-command quickstart](#0-one-command-quickstart)
- [1. Install](#1-install)
- [2. Index your database](#2-index-your-database)
- [3. Connect to Claude or Cursor](#3-connect-to-claude-or-cursor)
- [4. MCP tools reference](#4-mcp-tools-reference)
- [5. Manual smoke test against Pagila](#5-manual-smoke-test-against-pagila)
- [6. Visualize the graph](#6-visualize-the-graph)
- [7. Visibility and profiles](#7-visibility-and-profiles)
- [8. Reindex after schema changes](#8-reindex-after-schema-changes)

---

## 0. One-command quickstart

After installing Pretensor (Section 1), this is the fastest way to see it working end-to-end. Requires Docker.

```bash
pretensor quickstart
```

This:

1. Boots a throwaway Postgres on `localhost:55432` pre-loaded with the Pagila sample schema (`docker/quickstart/docker-compose.yml`).
2. Indexes it into `./.pretensor/`.
3. Prints the `mcpServers` snippet to paste into Claude or Cursor.

Tear it down when finished:

```bash
pretensor quickstart --down
```

If you already have a Postgres reachable at `postgresql://postgres:postgres@localhost:55432/pagila`, pass `--no-docker` to skip the compose step.

---

## 1. Install

```bash
pip install pretensor
# or
uv pip install pretensor
```

Once installed, the `pretensor` CLI is on `$PATH` and Sections 2–8 below assume you can call it directly. Pretensor is currently in alpha; `pip install pretensor` picks up the latest alpha automatically because no stable release exists yet (once `1.0.0` ships, you'll need `--pre` to keep installing alphas).

Optional features are exposed as extras:

| Extra | Adds | Use when |
|-------|------|----------|
| `pretensor[snowflake]` | `snowflake-sqlalchemy` | You're indexing a Snowflake warehouse. |
| `pretensor[bigquery]` | `google-cloud-bigquery` | You're indexing BigQuery. |
| `pretensor[clustering]` | `leidenalg` | You want Leiden community detection during indexing. Without this, Pretensor falls back to igraph Louvain (works, but no resolution tuning). |

Combine extras with comma separation, e.g. `pip install 'pretensor[snowflake,clustering]'`.

**Prerequisites:** Python 3.11 or 3.12 (3.13 not yet tested). A reachable database for `pretensor index` — Postgres is the fastest local path; the manual smoke test in Section 5 spins one up via Docker.

### Hacking on Pretensor itself

If you're modifying Pretensor source rather than just using it, install from a checkout instead:

```bash
git clone https://github.com/pretensor-ai/pretensor.git
cd pretensor
make install   # uses uv if present, falls back to pip in .venv
```

`make install` installs the project editable with the `dev` extra and sets up the pre-commit hooks — what `CONTRIBUTING.md` expects for PR work.

---

## 2. Index your database

Graph state is written to `.pretensor/` by default; override with `--state-dir`.

```bash
# PostgreSQL
pretensor index postgresql://USER:PASS@HOST:5432/DBNAME

# Snowflake
pretensor index 'snowflake://USER:PASS@ACCOUNT/DB/SCHEMA?warehouse=WH'

# Custom name and state dir
pretensor index postgresql://... --name mydb --state-dir ~/my-graphs
```

Useful indexing flags:

```bash
# Shared graph for multiple registered connections
pretensor index postgresql://... --name mydb --unified

# Index-time visibility rules
pretensor index postgresql://... --visibility .pretensor/visibility.yml --profile analyst

# dbt enrichment
pretensor index postgresql://... --dbt-manifest path/to/manifest.json --dbt-sources path/to/sources.json
```

What indexing writes:

| Step | What gets written |
|------|------------------|
| Schema introspection | `SchemaTable` + `SchemaColumn` nodes, explicit FK edges, snapshots |
| Structural lineage | `LINEAGE` edges from connector metadata and optional dbt enrichment |
| Table classification | `role`, `role_confidence`, `classification_signals` on tables |
| Clustering | `Cluster` nodes and `IN_CLUSTER` edges |
| Join-path precomputation | `JoinPath` nodes reachable via FK and inferred joins |
| Metric templates | Not emitted by the default OSS `pretensor index` flow today |
| Skill file | `.claude/skills/pretensor-{name}/SKILL.md` by default (or see `--skills-target`) |

List what has been indexed:

```bash
pretensor list
```

---

## 3. Connect to Claude or Cursor

After indexing, print the `mcpServers` config snippet and add it to your IDE:

```bash
pretensor serve --config-only
```

The output looks like:

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

Merge the `pretensor` entry into your MCP settings. The IDE starts the server automatically when it connects. If you indexed with `--state-dir`, the generated path will point to that state directory instead.

To run the server manually:

```bash
pretensor serve --graph-dir .pretensor
```

Serve-time visibility uses the same config format as indexing:

```bash
pretensor serve --graph-dir .pretensor --visibility .pretensor/visibility.yml --profile analyst
```

After indexing, a compact skill file is written to help agents navigate the indexed graph:

```bash
pretensor index ... --skills-target claude
pretensor index ... --skills-target cursor
pretensor index ... --skills-target all
pretensor index ... --skills-target /tmp/my-graph-skill.md
```

---

## 4. MCP tools reference

| Tool | What it does |
|------|--------------|
| `list_databases` | List indexed database connections with table counts, schemas, capabilities, and staleness |
| `schema` | Inspect node labels, edge types, and available properties before writing Cypher |
| `query` | BM25 full-text search over table and entity metadata; optional `db` filter |
| `cypher` | Read-only Kuzu Cypher; mutating clauses are rejected |
| `context` | Full context for one table: columns, classifier fields, joins, lineage, entity, cluster |
| `traverse` | Join paths between two physical tables, including confirmed cross-database paths |
| `impact` | Downstream tables reachable through FK and inferred-join edges |
| `detect_changes` | Compare a live schema to the last indexed snapshot without mutating the graph |
| `compile_metric` | Compile semantic-layer YAML into validated SQL for one indexed database |
| `validate_sql` | Validate SQL against the indexed graph before execution |

**Resources** (readable via `pretensor://…`):

| URI | Content |
|-----|---------|
| `pretensor://databases` | Registry overview (markdown) |
| `pretensor://cross-db/entities` | Confirmed cross-database entity links |
| `pretensor://db/{name}/overview` | Per-database stats |
| `pretensor://db/{name}/clusters` | Cluster groupings |
| `pretensor://db/{name}/metrics` | `MetricTemplate` nodes and dependent tables |

---

## 5. Manual smoke test against Pagila

[Pagila](https://github.com/devrimgunduz/pagila) is a small PostgreSQL sample database that works well for a manual end-to-end check.

Start Pagila:

```bash
docker run -d --name pagila \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  ghcr.io/devrimgunduz/pagila:latest
```

Index it and inspect the registry:

```bash
pretensor index "postgresql://postgres:postgres@localhost:5432/pagila" --name pagila
pretensor list
pretensor serve --config-only --graph-dir .pretensor
```

For repository-level verification, run the automated checks from the repo root:

```bash
make verify
```

The `Makefile` at the repo root shows the individual `uv` commands (`make test`, `make lint`, `make typecheck`, `make format`) if you want to run them separately.

For a larger regression surface on traversal behavior, run `make test-e2e` — this executes the AdventureWorks, TPC-H, and other end-to-end suites under `tests/e2e/`.

---

## 6. Visualize the graph

Kuzu ships an official web UI ([Kuzu Explorer](https://github.com/kuzudb/explorer)) that connects directly to the `.kuzu` file and renders Cypher query results as an interactive graph.

Mount the **parent `graphs/` directory** (not the `.kuzu` file directly) and tell Explorer which database to open via `KUZU_FILE`:

```bash
docker run --rm -p 8888:8000 \
  -e KUZU_FILE=pagila.kuzu \
  -v "$(pwd)/.pretensor/graphs:/database" \
  --name pretensor-kuzu-explorer \
  kuzudb/explorer:latest
```

Open **http://localhost:8888**.

Starter queries:

```cypher
-- Tables and their FK connections
MATCH (a:SchemaTable)-[r:FK_REFERENCES]->(b:SchemaTable)
RETURN a, r, b LIMIT 60

-- Cluster membership
MATCH (t:SchemaTable)-[:IN_CLUSTER]->(c:Cluster)
RETURN t, c LIMIT 80

-- Metric templates and their source tables
MATCH (m:MetricTemplate)-[:METRIC_DEPENDS]->(t:SchemaTable)
RETURN m, t

-- One table and its columns
MATCH (t:SchemaTable {table_name: 'film'})-[:HAS_COLUMN]->(c:SchemaColumn)
RETURN t, c
```

Stop the container when done:

```bash
docker stop pretensor-kuzu-explorer
```

---

## 7. Visibility and profiles

Visibility rules live in `visibility.yml` under the state directory by default.

Example:

```yaml
hidden_schemas:
  - information_schema
hidden_tables:
  - public.audit_*
allowed_tables:
  - public.customer
  - public.payment
profiles:
  analyst:
    allowed_tables:
      - public.customer
      - public.rental
```

Use the rules during indexing:

```bash
pretensor index postgresql://... --visibility .pretensor/visibility.yml --profile analyst
```

Or apply a serve-time restriction without rebuilding the graph:

```bash
pretensor serve --graph-dir .pretensor --profile analyst
```

To generate role-keyed profiles from database grants:

```bash
pretensor sync-grants --dsn postgresql://ADMIN:PASS@HOST:5432/DB --output .pretensor/visibility.yml
```

---

## 8. Reindex after schema changes

When your schema changes, run `reindex` to diff and patch the graph without rebuilding from scratch:

```bash
# Preview what would change
pretensor reindex postgresql://... --dry-run

# Apply the diff
pretensor reindex postgresql://...

# Recompute intelligence artifacts after the patch
pretensor reindex postgresql://... --recompute-intelligence
```

After reindex, stale `MetricTemplate` nodes remain marked stale until the next full intelligence recomputation.
