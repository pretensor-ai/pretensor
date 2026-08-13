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
- [9. Embeddings (optional)](#9-embeddings-optional)
- [10. Link application code to tables (`analyze`)](#10-link-application-code-to-tables-analyze)

---

## 0. One-command quickstart

After installing Pretensor (Section 1), this is the fastest way to see it working end-to-end. Requires Docker.

```bash
pretensor quickstart
```

This:

1. Boots a throwaway Postgres on `localhost:55432` pre-loaded with the Pagila sample schema.
2. Indexes it into `./.pretensor/`.
3. Prints the `mcpServers` snippet to paste into Claude or Cursor.

Tear it down when finished:

```bash
pretensor quickstart --down
```

If you already have a Postgres reachable at `postgresql://postgres:postgres@localhost:55432/pagila`, pass `--no-docker` to skip the compose step.

> **Troubleshooting `permission denied` from `docker compose`.** Quickstart
> copies its compose file and SQL fixtures into `./.pretensor/quickstart/`
> before running `docker compose`, specifically so that snap-confined
> Docker (the default `snap install docker` on Ubuntu) can read them —
> that Docker can't read files under hidden directories in `$HOME`, which
> broke quickstart under `pipx install` (venv under
> `~/.local/pipx/venvs/...`). If you still see a permission error, check
> whether your current working directory is itself under a hidden
> directory.

---

## 1. Install

```bash
# This guide indexes Postgres, so install the postgres extra:
pip install 'pretensor[postgres]'
# or
uv pip install 'pretensor[postgres]'
```

Once installed, the `pretensor` CLI is on `$PATH` and Sections 2–8 below assume you can call it directly. From `0.1.0` on, `pip install pretensor` resolves to the latest non-alpha release; pre-releases require `--pre`.

> **The database driver is an extra.** A bare `pip install pretensor` installs no
> DB driver, so `pretensor index postgresql://…` fails at connect time with a
> hint to run `pip install 'pretensor[postgres]'`. This quickstart uses Postgres,
> hence the `[postgres]` extra above.

Optional features are exposed as extras:

| Extra | Adds | Use when |
|-------|------|----------|
| `pretensor[postgres]` | `psycopg2-binary` | You're indexing PostgreSQL (this guide). **Required** for Postgres. |
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
| `semantic_search` | Cosine ranking over indexed table embeddings (requires the `[embeddings]` extra; falls back to BM25 otherwise) |
| `cypher` | Read-only Kuzu Cypher; mutating clauses are rejected |
| `context` | Full context for one table: columns, classifier fields, joins, lineage, entity, cluster |
| `traverse` | Join paths between two physical tables |
| `impact` | Downstream tables reachable through FK and inferred-join edges; each reached table carries its external code consumers from `pretensor analyze` |
| `consumers` | External code locations (service, file, line range, read/write op, confidence) that consume one table — see [section 10](#10-link-application-code-to-tables-analyze) |
| `detect_changes` | Compare a live schema to the last indexed snapshot without mutating the graph |
| `compile_metric` | Compile semantic-layer YAML into validated SQL for one indexed database |
| `validate_sql` | Validate SQL against the indexed graph before execution |

**Resources** (readable via `pretensor://…`):

| URI | Content |
|-----|---------|
| `pretensor://databases` | Registry overview (markdown) |
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

---

## 9. Embeddings (optional)

The `[embeddings]` extra ships a local ONNX embedding model
(`Snowflake/snowflake-arctic-embed-xs`, 384-dim) — entirely on-device, no remote
API calls. With it installed, three new behaviors become available:

1. The `semantic_search` MCP tool ranks tables by cosine similarity to a
   natural-language query.
2. The existing `query` tool fuses BM25 results with a cosine pass via
   Reciprocal Rank Fusion.
3. The intelligence layer can opt into embedding-aware clustering, semantic
   candidate joins, and an embedding role-classification vote.

**Default behavior is unchanged when the extra is absent.** Every embedding-
using path falls back cleanly to its heuristic counterpart.

### Install + index

```bash
# From a source checkout
uv sync --extra embeddings

# Or as a package
pip install "pretensor[embeddings]"

# Indexing now computes embeddings automatically — installing the extra IS the opt-in
pretensor index postgresql://USER:PASSWORD@HOST:5432/DBNAME
```

With the extra installed, both `pretensor index` and `pretensor quickstart`
compute table embeddings by default. On `pretensor index`, opt out per-run
with `--no-embeddings`. `pretensor quickstart` has no such flag, so opt out
there (or anywhere process-wide) with `PRETENSOR_EMBEDDINGS_DISABLED=1`.
Without the extra, indexing is unchanged — there is nothing to configure.

The first index run downloads and caches the pinned model from the Hugging Face
Hub (~88 MB). Subsequent runs reuse the local cache. The embedding pass itself
adds roughly 1% to index time (one batched ONNX call for all tables).

### Use semantic_search

`semantic_search` is always present in the MCP tool list — it's registered
unconditionally alongside `query`, `context`, and the others. What changes
once tables carry vectors is its return shape: the tool runs
cosine-similarity ranking against indexed table vectors. Before any tables
have been embedded (extra absent, or indexed with `--no-embeddings`),
`semantic_search` returns a structured `fallback_bm25` envelope pointing the
caller at the `query` tool — it never errors, just degrades cleanly. From an
agent:

```jsonc
{
  "tool": "semantic_search",
  "args": {
    "query": "where do we record customer transactions?",
    "k": 5
  }
}
```

When no tables carry vectors (e.g. you indexed with `--no-embeddings`), the
tool returns a structured fallback envelope pointing the caller at `query`
(BM25); it never raises.

### Notes

* Heuristic output without `[embeddings]` is byte-identical to prior releases —
  the determinism contract is enforced by a parametric null-path parity test
  that every embedding-related PR must keep green.
* A `PRETENSOR_EMBEDDINGS_DISABLED=1` environment variable forces the null
  path even when the extra is installed — useful for incident kill-switch
  flips in production. A unit test (`test_env_kill_switch_forces_null_path`)
  asserts the env var stops the embedding pipeline from invoking the
  embedder; "extras present but disabled produces output identical to
  extras absent" is enforced operationally by that contract, not by a
  full extras-present CI lane (the dedicated L1+L2 null-path-parity CI
  lane is a follow-up tracked alongside the benchmark harness).

---

## 10. Link application code to tables (`analyze`)

Once a database is indexed, `pretensor analyze` scans an application
repository for SQL and records which code consumes which tables:

```bash
pretensor analyze path/to/service-repo --connection mydb

# Common variations
pretensor analyze . --connection mydb --service billing-api
pretensor analyze . --connection mydb --dry-run          # preview, no writes
pretensor analyze . --connection mydb --json             # machine-readable summary
pretensor analyze . --connection mydb --default-schema analytics
```

What it does:

* Walks the repo gitignore-aware (v1 scans Python files; `--include` /
  `--exclude` narrow the sweep, `--max-file-bytes` caps file size).
* Lifts SQL string literals from the AST by syntactic context — no code is
  executed — classifies them, and resolves table references with sqlglot.
  Unqualified table names resolve against `--default-schema` (default
  `public`).
* Writes one external-consumer record per code location with service, file,
  line range, read/write op, and confidence. Raw SQL text is never stored,
  only a fingerprint. Re-scanning the same service replaces its prior rows.
* References that don't match a table indexed under `--connection` are
  dropped (counted in the summary), never written to the wrong connection.

Opt a statement out with a `# noqa: pretensor-analyze` comment on the same
line or the line above. If the graph has no tables for the connection, the
command exits with a copy-pasteable `pretensor index` hint — it never
indexes on its own.

The results surface through MCP: the `consumers` tool answers "which
services read or write this table?" directly, and every table reached by
`impact` carries its consumer list.
