# Pretensor OSS

[![PyPI](https://img.shields.io/pypi/v/pretensor.svg)](https://pypi.org/project/pretensor/)
[![CI](https://github.com/pretensor-ai/pretensor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pretensor-ai/pretensor/actions/workflows/ci.yml)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](#status)
[![Python: 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](#prerequisites)

**Pretensor OSS** introspects **PostgreSQL** and **Snowflake**, with optional **BigQuery** connector support, builds a **Kuzu** knowledge graph of tables, columns, foreign keys, inferred joins, and related metadata, and exposes that graph to AI tools through an **MCP** (Model Context Protocol) server. Agents query schema context and search without issuing raw SQL against your graph store.

> **Status: Alpha.** Pretensor is on PyPI as `pretensor` (latest alpha: `0.1.0a0`). CLI flags, MCP tools, and graph schema can still change between alpha versions — pin exact versions until `1.0.0`. See [docs/releases.md](https://github.com/pretensor-ai/pretensor/blob/main/docs/releases.md) for the versioning policy.

## Who is this for

- Data analysts using AI to explore warehouses.
- Data engineers tired of copy-pasting DDLs into chat.
- Data architects who need grounded schema context for agents.
- Anyone feeding database schemas to an LLM by hand.

## Prerequisites

- **Python 3.11 or 3.12** (3.13 not yet tested).
- A reachable database for `pretensor index`. PostgreSQL is the fastest local path; Snowflake and BigQuery are supported via the `pretensor[snowflake]` and `pretensor[bigquery]` extras.

## Install

```bash
pip install pretensor
# or, inside a uv-managed environment:
uv pip install pretensor
```

Optional features are exposed as extras:

| Extra | Adds | Use when |
|-------|------|----------|
| `pretensor[snowflake]` | `snowflake-sqlalchemy` | You're indexing a Snowflake warehouse. |
| `pretensor[bigquery]` | `google-cloud-bigquery` | You're indexing BigQuery. |
| `pretensor[clustering]` | `leidenalg` | You want Leiden community detection during indexing. Without this, Pretensor falls back to igraph Louvain (works, but no resolution tuning). |

Combine extras with comma separation, e.g. `pip install 'pretensor[snowflake,clustering]'`.

Try it without installing:

```bash
uvx --from pretensor pretensor --help
```

> **A note on alpha versions.** Pretensor is in alpha (`0.1.0aN`). The plain `pip install pretensor` command picks up the latest alpha because PyPI has no stable release yet. Once `1.0.0` ships, future alphas will require `--pre` (e.g. `pip install --pre pretensor`); pin to a specific version (`pretensor==0.1.0a0`) if you want a deterministic install today.

If you want to hack on Pretensor itself rather than use it, see the contributor setup in [CONTRIBUTING.md](https://github.com/pretensor-ai/pretensor/blob/main/CONTRIBUTING.md) for the `git clone` + `make install` flow.

## Quickstart

```bash
pretensor index postgresql://USER:PASSWORD@HOST:5432/DBNAME
pretensor serve --config-only   # prints mcpServers JSON for Claude / Cursor
```

`serve --config-only` prints the **`mcpServers` JSON** to stdout. Merge the `pretensor` entry into your Claude or Cursor MCP settings — the IDE starts the server automatically. Run `pretensor serve` directly if you prefer a long-running terminal process (config hints go to stderr, keeping stdout clean for JSON-RPC).

Use **`--state-dir`** on `index` / `reindex` and **`--graph-dir`** on `serve` when overriding the default state directory (`.pretensor`).

**Full guide — install, tools, visibility, reindexing, graph visualization:** [guides/quickstart.md](https://github.com/pretensor-ai/pretensor/blob/main/guides/quickstart.md)

## MCP tools

| Name | Role |
|------|------|
| `list_databases` | List indexed database connections with table counts and staleness. |
| `schema` | Inspect node labels, edge types, and available properties before writing Cypher. |
| `query` | BM25 keyword search over table and entity metadata. |
| `cypher` | Read-only Kuzu Cypher for one indexed database; mutating clauses are rejected. |
| `context` | Full context for one physical table, including columns, joins, lineage, and cluster metadata. |
| `traverse` | Join paths between two physical tables, including confirmed cross-database paths. |
| `impact` | Downstream tables reachable from a table via FK and inferred-join edges. |
| `detect_changes` | Compare the live database schema to the last indexed snapshot without mutating the graph. |
| `compile_metric` | Compile semantic-layer YAML into validated SQL for one indexed database. |
| `validate_sql` | Validate SQL against the indexed graph before execution. |

## Architecture

`src/pretensor/` is organized by subsystem:

- **`connectors/`** — database-specific introspection (PostgreSQL, Snowflake, BigQuery)
- **`core/`** — Kuzu graph store, schema writing, relationship discovery
- **`intelligence/`** — deterministic graph intelligence (classification, clustering, join-path precomputation; metric-template code exists but is not part of the default OSS indexing flow)
- **`mcp/`** — MCP server, tools, resources
- **`cli/`** — Typer CLI (`index`, `reindex`, `serve`, `list`, `quickstart`, `export`, `validate`, `sync-grants`, `add`, `remove`, plus the `semantic` subcommand group)

## Status

Pretensor is in **pre-release development**. Before the first packaged release:

- The package on PyPI is named `pretensor`. The first stable release will be `1.0.0`; everything before that is alpha. `pip install pretensor` works today because no stable version exists yet — `--pre` will be required once `1.0.0` ships and future alphas resume.
- There is no SemVer stability guarantee yet, so CLI flags, MCP tools, and graph schema may change between alphas. Pin exact versions.
- Treat current builds as evaluation software and test upgrades in a staging environment before production use.

Progress and release notes: [CHANGELOG.md](https://github.com/pretensor-ai/pretensor/blob/main/CHANGELOG.md).

## Contributing

See [CONTRIBUTING.md](https://github.com/pretensor-ai/pretensor/blob/main/CONTRIBUTING.md). Security issues: see [SECURITY.md](https://github.com/pretensor-ai/pretensor/blob/main/SECURITY.md).

## Tests

```bash
make verify
```

Individual commands are also available:

```bash
make test      # pytest
make lint      # ruff check
make typecheck # pyright
```

## License

MIT — see [LICENSE](https://github.com/pretensor-ai/pretensor/blob/main/LICENSE).
