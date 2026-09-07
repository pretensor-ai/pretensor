# Pretensor OSS

[![PyPI](https://img.shields.io/pypi/v/pretensor.svg)](https://pypi.org/project/pretensor/)
[![CI](https://github.com/pretensor-ai/pretensor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pretensor-ai/pretensor/actions/workflows/ci.yml)
[![Bench](https://github.com/pretensor-ai/pretensor/actions/workflows/bench.yml/badge.svg?branch=main)](https://github.com/pretensor-ai/pretensor/actions/workflows/bench.yml)
[![Status: Beta](https://img.shields.io/badge/status-beta-blue.svg)](#status)
[![Python: 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](#prerequisites)

**Pretensor OSS** introspects **PostgreSQL** and **Snowflake**, with optional **BigQuery** connector support, builds a **Kuzu** knowledge graph of tables, columns, foreign keys, inferred joins, and related metadata, and exposes that graph to AI tools through an **MCP** (Model Context Protocol) server. Agents query schema context and search without issuing raw SQL against your graph store.

> **Status: Beta.** Pretensor is on PyPI as `pretensor`; `0.1.0` is the first non-alpha release. CLI flags, MCP tools, and graph schema can still change between minor versions. Pin exact versions until `1.0.0`. See [docs/releases.md](https://github.com/pretensor-ai/pretensor/blob/main/docs/releases.md) for the versioning policy.

## Who is this for

- Data analysts using AI to explore warehouses.
- Data engineers tired of copy-pasting DDLs into chat.
- Data architects who need grounded schema context for agents.
- Anyone feeding database schemas to an LLM by hand.

## Prerequisites

- **Python 3.11 or 3.12** (3.13 not yet tested).
- A reachable database for `pretensor index`. Every database driver ships as an extra: PostgreSQL via `pretensor[postgres]`, Snowflake via `pretensor[snowflake]`, BigQuery via `pretensor[bigquery]`.

## Install

```bash
# Indexing PostgreSQL? Install the postgres extra:
pip install 'pretensor[postgres]'
# or, inside a uv-managed environment:
uv pip install 'pretensor[postgres]'
```

> **Heads-up:** the database drivers are *not* bundled in the base install. A
> bare `pip install pretensor` installs the CLI and MCP server but **no DB
> driver**. `pretensor index postgresql://…` will then fail at connect time
> with `Postgres connector requires psycopg2. Install the Postgres extra: pip
> install 'pretensor[postgres]' (or pip install psycopg2-binary).` Install the
> extra matching your database (`postgres`, `snowflake`, `bigquery`, or
> `mysql`), or `pretensor[all-connectors]` for all of them.

Optional features are exposed as extras:

| Extra | Adds | Use when |
|-------|------|----------|
| `pretensor[postgres]` | `psycopg2-binary` | You're indexing PostgreSQL. **Required**: a bare install has no Postgres driver. |
| `pretensor[snowflake]` | `snowflake-sqlalchemy` | You're indexing a Snowflake warehouse. |
| `pretensor[bigquery]` | `google-cloud-bigquery` | You're indexing BigQuery. |
| `pretensor[clustering]` | `leidenalg` | You want Leiden community detection during indexing. Without this, Pretensor falls back to igraph Louvain (works, but no resolution tuning). |
| `pretensor[embeddings]` | `onnxruntime`, `transformers`, `huggingface-hub`, `numpy` | You want local ONNX embeddings (`Snowflake/snowflake-arctic-embed-xs`, 384-dim). With the extra installed, `pretensor index` computes table embeddings automatically (opt out with `--no-embeddings` or `PRETENSOR_EMBEDDINGS_DISABLED=1`), the `semantic_search` MCP tool runs cosine-similarity ranking against indexed table vectors, and the `query` tool gains a hybrid BM25+cosine RRF rerank. The intelligence layer's experimental embedding-aware passes (clustering blend, role-classification vote, semantic candidate joins) remain opt-in config toggles. Without the extra, `semantic_search` is still registered but returns a structured `fallback_bm25` envelope; heuristic output is byte-identical to prior releases. |

Combine extras with comma separation, e.g. `pip install 'pretensor[postgres,clustering]'`.

Try it without installing:

```bash
uvx --from pretensor pretensor --help
```

> **A note on versions.** From `0.1.0` on, plain `pip install pretensor` resolves to the latest non-alpha release, and pre-releases require `--pre` (e.g. `pip install --pre pretensor`). Pin to a specific version (e.g. `pretensor==<version>`) if you want a deterministic install. See the [PyPI badge above](https://pypi.org/project/pretensor/) for the latest.

If you want to hack on Pretensor itself rather than use it, see the contributor setup in [CONTRIBUTING.md](https://github.com/pretensor-ai/pretensor/blob/main/CONTRIBUTING.md) for the `git clone` + `make install` flow.

## Quickstart

```bash
pip install 'pretensor[postgres]'
pretensor init
```

`init` finds your database connection, indexes it, links your code, and registers
pretensor with Claude or Cursor. Prefer to drive it yourself? Every step is still a
flag: see [guides/quickstart.md](https://github.com/pretensor-ai/pretensor/blob/main/guides/quickstart.md).

## Scan application code (`analyze`)

```bash
pretensor analyze path/to/service-repo --connection mydb
```

`analyze` scans a repository's Python (`.py`) and SQL (`.sql`) files: it lifts
SQL string literals from Python source via the stdlib AST (no code is
executed) and reads bare `.sql` files whole, resolves each statement's table
references with sqlglot, and links the issuing code to the matching tables in the graph as
external consumers: service, file, line range, read/write op, and a confidence
score. Raw SQL text is never stored, only a fingerprint. The results power the
`consumers` MCP tool and enrich `impact`, so an agent can answer "which
services consume this table?" with provenance.

Useful flags: `--service` labels the scanned repo (defaults to the directory
name), `--default-schema` sets the schema assumed for unqualified table names,
`--dry-run` previews without writing, `--json` emits a machine-readable
summary. A `# noqa: pretensor-analyze` comment on or above a statement opts it
out.

## MCP tools

| Name | Role |
|------|------|
| `list_databases` | List indexed database connections with table counts and staleness. |
| `schema` | Inspect node labels, edge types, and available properties before writing Cypher. |
| `query` | BM25 keyword search over table and entity metadata. Hybrid BM25 + cosine RRF rerank when `[embeddings]` is installed and tables carry vectors. |
| `semantic_search` | Cosine ranking over indexed `SchemaTable` embeddings. Requires `pretensor[embeddings]`; returns a structured BM25-fallback envelope when the extra is absent or no tables have been embedded. |
| `cypher` | Read-only Kuzu Cypher for one indexed database; mutating clauses are rejected. |
| `context` | Full context for one physical table, including columns, joins, lineage, and cluster metadata. Every relationship carries `confidence` + `source` provenance. Optional `include_similar` arg surfaces cross-cluster nearest neighbors when embeddings are present. |
| `traverse` | Join paths between two physical tables. Every step carries `confidence` + `source` provenance. When ambiguous and tables carry embeddings, ranks tied paths by embedding similarity. |
| `impact` | Downstream tables reachable from a table via FK and inferred-join edges. Each reached table carries the external code consumers found by `pretensor analyze`. |
| `consumers` | External code locations (service, file, line range, read/write op, confidence) that consume one table, from `pretensor analyze`. |
| `detect_changes` | Compare the live database schema to the last indexed snapshot without mutating the graph. |
| `compile_metric` | Compile semantic-layer YAML into validated SQL for one indexed database. The error string includes a "did you mean: …" suggestion list when an unresolved metric, table, or column name has close matches. |
| `validate_sql` | Validate SQL against the indexed graph before execution. |

### Join confidence & provenance

`context` (per relationship) and `traverse` (per path step) both report how a
join was derived, so an agent can decide how much to trust it before writing
SQL:

- `confidence` — a `0.0`–`1.0` score. Declared foreign keys always report
  `1.0`; inferred joins carry the score produced by the pass that found them.
- `source` — how the join was derived:
  - `declared_fk` — a real foreign-key constraint in the database. Always
    confidence `1.0`.
  - `heuristic` — inferred from naming conventions and column-type overlap.
  - `llm_inferred` — inferred by an LLM pass over schema metadata.
  - `embedding` — inferred from vector similarity between column/table
    descriptions.
  - `statistical` — a heuristic or LLM candidate whose confidence was
    re-scored using sampled value overlap; the original hypothesis source is
    folded into this value once the statistical pass adjusts it.
  - `entity_link` — cross-database bridge steps in `traverse` derived from a
    confirmed `SAME_ENTITY` link, not a same-database join.
- `reasoning` — an optional short, human-readable rationale for inferred
  joins (absent for declared FKs, where the constraint itself is the
  rationale).

Agents should prefer `declared_fk` and high-confidence edges when composing
SQL, and treat sub-`0.5` confidence as speculative — verify with
`validate_sql` (or inspect `reasoning`) before relying on it.

## Agent-framework adapters

Agents that don't run over MCP can still reach the graph tools. Pretensor exposes
`schema`, `context`, `traverse`, `impact`, `query`, and `validate_sql` as native
tool objects for LangChain, LlamaIndex, and Google ADK. No MCP server process is
required. The adapters call the same underlying functions the MCP server uses, so
output is identical.

```python
from pathlib import Path
from pretensor.integrations import load_langchain_tools  # or load_llamaindex_tools, load_adk_tools

tools = load_langchain_tools(Path(".pretensor"))
```

Install the matching extra (`pretensor[langchain]`, `pretensor[llama-index]`, or
`pretensor[google-adk]`). The `docs/agent-framework-adapters.md` file has a full
example per framework.

## Architecture

`src/pretensor/` is organized by subsystem:

- **`connectors/`**: database-specific introspection (PostgreSQL, Snowflake, BigQuery)
- **`core/`**: Kuzu graph store, schema writing, relationship discovery
- **`intelligence/`**: deterministic graph intelligence (classification, clustering, join-path precomputation; metric-template code exists but is not part of the default OSS indexing flow)
- **`enrichment/`**: optional graph enrichment passes (dbt manifest, `analyze` code scanner)
- **`mcp/`**: MCP server, tools, resources
- **`cli/`**: Typer CLI (`init`, `index`, `reindex`, `analyze`, `serve`, `list`, `quickstart`, `export`, `validate`, `sync-grants`, `add`, `remove`, plus the `semantic` subcommand group)

## Status

Pretensor is **early software**:

- The package on PyPI is named `pretensor`. `0.1.0` is the first non-alpha release; pre-releases published after it require `--pre` to install.
- There is no SemVer stability guarantee before `1.0.0`, so CLI flags, MCP tools, and graph schema may still change between releases. Pin exact versions.
- Test upgrades in a staging environment before production use.

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

MIT: see [LICENSE](https://github.com/pretensor-ai/pretensor/blob/main/LICENSE).
