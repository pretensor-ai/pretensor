# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) versioning.

`0.1.0` is the first non-alpha release: `pip install pretensor` resolves
to the latest non-prerelease version, and pre-releases published after it
require `--pre`. Until `1.0.0` ships, the schema and CLI surface can
still change between minor versions; this changelog tracks what moves
between releases.

## [0.2.0] - 2026-09-04

### Added
- The release gate honours a per-release
  `tests/benchmark/results/<tag>/accepted-regressions.toml`. Each entry names
  one dataset, level and metric, the value the release ships at, and a
  reason. The value is a floor, not a waiver: a run that is worse than the
  accepted value still blocks, and an entry that matches no regression is an
  error. Deliberate precision/recall trade-offs no longer require bypassing
  the gate.
- `pretensor init`, an interactive first-run setup wizard.
- `pretensor init`'s build-from-parts flow now supports mysql, snowflake,
  and bigquery in addition to postgres, and a `--dialect` flag to force the
  dialect instead of inferring it from the DSN.
- `analyze --all` over repositories configured in config.yaml.
- Sources can now hold a whole-DSN `url:` reference (e.g. `${DATABASE_URL}`)
  instead of individual connection fields. `pretensor init` writes one of
  these automatically when the indexed DSN came from a single environment
  variable, so `config.yaml` stays safe to commit; `pretensor index --source`
  and `--all` resolve the reference at run time.
- Bare `pretensor` (no subcommand) prints a one-line hint to run
  `pretensor init` when no connections are indexed yet, before showing help.
- `pretensor init --client <key>` (repeatable, one of `claude-code`,
  `claude-desktop`, `cursor`) to explicitly register with a detected MCP
  client under `--yes` or a non-interactive run, since merely detecting a
  client installed is not consent to write its config.
- `pretensor init` now detects Claude Desktop on Windows as well
  (`%APPDATA%\Claude\claude_desktop_config.json`).
- `pretensor analyze` scans bare `.sql` files, not just SQL embedded in
  Python source. Each `.sql` file is treated as one high-confidence
  consumer, with table references unioned across all of its statements.

### Security
- `pretensor quickstart` now binds its throwaway Postgres to `127.0.0.1:55432`
  instead of all interfaces. In 0.1.0 the sandbox listened on every interface
  with the default `postgres:postgres` credentials, and Docker publishes ports
  past host firewalls, so on a machine with a public IP it was reachable from
  the internet. If you ran the 0.1.0 quickstart on a server, run
  `pretensor quickstart --down`, upgrade, and start it again.

### Changed
- Same-name join inference is gated by how many tables share a column name
  (`GraphConfig.same_name_max_tables`, default 8). Hub columns shared by more
  than 8 tables (`customer_id` on every fact table, `date_id` on 40 tables)
  no longer produce same-name candidates, same-name confidence decays with
  how widely the name is shared, and a join path is only flagged ambiguous
  when another path ties its cost. On warehouse-sized schemas this cuts
  inferred edges and index time by an order of magnitude at the cost of some
  recall on hub columns. Set `same_name_max_tables` to `null` to restore the
  ungated behaviour.
- `analyze --connection` is now optional: it is required unless `--all` is
  passed. Omitting it without `--all` exits 1 with a runtime message
  instead of failing argument parsing with a usage error.
- `pretensor init`'s interactive flow now opens with a summary of everything
  inferred and a single Yes/No/Customize choice instead of asking a
  question per item up front. Yes proceeds with no further questions;
  Customize walks through the same per-item questions as before. After
  indexing (and after linking a repository), interactive runs offer to add
  another database connection or repository. When the optional
  `[embeddings]` extra is installed, the guided flow also asks once,
  before indexing, whether to compute table embeddings.
- `analyze --all --json` now emits a single JSON array of per-repository
  summaries instead of interleaving unparseable "Analyzing <path>" lines
  between them.

## [0.1.0] - 2026-08-13

### Fixed
- `pretensor serve` crashed at startup on fresh installs: the unbounded
  `mcp>=1.0` dependency resolved to the new `mcp` 2.0 major release,
  whose `Server` API is incompatible. The dependency is now capped to
  `mcp>=1.0,<2` (2.x support is tracked as follow-up work), and an
  end-to-end stdio smoke test now exercises the real CLI handshake so a
  startup-breaking dependency drift fails in CI instead of on users'
  machines.
- `pretensor serve` startup errors are printed to stderr instead of
  stdout. Stdout is the MCP JSON-RPC channel and clients only surface
  stderr in their logs, so a startup failure looked like a silent
  disconnect.
- `pretensor analyze`: statements quoting identifiers with backticks
  (MySQL, BigQuery `` `project.dataset.table` ``) no longer fail table
  resolution. The multi-dialect parse retry accepted the first dialect
  returning any refs, and generic-dialect tokenization of backticks
  produced garbage names that stopped the retry before the right dialect
  ran; implausible refs are now treated as a failed parse.
- `pretensor analyze`: table references resolve case-insensitively when
  exactly one indexed table matches, so conventional lowercase SQL
  resolves against Snowflake's uppercase-stored identifiers. Exact
  matches still win, and ambiguous case-variant twins stay dropped.
- `pretensor analyze`: MySQL `REPLACE INTO` statements are recognized and
  classified as writes (sqlglot cannot parse them in any dialect; the
  pre-parse normalizer rewrites them to `INSERT INTO`, which has the same
  write-target shape).
- `pretensor analyze`: SQL-bearing variable detection also matches
  affixed names (`merge_sql`, `orders_query`, `sql_fetch_users`), not
  only the exact names `query`/`sql`/`stmt`/`statement`. The is-this-SQL
  classifier still gates every candidate, so non-SQL strings in such
  variables are rejected as before.
- The MCP `cypher` tool no longer leaks a worker thread per timed-out
  query: all queries now run on one shared, bounded thread pool, and a
  timeout returns immediately instead of blocking until the runaway
  query finishes. Under sustained slow-query load the previous
  per-query executors could accumulate threads without limit.

### Added
- `pretensor analyze`: scan an application repository for SQL string
  literals in Python source and link the code that issues them to the
  indexed tables in the graph. Produces `ExternalConsumer` nodes and
  `CONSUMES` edges carrying provenance (service, file, line range,
  read/write op, confidence), written idempotently per scan run with
  stale rows swept on re-scan. Raw SQL text is never stored: only a
  `sha256[:16]` fingerprint and the resolved table references. Flags:
  required `--connection`, `--service`, repeatable `--include` /
  `--exclude`, `--max-file-bytes`, `--min-confidence`,
  `--default-schema` (schema assumed for unqualified table names),
  `--dry-run`, and `--json`. When the graph has no tables for the
  connection the command errors with a copy-pasteable `pretensor index`
  hint; it never auto-indexes. A `# noqa: pretensor-analyze` comment on
  or above a statement opts it out of extraction.
- `consumers` MCP tool: the external code locations that read or write
  one table (service, file, line range, op, kind, confidence), from the
  data produced by `pretensor analyze`. Optional `op` filter and
  `min_confidence` cutoff.
- The `impact` MCP tool now attaches a `consumers` list to every
  reached table. The field is additive. Existing callers that ignore
  it are unaffected.
- `pathspec` added to base dependencies (gitignore-aware repository
  walking for `analyze`).

## [0.1.0a4] - 2026-07-13

### Fixed
- `pretensor export` no longer crashes with `Parameter database_name not
  found`. The export query builder bound scope parameters that some node
  tables' queries never referenced (e.g. the semantic-layer tables), which
  Kuzu's prepared statements reject; parameters are now built to match
  exactly the filter each query generates. `export` is also covered by an
  end-to-end CLI test against a real graph now.
- `pretensor reindex <name>` with a registry connection name no longer
  dumps a raw traceback; it exits cleanly and suggests
  `pretensor reindex --database <name>` when the value matches a known
  connection.
- `pretensor quickstart` no longer points `docker compose` at the
  compose file inside `site-packages`. Snap-confined Docker (the default
  `snap install docker` on Ubuntu) can't read files under hidden
  directories in `$HOME`, so a `pipx install` (venv under
  `~/.local/pipx/venvs/...`) made quickstart fail with an opaque
  `permission denied` error. Quickstart now copies the compose file and
  Pagila SQL fixtures into `./.pretensor/quickstart/` (next to the graph
  output) before invoking compose, and prints a hint about snap Docker +
  hidden paths if a permission error slips through anyway.

### Added
- Agent-framework SDK adapters for LangChain, LlamaIndex, and Google ADK,
  exposing the MCP tools as native framework tools.
- `pretensor --version` prints the installed package version.
- CLI flag names unified across subcommands: the workspace directory is
  `--state-dir` everywhere and the connection selector is `--database`/`-d`
  everywhere. The previous spellings (`--graph-dir` on `serve`, `--db` on
  `validate`) keep working as hidden deprecated aliases.
- `SECURITY.md` with the supported-versions and responsible-disclosure
  policy.
- `pretensor benchmark {l1,l2,l3}` CLI scaffolding for the 3-level benchmark
  harness. Dispatch / I/O wiring only; metric computation and the L3 runners
  land in follow-up changes. `l1`/`l2` accept `--embeddings`; `l3` accepts
  `--runner {baseline,pretensor}`, `--model`, and `--seed`.
- `pretensor benchmark compare --baseline <path> --current <path>
  [--tolerance <float>]` regression-gate subcommand. Reads two
  `BenchmarkResult` JSON files, classifies metric changes by direction
  (`higher_is_better` drops and `lower_is_better` rises beyond tolerance
  fail), prints a human-readable diff to stderr, and exits `0` on no
  regression, `1` on regression, `2` on bad input (dataset/level mismatch
  or malformed JSON).
- `pretensor.benchmark.results` module: `BenchmarkResult` / `Metric`
  dataclasses, deterministic `write_json` / `read_json` (sorted keys,
  indent 2, trailing newline), `compare()` returning a `ComparisonReport`,
  and `write_csv()` for spreadsheet review.

### Changed
- The MCP server now reuses one graph connection per database for the
  lifetime of the server instead of re-opening the graph file (and
  re-running schema migration probes) on every tool call. A separate
  `pretensor reindex` run is picked up on server restart, as before.

### Security
- The MCP `cypher` tool now opens the graph in Kuzu's engine-level
  read-only mode, so mutations cannot commit even if the statement
  allowlist were bypassed; the allowlist itself also rejects DDL and
  file-I/O commands (`COPY`, `EXPORT`, `ATTACH`, `LOAD`, …) explicitly.
- Encryption keystore and registry writes are atomic and created with
  owner-only permissions; a warning is printed if an existing keystore is
  world-readable.
- MCP tool and resource errors returned to clients are sanitized: the full
  stack is logged server-side under a correlation id, and the client
  receives a generic message with that id instead of exception text or
  filesystem paths.

### Removed
- MCP `context` tool: removed the deprecated `name` field from column
  entries: use `column_name`.
- `scripts/benchmark_graph_rag_nl2sql.py`: unrunnable after the LLM
  stack was stripped from OSS (imported the since-deleted `suggest_query` MCP
  tool). The new `pretensor benchmark` CLI covers its namespace.
- `scripts/e2e_pagila.py`: unrunnable for the same reason (imported deleted
  `llm_runtime` helpers, `suggest_query_payload`, and the `llm_options` CLI
  module). The Docker-backed `tests/e2e/` suite is the canonical full-stack
  E2E path now.

## [0.1.0a3] - 2026-05-03

### Fixed
- Ship the quickstart assets in the wheel so the documented quickstart
  runs from a clean `pip install pretensor` (they were missing from the
  built distribution in earlier alphas).

## [0.1.0a2] - 2026-04-28

### Fixed
- Ship the `pretensor.skills` package data in the wheel; earlier wheels
  omitted it, so skill-backed flows failed on a clean install.

## [0.1.0a1] - 2026-04-26

### Added
- CD pipeline at `.github/workflows/release.yml` that publishes tagged
  `v*.*.*` versions to PyPI via OIDC trusted publishing, routing
  prereleases through TestPyPI first.
- `"Typing :: Typed"` PyPI classifier, pairing the existing
  `src/pretensor/py.typed` PEP 561 marker with its public advertisement.
- Release runbook sections in `docs/releases.md` covering publishing,
  one-time OIDC setup, an optional pre-merge dry-run, cutting a release,
  and the post-transfer re-point procedure.

### Fixed
- Drop the embedding dependencies from the `[all-connectors]` extra so it
  installs only connector drivers, not the heavy ML stack.

## [0.1.0a0] - 2026-04-17

### Added
- Initial public alpha on PyPI: reserves the `pretensor` name and ships
  the first published build of the schema-introspection MCP server, CLI,
  and bundled connectors.

<!--
When cutting a release, copy the contents of [Unreleased] into a new
`## [X.Y.Z] - YYYY-MM-DD` section above, then reset [Unreleased] to empty
subsections (Added / Changed / Deprecated / Removed / Fixed / Security).
-->
