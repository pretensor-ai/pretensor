# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) versioning.

Pretensor is currently in alpha on PyPI (`pip install pretensor` —
no `--pre` needed yet because no stable release exists). Until `1.0.0`
ships, the schema and CLI surface can change between versions; this
changelog tracks what moves between releases.

## [Unreleased]

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
  entries — use `column_name`.
- `scripts/benchmark_graph_rag_nl2sql.py` — unrunnable after the LLM
  stack was stripped from OSS (imported the since-deleted `suggest_query` MCP
  tool). The new `pretensor benchmark` CLI covers its namespace.
- `scripts/e2e_pagila.py` — unrunnable for the same reason (imported deleted
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
- Initial public alpha on PyPI — reserves the `pretensor` name and ships
  the first published build of the schema-introspection MCP server, CLI,
  and bundled connectors.

<!--
When cutting a release, copy the contents of [Unreleased] into a new
`## [X.Y.Z] - YYYY-MM-DD` section above, then reset [Unreleased] to empty
subsections (Added / Changed / Deprecated / Removed / Fixed / Security).
-->
