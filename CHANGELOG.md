# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) versioning.

Pretensor is currently in alpha on PyPI (`pip install pretensor` —
no `--pre` needed yet because no stable release exists). Until `1.0.0`
ships, the schema and CLI surface can change between versions; this
changelog tracks what moves between releases.

## [Unreleased]

### Added
- CD pipeline at `.github/workflows/release.yml` that publishes tagged
  `v*.*.*` versions to PyPI via OIDC trusted publishing, routing
  prereleases through TestPyPI first.
- `"Typing :: Typed"` PyPI classifier, pairing the existing
  `src/pretensor/py.typed` PEP 561 marker with its public advertisement.
- Release runbook sections in `docs/releases.md` covering publishing,
  one-time OIDC setup, an optional pre-merge dry-run, cutting a release,
  and the post-transfer re-point procedure.
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

### Removed
- `scripts/benchmark_graph_rag_nl2sql.py` — unrunnable after the Cloud LLM
  stack was stripped from OSS (imported the since-deleted `suggest_query` MCP
  tool). The new `pretensor benchmark` CLI covers its namespace.
- `scripts/e2e_pagila.py` — unrunnable for the same reason (imported deleted
  `llm_runtime` helpers, `suggest_query_payload`, and the `llm_options` CLI
  module). The Docker-backed `tests/e2e/` suite is the canonical full-stack
  E2E path now.

<!--
When cutting a release, copy the contents of [Unreleased] into a new
`## [X.Y.Z] - YYYY-MM-DD` section above, then reset [Unreleased] to empty
subsections (Added / Changed / Deprecated / Removed / Fixed / Security).
-->
