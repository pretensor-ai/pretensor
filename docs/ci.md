# Continuous integration

Pretensor runs two GitHub Actions workflows.

## `ci.yml` — correctness gate

Runs on every push and every pull request:

- **Lint** — `ruff check` against `src` and `tests`.
- **Typecheck** — `pyright` against the project.
- **Test** — `pytest` across Python 3.11 and 3.12.

A PR cannot be merged unless these jobs pass.

## `bench.yml` — benchmark gate

The benchmark harness has three levels, split across two CI lanes.

### PR lane (gates merges to `main`)

For every pull request to `main`, runs:

- **L1** — graph-quality metrics (inferred-join precision/recall, cluster Jaccard, role F1) for Pagila, AdventureWorks, and TPC-H.
- **L2** — MCP tool quality metrics (`query` and `semantic_search` Recall@K, `traverse` correctness, `compile_metric` correctness) for the same three datasets.

L1 and L2 share a single matrix job (`pr-bench`) with axes `level × dataset × extras`. Each combination is exercised twice — once with only the `dev` extra installed, once with `dev,embeddings` — to confirm that having the embeddings extra installed does not perturb the heuristic-only path. Every leg compares its JSON output against the committed baseline at `tests/benchmark/baselines/<dataset>-<level>.json` and fails on any regression beyond the comparator's tolerance.

This lane is intended to be a **required status check**; promotion is a repo-settings change, applied by a maintainer.

### Release lane (reported, non-gating)

On release-tag pushes (`v*.*.*`) and on manual `workflow_dispatch`, runs:

- **L3** — end-to-end agent task success against Pagila, AdventureWorks, and TPC-H, with both runners: `baseline` (agent + raw schema) and `pretensor` (agent + MCP).

The lane uses `continue-on-error: true` and never blocks a tag cut. Each leg writes its output to `results.json` and uploads it as a workflow artifact with 90-day retention. L3 requires access to an LLM provider; the workflow reads its API key from a repo secret that the PR lane has no access to.

The L3 runners are stubs in this release; until they land, each leg exits early without producing `results.json` and the upload step warn-skips. The lane is wired now so the trigger, secrets, and artifact pipeline are in place when the runner code merges.

## Updating baselines

Baselines under `tests/benchmark/baselines/` are committed to `main` and require explicit PR review to change. Regenerate locally with:

```bash
uv run pretensor benchmark l1 --dataset <name> --out tests/benchmark/baselines/<name>-l1.json
uv run pretensor benchmark l2 --dataset <name> --out tests/benchmark/baselines/<name>-l2.json
```

Open a PR; reviewers should sanity-check that the metric movement is intentional before approving.
