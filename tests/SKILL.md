# tests/ — test suite contract

## Layout law

Test files **must** mirror the source tree one-for-one:

| Source | Tests |
|--------|-------|
| `src/pretensor/core/` | `tests/core/` |
| `src/pretensor/mcp/` | `tests/mcp/` |
| `src/pretensor/intelligence/` | `tests/intelligence/` |
| `src/pretensor/staleness/` | `tests/staleness/` |
| `src/pretensor/connectors/` | `tests/connectors/` |
| `src/pretensor/entities/` | `tests/entities/` |
| `src/pretensor/cross_db/` | `tests/cross_db/` |
| `src/pretensor/search/` | `tests/search/` |
| `src/pretensor/skills/` | `tests/skills/` |
| `src/pretensor/validation/` | `tests/validation/` |
| `src/pretensor/cli/` | `tests/cli/` |
| _(no source mirror)_ | `tests/e2e/` — Docker-backed E2E suite (layout exception) |

New source subdirectory → new matching test subdirectory. `tests/e2e/` is the one exception: it tests the assembled system end-to-end against a live Docker container, not a single source module.

## Shared infrastructure

| File | Role |
|------|------|
| `conftest.py` | Root-level fixtures; adds `src/` to `sys.path` for bare `pytest` runs |
| `query_helpers.py` | Shared query utilities for MCP and integration tests |

Fixtures that apply to more than one subdirectory belong in the root `conftest.py`. Fixtures scoped to a single subdirectory belong in that subdirectory's own `conftest.py`.

## Running tests

```bash
# Full suite
uv run pytest tests/

# Single subsystem
uv run pytest tests/core/
uv run pytest tests/mcp/

# Single file
uv run pytest tests/intelligence/test_heuristic.py

# Via Makefile
make test
```

## Invariants

1. **New behavior requires new tests.** A PR that changes observable behavior without touching `tests/` is incomplete.
2. **Tests must pass before a PR is opened.** CI runs `pytest tests/` on every PR to `main`; a red suite blocks merge.
3. **No production mutations in tests.** Tests must not write to real databases, `.pretensor/` state, or `registry.json` outside of isolated temp directories.
4. **No real LLM calls in unit tests.** Tests that exercise LLM paths must mock the provider client or be explicitly gated behind an integration marker.
5. **Connector tests that require a live database** must be skippable when no DSN is available (use `pytest.mark.skip` or environment-gated fixtures — never hard-fail in CI without the DB).
6. **`conftest.py` path injection is the only `sys.path` manipulation allowed.** Do not add `PYTHONPATH` hacks inside individual test files.

## What does NOT belong here

- Performance or load tests (not in scope)
- Manual developer scripts → [`scripts/e2e_pagila.py`](../scripts/e2e_pagila.py) (unchanged; separate tool)

## E2E tests

`tests/e2e/` is a Docker-backed pytest suite that exercises the full pretensor pipeline against a real Postgres container. It is collected by `pytest tests/`, but the modules skip themselves unless `PRETENSOR_E2E=1` is set.

### Invariants

1. **`PRETENSOR_E2E=1` gate is mandatory in every E2E file.** Each file must have this guard at module level before any fixture or test definition:
   ```python
   import os, pytest
   if not os.getenv("PRETENSOR_E2E"):
       pytest.skip("set PRETENSOR_E2E=1", allow_module_level=True)
   ```
2. **No persistent containers.** The session-scoped `pagila_dsn` fixture owns the container lifecycle; teardown must stop and remove the container.
3. **Reindex tests must use `tmp_path` + a distinct `connection_name`.** Tests that mutate the Postgres schema (ALTER TABLE / CREATE TABLE) must build an isolated index in a `tmp_path`-based directory with a unique `connection_name` and must never touch the shared `indexed_state` session fixture.

### Running E2E tests

```bash
# On demand (requires Docker)
PRETENSOR_E2E=1 uv run pytest tests/e2e/ -v

# Via Makefile
make test-e2e

# Single file
PRETENSOR_E2E=1 uv run pytest tests/e2e/test_mcp_tools.py -v
```
