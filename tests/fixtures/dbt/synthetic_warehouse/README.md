# Synthetic dbt warehouse fixture

Deterministic stand-in for a "real" warehouse-shape dbt manifest used by
`tests/e2e/test_dbt_warehouse_scale_enrichment.py` to regression-guard the dbt
manifest enrichment pipeline (`src/pretensor/enrichment/dbt/`) at ~1,500-model
scale.

## Why synthetic instead of vendoring `gitlab-data/analytics`

The original plan called for vendoring a pinned snapshot of
`gitlab-data/analytics`'s compiled `target/manifest.json`. That artifact is a
`dbt compile` output and is **not committed** in the upstream repo, so
producing one requires:

1. Cloning `gitlab-data/analytics` at a pinned commit
2. Installing `dbt-snowflake` plus the project's Python dependencies
3. Configuring a credential stub and running `dbt parse` against an offline
   target

That toolchain is too fragile to bake into a CI fixture. After discussion the
chore was reframed to ship a deterministic synthetic manifest that exercises
the same code paths the original vendored snapshot would have exercised:

- parser at warehouse scale (1,500 models, 150 sources, 12 exposures, 50 tests)
- resolver `(connection, schema, physical_name)` lookups
- lineage writer with realistic fan-out (some hub tables with 20+ children)
- exposure BFS propagation across multiple DAG levels
- description coverage / non-destructive metadata writes
- silent-skip path for dangling refs
- the ~30 s end-to-end performance budget

The trade-off is that the test cannot catch bugs that only trigger on
manifest oddities present in real-world dbt projects (deeply nested configs,
unicode descriptions, weird `ref`/`source` patterns). If a real GitLab manifest
ever becomes acquireable, a follow-up issue can swap in the vendored snapshot
without changing the assertion shape.

## Schema version

The generator emits `manifest/v10` — the newest version supported by
`src/pretensor/enrichment/dbt/manifest.py` (see `_SUPPORTED_SCHEMA_MARKERS`).
**Do not bump the schema version without first updating that parser** — the
test will warn but otherwise pass against an unsupported version, which is
exactly the kind of silent regression this test exists to prevent.

## Refresh policy

The fixture is regenerated on every test run from a fixed PRNG seed (`0`).
There is no committed binary — refreshing means editing
`generator.py`. Things that warrant a refresh:

- The dbt manifest schema bumps and the parser is updated to support a new
  version. Update `SCHEMA_VERSION_URL`.
- The enrichment writers grow new behaviour worth covering at scale. Add the
  shape to the generator and a corresponding assertion.
- The test starts flaking on the performance budget on real CI hardware.
  Tune the generator knobs (`TARGET_MODEL_COUNT`, fan-out, etc.) before
  relaxing the budget in the test.

When you do refresh, run the e2e test once locally with `PRETENSOR_E2E=1` to
confirm the assertions still hold and the perf budget is comfortably met:

```bash
PRETENSOR_E2E=1 uv run pytest \
    tests/e2e/test_dbt_warehouse_scale_enrichment.py -v
```

## Reference data exposed to the test

`write_synthetic_manifest()` returns a `SyntheticManifestSpec` recording every
quantity the test asserts on:

- `physical_tables` — `(schema, physical_name)` pairs to materialize as DDL.
- `expected_lineage_edges` — count of resolvable `parent_map` edges (excludes
  injected dangling refs).
- `documented_resolvable_models` — `(schema, physical_name)` pairs whose
  manifest model carries a description.
- `known_exposures` — hand-picked subset of exposures with their expected
  flagged-table sets.

These are computed at generation time so the test does not have to re-derive
them and so a refresh of the generator automatically updates the reference
counts.
