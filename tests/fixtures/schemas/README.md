# Benchmark schema fixtures

Each YAML here is a `SchemaSnapshot` produced by
`pretensor.connectors.inspect.inspect()` running against a live Postgres
instance loaded with that dataset's DDL. Fixtures are consumed by:

- `tests/conftest.py::load_schema` (pytest fixture — `load_schema("pagila")`)
- `pretensor.benchmark.fixtures.load_dataset()` (benchmark CLI entry point)
- the L1/L2 metric modules landing in follow-up work

All snapshots have `introspected_at: 2024-01-01T00:00:00Z` normalized so
re-running the generator produces byte-identical diffs outside of genuine
schema changes.

## Fixture catalog

| Dataset             | Schemas                                         | Tables + views | Size  | DDL dump (`scripts/data/<name>/schema.sql`) | NL-to-SQL questions (`scripts/data/<name>_nl2sql_bench.json`) |
|---------------------|-------------------------------------------------|----------------|-------|---------------------------------------------|----------------------------------------------------------------|
| `pagila`            | `public`                                        | 13             | 14 KB | yes                                         | 10                                                             |
| `adventureworks`    | `person`, `humanresources`, `production`, `purchasing`, `sales` | 86 | 470 KB | yes                                         | 22                                                             |
| `tpch`              | `public`                                        | 8              | 16 KB | yes                                         | 22                                                             |
| `analytics_dwh`     | synthetic                                       | —              | 12 KB | no (hand-crafted for clustering tests)      | no                                                             |
| `saas_multitenant`  | synthetic                                       | —              | 12 KB | no (hand-crafted for tenancy tests)         | no                                                             |
| `adversarial`       | synthetic                                       | —              | 7 KB  | no (edge cases for entity resolution)       | no                                                             |

## Provenance + license per dataset

### `pagila`

- **Upstream:** <https://github.com/devrimgunduz/pagila> (`pagila-schema.sql`).
- **License:** BSD-3-Clause (see the upstream `LICENSE.TXT`). Compatible with
  this repo's MIT license.
- **Shape:** public-domain DVD-rental schema derived from MySQL Sakila.
- **Regenerate:**
  ```bash
  uv run --with pgserver --with psycopg2-binary python scripts/generate_schema_snapshot.py \
      --ddl scripts/data/pagila/schema.sql \
      --name pagila \
      --out tests/fixtures/schemas/pagila.yaml
  ```

### `adventureworks`

- **Upstream:** <https://github.com/lorint/AdventureWorks-for-Postgres>
  (`install.sql`). That repo is a PostgreSQL port of Microsoft's
  AdventureWorks 2014 OLTP sample.
- **License:** MIT (on the Postgres port). The original AdventureWorks sample
  is published by Microsoft under a permissive sample-code license. The
  Postgres port is redistributable. Compatible with this repo's MIT license.
- **Redistribution verdict:** OK. Both the DDL dump and the generated
  `SchemaSnapshot` YAML are derivative of MIT-licensed DDL.
- **Shape adaptations** (applied when producing `scripts/data/adventureworks/schema.sql`
  from upstream `install.sql`):
  - `\copy` / `\pset` / `\dt` psql meta-commands stripped.
  - Inline `INSERT INTO Production.ProductReview ...` removed (hand-coded
    sample data, not DDL).
  - Türkiye `UPDATE person.countryregion ...` data-fix line removed.
  - `CREATE EXTENSION "uuid-ossp"` and `CREATE EXTENSION tablefunc` lines
    stripped — `pgserver`'s bundled Postgres ships without contrib. The DDL
    remains loadable on any contrib-free Postgres 13+.
  - `uuid_generate_v1()` default-value calls replaced with the core
    `gen_random_uuid()` (Postgres 13+).
  - `CREATE VIEW Sales.vSalesPersonSalesByFiscalYears` stripped — requires
    `crosstab()` from the tablefunc contrib extension.
  - Five "convenience" alias schemas (`pe`, `hr`, `pr`, `pu`, `sa`) dropped
    — they re-expose the five business schemas under short names and add no
    schema knowledge that the benchmark cares about.
- **Regenerate:**
  ```bash
  uv run --with pgserver --with psycopg2-binary python scripts/generate_schema_snapshot.py \
      --ddl scripts/data/adventureworks/schema.sql \
      --name adventureworks \
      --out tests/fixtures/schemas/adventureworks.yaml
  ```

### `tpch`

- **Upstream:** TPC's TPC-H 3.0.1 reference kit (`dss.ddl` + `dss.ri`).
  Publicly mirrored at <https://github.com/electrum/tpch-dbgen>.
- **License:** TPC EULA. §2.3 permits redistribution of the schema DDL in
  tooling. The EULA restriction applies to publication of *benchmark
  results*, not to the schema itself. Compatible with this repo's MIT
  license for the committed DDL + derived YAML.
- **Shape:** 8 tables (region, nation, part, supplier, partsupp, customer,
  orders, lineitem) plus canonical PK/FK constraints. No generated rows are
  committed — spin up TPC's `dbgen` locally to populate data at runtime.
- **DDL adaptations** (applied in `scripts/data/tpch/schema.sql`):
  - Translated to Postgres-native syntax: `ADD CONSTRAINT name FOREIGN KEY`
    (the upstream `dss.ri` uses DB2 syntax that Postgres does not accept).
  - Identifiers written in lowercase; Postgres folds unquoted identifiers
    to lowercase anyway, so this matches the existing `tpch.yaml`.
- **Regenerate:**
  ```bash
  uv run --with pgserver --with psycopg2-binary python scripts/generate_schema_snapshot.py \
      --ddl scripts/data/tpch/schema.sql \
      --name tpch \
      --out tests/fixtures/schemas/tpch.yaml
  ```

### `analytics_dwh` / `saas_multitenant` / `adversarial`

Synthetic, hand-crafted snapshots used by specific test suites (clustering,
tenancy, entity-resolution). Not part of the benchmark dataset bundle; they
have no DDL dump and no NL-to-SQL question set. Their checked-in YAMLs are
the source of truth.

## How to add a new dataset

1. Commit `scripts/data/<name>/schema.sql` — pure DDL, no `INSERT`, no data.
2. Commit `scripts/data/<name>_nl2sql_bench.json` — list of objects with keys
   `{id, question, expected_sql}` (≥ 20 questions for new bundle entries).
3. Add `NAME = "<name>"` to `pretensor.benchmark.runner.Dataset`.
4. Run the generator above; commit the resulting YAML here.
5. Add the row to the **Fixture catalog** table above.
6. Confirm license compatibility and capture the verdict in this README.
