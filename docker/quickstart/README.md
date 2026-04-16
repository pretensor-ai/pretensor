# Pretensor quickstart database

A throwaway Postgres instance pre-loaded with the Pagila sample schema (DVD
rental store), sized for `pretensor quickstart`. Listens on **localhost:55432**.

## Use it

The recommended path is `pretensor quickstart`, which starts the container,
indexes the DB into `./.pretensor/`, and prints an MCP config snippet.

To run the container directly:

```bash
docker compose -f docker/quickstart/docker-compose.yml up -d
# DSN: postgresql://postgres:postgres@localhost:55432/pagila
docker compose -f docker/quickstart/docker-compose.yml down -v
```

## Contents

- 8 tables across `public` and `staff` schemas
- ~35 seed rows total — enough for `query`, `context`, and `traverse` to
  return non-empty results
- FK chain: `payment → rental → inventory → film` plus `rental → customer`

The seed lives at `tests/e2e/fixtures/sql/pagila_data.sql` and is a curated
subset of upstream Pagila.
