# Pretensor quickstart database

A throwaway Postgres instance pre-loaded with the Pagila sample schema (DVD
rental store), sized for `pretensor quickstart`. Listens on **localhost:55432** (bound to 127.0.0.1 only; the default
`postgres:postgres` credentials are for local use and must never be exposed
to the network).

## Use it

The recommended path is `pretensor quickstart`, which starts the container,
indexes the DB into `./.pretensor/`, and prints an MCP config snippet.

To run the container directly from a clone of the repo:

```bash
docker compose -f src/pretensor/quickstart/docker-compose.yml up -d
# DSN: postgresql://postgres:postgres@localhost:55432/pagila
docker compose -f src/pretensor/quickstart/docker-compose.yml down -v
```

## Contents

- 8 tables across `public` and `staff` schemas
- ~35 seed rows total — enough for `query`, `context`, and `traverse` to
  return non-empty results
- FK chain: `payment → rental → inventory → film` plus `rental → customer`

The schema and seed live next to this README as `pagila_ddl.sql` and
`pagila_data.sql` — a curated subset of upstream Pagila. They ship inside
the wheel so `pretensor quickstart` works after a plain `pip install`
without needing a clone of the repo.
