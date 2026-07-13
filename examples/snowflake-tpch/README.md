# Snowflake TPCH example

Every Snowflake account — trial or paid — includes the `SNOWFLAKE_SAMPLE_DATA` database with TPC-H schemas at multiple scale factors (`TPCH_SF1`, `TPCH_SF10`, `TPCH_SF100`, `TPCH_SF1000`). This example indexes the smallest one (`TPCH_SF1`) and connects it to your IDE via Pretensor's MCP server.

## Prerequisites

- Python 3.11 or 3.12
- A Snowflake account (a [free trial](https://signup.snowflake.com/) works)
- Pretensor with the Snowflake connector: `pip install 'pretensor[snowflake]'`

## What you need from Snowflake

| Item | Where to find it |
|------|-----------------|
| Account identifier | Snowflake UI → bottom-left account menu → copy the full identifier (e.g. `xy12345.us-east-1.aws`) |
| Username | Your Snowflake login username |
| Password | Your Snowflake login password |
| Warehouse | Any active warehouse — `COMPUTE_WH` is the default on new accounts |

## Index the TPCH sample data

```bash
pretensor index \
  'snowflake://USER:PASS@ACCOUNT/SNOWFLAKE_SAMPLE_DATA/TPCH_SF1?warehouse=COMPUTE_WH' \
  --name tpch-sf1
```

Replace `USER`, `PASS`, `ACCOUNT`, and `COMPUTE_WH` with your actual values. The schema and database names (`SNOWFLAKE_SAMPLE_DATA`, `TPCH_SF1`) are fixed — they exist on every Snowflake account.

> **Larger scale factors:** Substitute `TPCH_SF10` or `TPCH_SF100` in the DSN for a more realistic dataset. Indexing takes longer but the MCP tools still run in milliseconds — the graph is built offline.

## Get the MCP config snippet

```bash
pretensor serve --config-only
```

## Connect to your IDE

Paste the output into your IDE's MCP config:

- [Claude Code](../claude-code/)
- [Cursor](../cursor/)
- [Windsurf](../windsurf/)

## Sample prompts

The TPC-H schema models a supply chain: orders from customers, fulfilled via line items, sourced from suppliers and parts.

```
What tables are in the tpch-sf1 database?
```

```
Show me how ORDERS connects to LINEITEM, PART, and SUPPLIER.
```

```
What is the join path from CUSTOMER to LINEITEM?
```

```
Which tables are downstream of PART via foreign key edges?
```

## DSN reference

The Snowflake DSN format used by Pretensor:

```
snowflake://USER:PASS@ACCOUNT/DATABASE/SCHEMA?warehouse=WAREHOUSE&role=ROLE
```

| Component | Required | Example |
|-----------|----------|---------|
| `USER` | yes | `alice` |
| `PASS` | yes | URL-encode special characters |
| `ACCOUNT` | yes | `xy12345.us-east-1.aws` |
| `DATABASE` | yes | `SNOWFLAKE_SAMPLE_DATA` |
| `SCHEMA` | no | `TPCH_SF1` (filters to one schema) |
| `warehouse` | recommended | `COMPUTE_WH` |
| `role` | no | `SYSADMIN` |

Omitting `SCHEMA` indexes all schemas in the database. For large warehouses that is usually too broad — scope to a specific schema to keep indexing fast.

## Key-pair authentication

Instead of a password, you can authenticate with an RSA private key. Add `private_key_path` (and, for encrypted keys, `private_key_passphrase`) to `sources.secrets.yaml`:

```yaml
# .pretensor/config.yaml
sources:
  tpch:
    dialect: snowflake
    account: xy12345.us-east-1.aws
    user: alice
    database: SNOWFLAKE_SAMPLE_DATA
    schema: TPCH_SF1
    warehouse: COMPUTE_WH
```

```yaml
# .pretensor/sources.secrets.yaml  (keep out of version control)
tpch:
  private_key_path: /home/alice/.snowflake/rsa_key.p8
  private_key_passphrase: my_passphrase   # omit if key is unencrypted
```

Then index using the source name instead of a DSN:

```bash
pretensor index --source tpch
```

**Notes:**

- `private_key_path` takes precedence over `password` — if both are present, the key is used.
- The path supports `~` expansion and must be absolute or resolvable from the current working directory. Use `${HOME}` for portability across environments, e.g. `${HOME}/.snowflake/rsa_key.p8`.
- The key file must be in PEM format (`.p8` / `.pem`). Generate one with:

  ```bash
  openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_key.p8 -nocrypt
  ```

  For an encrypted key omit `-nocrypt` and supply the passphrase as `private_key_passphrase`.
- Errors for a missing, unreadable, or malformed key file are surfaced as connection errors with an actionable message.
