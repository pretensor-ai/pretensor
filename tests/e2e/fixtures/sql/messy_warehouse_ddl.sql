-- Messy Warehouse benchmark fixture — DDL only, no data.
--
-- Exhibits real-world warehouse warts for pretensor intelligence-layer testing:
--   * 4-layer dbt-style hierarchy: raw → staging → intermediate → mart
--   * Near-duplicate entity cluster: mart.users / mart.user_accounts / mart.customer_master
--   * 4 audit/log tables (user_audit_log, order_audit_log, login_log, schema_change_log)
--   * SCD2 snapshot: mart.users_snapshot (valid_from / valid_to / is_current)
--   * Cross-schema FK: raw.raw_events.user_id → mart.users.user_id
--   * Mixed naming convention: camelCase columns in raw/mart alongside snake_case
--   * Junk schema with tmp_* and _legacy_* tables for role-classification noise
--
-- No INSERT statements — pretensor reads pg_catalog metadata only.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS intermediate;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS junk;

-- ---------------------------------------------------------------------------
-- raw schema
-- ---------------------------------------------------------------------------

CREATE TABLE raw.raw_users (
    id            INTEGER      NOT NULL,
    "externalId"  VARCHAR(255),
    email         VARCHAR(255),
    created_ts    TIMESTAMP,
    _src_file     VARCHAR(512),
    _loaded_at    TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE raw.raw_orders (
    "orderId"    INTEGER      NOT NULL,
    "userId"     INTEGER,
    total        DECIMAL(15,2),
    status       VARCHAR(50),
    created_ts   TIMESTAMP,
    PRIMARY KEY ("orderId"),
    FOREIGN KEY ("userId") REFERENCES raw.raw_users (id)
);

-- raw_events.user_id has a cross-schema FK to mart.users.user_id.
-- Declared after mart.users is created (see end of file).
CREATE TABLE raw.raw_events (
    event_id     INTEGER      NOT NULL,
    user_id      INTEGER,
    event_type   VARCHAR(100),
    properties   JSONB,
    occurred_at  TIMESTAMP,
    PRIMARY KEY (event_id)
);

CREATE TABLE raw.raw_products (
    product_id  INTEGER      NOT NULL,
    sku         VARCHAR(100),
    name        VARCHAR(255),
    category    VARCHAR(100),
    cost        DECIMAL(15,2),
    PRIMARY KEY (product_id)
);

-- ---------------------------------------------------------------------------
-- staging schema
-- ---------------------------------------------------------------------------

CREATE TABLE staging.stg_users (
    user_id    INTEGER      NOT NULL,
    source_id  INTEGER,
    email      VARCHAR(255),
    first_name VARCHAR(100),
    last_name  VARCHAR(100),
    created_at TIMESTAMP,
    PRIMARY KEY (user_id),
    FOREIGN KEY (source_id) REFERENCES raw.raw_users (id)
);

CREATE TABLE staging.stg_orders (
    order_id     INTEGER      NOT NULL,
    user_id      INTEGER,
    order_total  DECIMAL(15,2),
    order_status VARCHAR(50),
    created_at   TIMESTAMP,
    PRIMARY KEY (order_id),
    FOREIGN KEY (user_id) REFERENCES staging.stg_users (user_id)
);

CREATE TABLE staging.stg_products (
    product_id        INTEGER      NOT NULL,
    source_product_id INTEGER,
    product_name      VARCHAR(255),
    category          VARCHAR(100),
    unit_cost         DECIMAL(15,2),
    PRIMARY KEY (product_id),
    FOREIGN KEY (source_product_id) REFERENCES raw.raw_products (product_id)
);

CREATE TABLE staging.stg_events (
    event_id         INTEGER      NOT NULL,
    user_id          INTEGER,
    event_type       VARCHAR(100),
    event_properties JSONB,
    occurred_at      TIMESTAMP,
    PRIMARY KEY (event_id),
    FOREIGN KEY (user_id) REFERENCES staging.stg_users (user_id)
);

-- ---------------------------------------------------------------------------
-- intermediate schema
-- ---------------------------------------------------------------------------

CREATE TABLE intermediate.int_user_orders (
    int_id       INTEGER      NOT NULL,
    user_id      INTEGER,
    order_id     INTEGER,
    user_email   VARCHAR(255),
    order_total  DECIMAL(15,2),
    order_status VARCHAR(50),
    order_date   DATE,
    PRIMARY KEY (int_id),
    FOREIGN KEY (user_id)  REFERENCES staging.stg_users (user_id),
    FOREIGN KEY (order_id) REFERENCES staging.stg_orders (order_id)
);

CREATE TABLE intermediate.int_order_items (
    int_id     INTEGER      NOT NULL,
    order_id   INTEGER,
    product_id INTEGER,
    quantity   INTEGER,
    unit_price DECIMAL(15,2),
    line_total DECIMAL(15,2),
    PRIMARY KEY (int_id),
    FOREIGN KEY (order_id)   REFERENCES staging.stg_orders (order_id),
    FOREIGN KEY (product_id) REFERENCES staging.stg_products (product_id)
);

-- ---------------------------------------------------------------------------
-- mart schema
-- ---------------------------------------------------------------------------

CREATE TABLE mart.users (
    user_id    INTEGER      NOT NULL,
    email      VARCHAR(255),
    first_name VARCHAR(100),
    last_name  VARCHAR(100),
    status     VARCHAR(50),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (user_id)
);

-- Near-duplicate of mart.users — same business entity, camelCase naming
CREATE TABLE mart.user_accounts (
    account_id      INTEGER      NOT NULL,
    "accountEmail"  VARCHAR(255),
    "firstName"     VARCHAR(100),
    "lastName"      VARCHAR(100),
    "accountStatus" VARCHAR(50),
    "createdDate"   TIMESTAMP,
    PRIMARY KEY (account_id)
);

-- Near-duplicate of mart.users — legacy source system surface
CREATE TABLE mart.customer_master (
    customer_id  INTEGER      NOT NULL,
    cust_email   VARCHAR(255),
    full_name    VARCHAR(255),
    cust_status  VARCHAR(50),
    created_dt   DATE,
    source_system VARCHAR(100),
    PRIMARY KEY (customer_id)
);

CREATE TABLE mart.products (
    product_id   INTEGER      NOT NULL,
    product_name VARCHAR(255),
    category     VARCHAR(100),
    subcategory  VARCHAR(100),
    unit_price   DECIMAL(15,2),
    PRIMARY KEY (product_id)
);

CREATE TABLE mart.orders (
    order_id        INTEGER      NOT NULL,
    user_id         INTEGER,
    product_id      INTEGER,
    order_date      DATE,
    total_amount    DECIMAL(15,2),
    discount_amount DECIMAL(15,2),
    tax_amount      DECIMAL(15,2),
    order_status    VARCHAR(50),
    PRIMARY KEY (order_id),
    FOREIGN KEY (user_id)    REFERENCES mart.users (user_id),
    FOREIGN KEY (product_id) REFERENCES mart.products (product_id)
);

-- SCD2 snapshot of mart.users
CREATE TABLE mart.users_snapshot (
    snapshot_id   INTEGER      NOT NULL,
    surrogate_key BIGINT       NOT NULL,
    user_id       INTEGER,
    email         VARCHAR(255),
    first_name    VARCHAR(100),
    last_name     VARCHAR(100),
    status        VARCHAR(50),
    valid_from    TIMESTAMP    NOT NULL,
    valid_to      TIMESTAMP,
    is_current    BOOLEAN      NOT NULL,
    PRIMARY KEY (snapshot_id),
    FOREIGN KEY (user_id) REFERENCES mart.users (user_id)
);

CREATE TABLE mart.user_audit_log (
    log_id     INTEGER   NOT NULL,
    user_id    INTEGER,
    action     VARCHAR(100),
    changed_by VARCHAR(255),
    changed_at TIMESTAMP,
    old_values JSONB,
    new_values JSONB,
    PRIMARY KEY (log_id),
    FOREIGN KEY (user_id) REFERENCES mart.users (user_id)
);

CREATE TABLE mart.order_audit_log (
    log_id     INTEGER   NOT NULL,
    order_id   INTEGER,
    action     VARCHAR(100),
    changed_by VARCHAR(255),
    changed_at TIMESTAMP,
    old_values JSONB,
    new_values JSONB,
    PRIMARY KEY (log_id),
    FOREIGN KEY (order_id) REFERENCES mart.orders (order_id)
);

CREATE TABLE mart.login_log (
    log_id     INTEGER   NOT NULL,
    user_id    INTEGER,
    ip_address VARCHAR(45),
    user_agent TEXT,
    success    BOOLEAN,
    logged_at  TIMESTAMP,
    PRIMARY KEY (log_id),
    FOREIGN KEY (user_id) REFERENCES mart.users (user_id)
);

-- DDL audit trail — no FK (schema-agnostic)
CREATE TABLE mart.schema_change_log (
    log_id        INTEGER   NOT NULL,
    table_name    VARCHAR(255),
    action        VARCHAR(100),
    changed_by    VARCHAR(255),
    changed_at    TIMESTAMP,
    ddl_statement TEXT,
    PRIMARY KEY (log_id)
);

-- Cross-schema FK: raw.raw_events → mart.users (declared after mart.users exists)
ALTER TABLE raw.raw_events
    ADD CONSTRAINT fk_raw_events_user_id
    FOREIGN KEY (user_id) REFERENCES mart.users (user_id);

-- ---------------------------------------------------------------------------
-- junk schema
-- ---------------------------------------------------------------------------

CREATE TABLE junk.tmp_user_import (
    id        INTEGER  NOT NULL,
    raw_line  TEXT,
    status    VARCHAR(50),
    loaded_at TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE junk.tmp_order_backfill (
    id              INTEGER  NOT NULL,
    order_id        INTEGER,
    backfill_status VARCHAR(50),
    created_at      TIMESTAMP,
    PRIMARY KEY (id)
);

CREATE TABLE junk._legacy_customers (
    cid         INTEGER      NOT NULL,
    name        VARCHAR(255),
    email       VARCHAR(255),
    migrated_at TIMESTAMP,
    PRIMARY KEY (cid)
);

CREATE TABLE junk._legacy_order_items (
    item_id     INTEGER      NOT NULL,
    order_ref   INTEGER,
    product_ref INTEGER,
    qty         INTEGER,
    price       DECIMAL(15,2),
    PRIMARY KEY (item_id)
);
