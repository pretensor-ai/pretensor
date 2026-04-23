-- TPC-H schema — DDL only (tables + primary keys + foreign keys).
-- Source: Transaction Processing Performance Council, "TPC-H" reference kit
--   (dss.ddl + dss.ri). Mirrored at https://github.com/electrum/tpch-dbgen.
-- The TPC EULA permits redistribution of the schema DDL; it restricts only
-- the publication of benchmark *results*. No generated rows or dbgen output
-- is checked in; use TPC's dbgen locally to populate data at runtime.
--
-- Postgres-native syntax: FK clauses below use the SQL-standard ADD
-- CONSTRAINT form (TPC's reference dss.ri uses DB2 syntax which Postgres
-- does not accept). Column names are identical to the reference spec.
--
-- To regenerate tests/fixtures/schemas/tpch.yaml from this file:
--   uv run --with pgserver python scripts/generate_schema_snapshot.py \
--     --ddl scripts/data/tpch/schema.sql \
--     --name tpch \
--     --out tests/fixtures/schemas/tpch.yaml

CREATE TABLE region (
    r_regionkey  INTEGER NOT NULL,
    r_name       CHAR(25) NOT NULL,
    r_comment    VARCHAR(152)
);

CREATE TABLE nation (
    n_nationkey  INTEGER NOT NULL,
    n_name       CHAR(25) NOT NULL,
    n_regionkey  INTEGER NOT NULL,
    n_comment    VARCHAR(152)
);

CREATE TABLE part (
    p_partkey     INTEGER NOT NULL,
    p_name        VARCHAR(55) NOT NULL,
    p_mfgr        CHAR(25) NOT NULL,
    p_brand       CHAR(10) NOT NULL,
    p_type        VARCHAR(25) NOT NULL,
    p_size        INTEGER NOT NULL,
    p_container   CHAR(10) NOT NULL,
    p_retailprice DECIMAL(15,2) NOT NULL,
    p_comment     VARCHAR(23) NOT NULL
);

CREATE TABLE supplier (
    s_suppkey     INTEGER NOT NULL,
    s_name        CHAR(25) NOT NULL,
    s_address     VARCHAR(40) NOT NULL,
    s_nationkey   INTEGER NOT NULL,
    s_phone       CHAR(15) NOT NULL,
    s_acctbal     DECIMAL(15,2) NOT NULL,
    s_comment     VARCHAR(101) NOT NULL
);

CREATE TABLE partsupp (
    ps_partkey     INTEGER NOT NULL,
    ps_suppkey     INTEGER NOT NULL,
    ps_availqty    INTEGER NOT NULL,
    ps_supplycost  DECIMAL(15,2)  NOT NULL,
    ps_comment     VARCHAR(199) NOT NULL
);

CREATE TABLE customer (
    c_custkey     INTEGER NOT NULL,
    c_name        VARCHAR(25) NOT NULL,
    c_address     VARCHAR(40) NOT NULL,
    c_nationkey   INTEGER NOT NULL,
    c_phone       CHAR(15) NOT NULL,
    c_acctbal     DECIMAL(15,2)   NOT NULL,
    c_mktsegment  CHAR(10) NOT NULL,
    c_comment     VARCHAR(117) NOT NULL
);

CREATE TABLE orders (
    o_orderkey       INTEGER NOT NULL,
    o_custkey        INTEGER NOT NULL,
    o_orderstatus    CHAR(1) NOT NULL,
    o_totalprice     DECIMAL(15,2) NOT NULL,
    o_orderdate      DATE NOT NULL,
    o_orderpriority  CHAR(15) NOT NULL,
    o_clerk          CHAR(15) NOT NULL,
    o_shippriority   INTEGER NOT NULL,
    o_comment        VARCHAR(79) NOT NULL
);

CREATE TABLE lineitem (
    l_orderkey       INTEGER NOT NULL,
    l_partkey        INTEGER NOT NULL,
    l_suppkey        INTEGER NOT NULL,
    l_linenumber     INTEGER NOT NULL,
    l_quantity       DECIMAL(15,2) NOT NULL,
    l_extendedprice  DECIMAL(15,2) NOT NULL,
    l_discount       DECIMAL(15,2) NOT NULL,
    l_tax            DECIMAL(15,2) NOT NULL,
    l_returnflag     CHAR(1) NOT NULL,
    l_linestatus     CHAR(1) NOT NULL,
    l_shipdate       DATE NOT NULL,
    l_commitdate     DATE NOT NULL,
    l_receiptdate    DATE NOT NULL,
    l_shipinstruct   CHAR(25) NOT NULL,
    l_shipmode       CHAR(10) NOT NULL,
    l_comment        VARCHAR(44) NOT NULL
);

-- Primary keys
ALTER TABLE region   ADD CONSTRAINT pk_region   PRIMARY KEY (r_regionkey);
ALTER TABLE nation   ADD CONSTRAINT pk_nation   PRIMARY KEY (n_nationkey);
ALTER TABLE part     ADD CONSTRAINT pk_part     PRIMARY KEY (p_partkey);
ALTER TABLE supplier ADD CONSTRAINT pk_supplier PRIMARY KEY (s_suppkey);
ALTER TABLE partsupp ADD CONSTRAINT pk_partsupp PRIMARY KEY (ps_partkey, ps_suppkey);
ALTER TABLE customer ADD CONSTRAINT pk_customer PRIMARY KEY (c_custkey);
ALTER TABLE orders   ADD CONSTRAINT pk_orders   PRIMARY KEY (o_orderkey);
ALTER TABLE lineitem ADD CONSTRAINT pk_lineitem PRIMARY KEY (l_orderkey, l_linenumber);

-- Foreign keys
ALTER TABLE nation   ADD CONSTRAINT fk_nation_region     FOREIGN KEY (n_regionkey) REFERENCES region   (r_regionkey);
ALTER TABLE supplier ADD CONSTRAINT fk_supplier_nation   FOREIGN KEY (s_nationkey) REFERENCES nation   (n_nationkey);
ALTER TABLE customer ADD CONSTRAINT fk_customer_nation   FOREIGN KEY (c_nationkey) REFERENCES nation   (n_nationkey);
ALTER TABLE partsupp ADD CONSTRAINT fk_partsupp_part     FOREIGN KEY (ps_partkey)  REFERENCES part     (p_partkey);
ALTER TABLE partsupp ADD CONSTRAINT fk_partsupp_supplier FOREIGN KEY (ps_suppkey)  REFERENCES supplier (s_suppkey);
ALTER TABLE orders   ADD CONSTRAINT fk_orders_customer   FOREIGN KEY (o_custkey)   REFERENCES customer (c_custkey);
ALTER TABLE lineitem ADD CONSTRAINT fk_lineitem_orders   FOREIGN KEY (l_orderkey)  REFERENCES orders   (o_orderkey);
ALTER TABLE lineitem ADD CONSTRAINT fk_lineitem_partsupp FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey);
