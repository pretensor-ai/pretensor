-- TPC-DS-shaped snowflake schema for Pretensor e2e testing
--
-- Modeled on the TPC-DS v3.2.0 specification (Transaction Processing Performance
-- Council, Decision Support benchmark). The official schema generator is
-- `dsdgen` from gregrahn/tpcds-kit; this file is hand-crafted from the spec,
-- NOT a verbatim vendor dump.
--
-- Reasons for hand-crafting (same rationale as the AdventureWorks fixture):
--   1. The official `dsdgen` DDL uses dialect-specific syntax and data-loading
--      scripts that do not work via psycopg2's execute().
--   2. Pretensor's graph builder reads pg_catalog metadata only; row data is
--      irrelevant — only table/column/FK structure matters.
--   3. Hand-crafting lets us preserve the exact FK chains needed by the
--      acceptance tests while keeping the DDL compact.
--
-- What is preserved from the TPC-DS specification:
--   * 24 tables: 7 fact tables and 17 dimension tables
--   * Snowflaked dimensions: item -> item_brand -> item_category
--   * Multiple valid FK paths between the same table pairs:
--       - store_sales -> customer: direct via ss_customer_sk (1 hop)
--       - store_sales <- store_returns -> customer: indirect via
--         sr_ticket_number/sr_item_sk composite FK back to store_sales,
--         then sr_customer_sk -> customer (2 hops undirected)
--   * Every fact table has *_date_sk and *_time_sk audit-style columns
--     referencing date_dim / time_dim — the "date_sk everywhere" pattern
--     that must NOT be preferred over real FK chains by the traverse ranker.
--
-- Table inventory (24):
--   Dimensions (17): date_dim, time_dim, item_category, item_brand, item,
--     customer_address, income_band, household_demographics,
--     customer_demographics, customer, promotion, reason, store, warehouse,
--     web_page, web_site, call_center
--   Facts (7): store_sales, store_returns, catalog_sales, catalog_returns,
--     web_sales, web_returns, inventory
--
-- Re-vendoring: manual decision. Pin a TPC-DS spec version above and
-- reconcile table/FK shape by hand. Do not automate.

-- ============================================================================
-- Dimension tables (created first — referenced by fact tables)
-- ============================================================================

-- date_dim: shared calendar dimension referenced by every fact table
CREATE TABLE date_dim (
    d_date_sk           INT          PRIMARY KEY,
    d_date_id           VARCHAR(16)  NOT NULL,
    d_date              DATE,
    d_month_seq         INT,
    d_week_seq          INT,
    d_quarter_seq       INT,
    d_year              INT,
    d_dow               INT,
    d_moy               INT,
    d_dom               INT,
    d_qoy               INT,
    d_fy_year           INT,
    d_fy_quarter_seq    INT,
    d_fy_week_seq       INT,
    d_day_name          VARCHAR(9),
    d_quarter_name      VARCHAR(6),
    d_holiday           CHAR(1),
    d_weekend           CHAR(1),
    d_following_holiday  CHAR(1),
    d_first_dom         INT,
    d_last_dom          INT,
    d_same_day_ly       INT,
    d_same_day_lq       INT,
    d_current_day       CHAR(1),
    d_current_week      CHAR(1),
    d_current_month     CHAR(1),
    d_current_quarter   CHAR(1),
    d_current_year      CHAR(1)
);

-- time_dim: intraday time dimension
CREATE TABLE time_dim (
    t_time_sk           INT          PRIMARY KEY,
    t_time_id           VARCHAR(16)  NOT NULL,
    t_time              INT,
    t_hour              INT,
    t_minute            INT,
    t_second            INT,
    t_meal_time         CHAR(1),
    t_sub_shift         VARCHAR(20),
    t_shift             VARCHAR(20),
    t_am_pm             CHAR(2)
);

-- item_category: top of the snowflaked item hierarchy
CREATE TABLE item_category (
    ic_category_sk      SERIAL       PRIMARY KEY,
    ic_category_id      VARCHAR(16)  NOT NULL,
    ic_category_name    VARCHAR(50)
);

-- item_brand: middle of the snowflaked item hierarchy
CREATE TABLE item_brand (
    ib_brand_sk         SERIAL       PRIMARY KEY,
    ib_brand_id         VARCHAR(16)  NOT NULL,
    ib_brand_name       VARCHAR(50),
    ib_category_sk      INT          REFERENCES item_category(ic_category_sk)
);

-- item: product dimension — references item_brand (snowflaked)
CREATE TABLE item (
    i_item_sk           INT          PRIMARY KEY,
    i_item_id           VARCHAR(16)  NOT NULL,
    i_rec_start_date    DATE,
    i_rec_end_date      DATE,
    i_item_desc         VARCHAR(200),
    i_current_price     NUMERIC(7,2),
    i_wholesale_cost    NUMERIC(7,2),
    i_brand_sk          INT          REFERENCES item_brand(ib_brand_sk),
    i_class_id          INT,
    i_class             VARCHAR(50),
    i_manufact_id       INT,
    i_manufact          VARCHAR(50),
    i_size              VARCHAR(20),
    i_formulation       VARCHAR(20),
    i_color             VARCHAR(20),
    i_units             VARCHAR(10),
    i_container         VARCHAR(10),
    i_manager_id        INT,
    i_product_name      VARCHAR(50)
);

-- customer_address: shared address dimension
CREATE TABLE customer_address (
    ca_address_sk       INT          PRIMARY KEY,
    ca_address_id       VARCHAR(16)  NOT NULL,
    ca_street_number    VARCHAR(10),
    ca_street_name      VARCHAR(60),
    ca_street_type      VARCHAR(15),
    ca_suite_number     VARCHAR(10),
    ca_city             VARCHAR(60),
    ca_county           VARCHAR(30),
    ca_state            CHAR(2),
    ca_zip              VARCHAR(10),
    ca_country          VARCHAR(20),
    ca_gmt_offset       NUMERIC(5,2),
    ca_location_type    VARCHAR(20)
);

-- income_band: demographic band
CREATE TABLE income_band (
    ib_income_band_sk   INT          PRIMARY KEY,
    ib_lower_bound      INT,
    ib_upper_bound      INT
);

-- household_demographics: references income_band
CREATE TABLE household_demographics (
    hd_demo_sk          INT          PRIMARY KEY,
    hd_income_band_sk   INT          REFERENCES income_band(ib_income_band_sk),
    hd_buy_potential    VARCHAR(15),
    hd_dep_count        INT,
    hd_vehicle_count    INT
);

-- customer_demographics: standalone demographic dimension
CREATE TABLE customer_demographics (
    cd_demo_sk          INT          PRIMARY KEY,
    cd_gender           CHAR(1),
    cd_marital_status   CHAR(1),
    cd_education_status VARCHAR(20),
    cd_purchase_estimate INT,
    cd_credit_rating    VARCHAR(10),
    cd_dep_count        INT,
    cd_dep_employed_count INT,
    cd_dep_college_count INT
);

-- customer: central customer dimension (snowflaked via address, demographics)
CREATE TABLE customer (
    c_customer_sk       INT          PRIMARY KEY,
    c_customer_id       VARCHAR(16)  NOT NULL,
    c_current_cdemo_sk  INT          REFERENCES customer_demographics(cd_demo_sk),
    c_current_hdemo_sk  INT          REFERENCES household_demographics(hd_demo_sk),
    c_current_addr_sk   INT          REFERENCES customer_address(ca_address_sk),
    c_first_shipto_date_sk INT       REFERENCES date_dim(d_date_sk),
    c_first_sales_date_sk  INT       REFERENCES date_dim(d_date_sk),
    c_salutation        VARCHAR(10),
    c_first_name        VARCHAR(20),
    c_last_name         VARCHAR(30),
    c_preferred_cust_flag CHAR(1),
    c_birth_day         INT,
    c_birth_month       INT,
    c_birth_year        INT,
    c_birth_country     VARCHAR(20),
    c_login             VARCHAR(13),
    c_email_address     VARCHAR(50),
    c_last_review_date_sk INT        REFERENCES date_dim(d_date_sk)
);

-- promotion: marketing promotion dimension
CREATE TABLE promotion (
    p_promo_sk          INT          PRIMARY KEY,
    p_promo_id          VARCHAR(16)  NOT NULL,
    p_start_date_sk     INT          REFERENCES date_dim(d_date_sk),
    p_end_date_sk       INT          REFERENCES date_dim(d_date_sk),
    p_item_sk           INT          REFERENCES item(i_item_sk),
    p_cost              NUMERIC(15,2),
    p_response_target   INT,
    p_promo_name        VARCHAR(50),
    p_channel_dmail     CHAR(1),
    p_channel_email     CHAR(1),
    p_channel_catalog   CHAR(1),
    p_channel_tv        CHAR(1),
    p_channel_radio     CHAR(1),
    p_channel_press     CHAR(1),
    p_channel_event     CHAR(1),
    p_channel_demo      CHAR(1),
    p_channel_details   VARCHAR(100),
    p_purpose           VARCHAR(15),
    p_discount_active   CHAR(1)
);

-- reason: return-reason dimension
CREATE TABLE reason (
    r_reason_sk         INT          PRIMARY KEY,
    r_reason_id         VARCHAR(16)  NOT NULL,
    r_reason_desc       VARCHAR(100)
);

-- store: brick-and-mortar store dimension
CREATE TABLE store (
    s_store_sk          INT          PRIMARY KEY,
    s_store_id          VARCHAR(16)  NOT NULL,
    s_rec_start_date    DATE,
    s_rec_end_date      DATE,
    s_closed_date_sk    INT          REFERENCES date_dim(d_date_sk),
    s_store_name        VARCHAR(50),
    s_number_employees  INT,
    s_floor_space       INT,
    s_hours             VARCHAR(20),
    s_manager           VARCHAR(40),
    s_market_id         INT,
    s_geography_class   VARCHAR(100),
    s_market_desc       VARCHAR(100),
    s_market_manager    VARCHAR(40),
    s_division_id       INT,
    s_division_name     VARCHAR(50),
    s_company_id        INT,
    s_company_name      VARCHAR(50),
    s_street_number     VARCHAR(10),
    s_street_name       VARCHAR(60),
    s_street_type       VARCHAR(15),
    s_suite_number      VARCHAR(10),
    s_city              VARCHAR(60),
    s_county            VARCHAR(30),
    s_state             CHAR(2),
    s_zip               VARCHAR(10),
    s_country           VARCHAR(20),
    s_gmt_offset        NUMERIC(5,2),
    s_tax_percentage    NUMERIC(5,2)
);

-- warehouse: distribution warehouse dimension
CREATE TABLE warehouse (
    w_warehouse_sk      INT          PRIMARY KEY,
    w_warehouse_id      VARCHAR(16)  NOT NULL,
    w_warehouse_name    VARCHAR(20),
    w_warehouse_sq_ft   INT,
    w_street_number     VARCHAR(10),
    w_street_name       VARCHAR(60),
    w_street_type       VARCHAR(15),
    w_suite_number      VARCHAR(10),
    w_city              VARCHAR(60),
    w_county            VARCHAR(30),
    w_state             CHAR(2),
    w_zip               VARCHAR(10),
    w_country           VARCHAR(20),
    w_gmt_offset        NUMERIC(5,2)
);

-- web_page: web content page dimension
CREATE TABLE web_page (
    wp_web_page_sk      INT          PRIMARY KEY,
    wp_web_page_id      VARCHAR(16)  NOT NULL,
    wp_rec_start_date   DATE,
    wp_rec_end_date     DATE,
    wp_creation_date_sk INT          REFERENCES date_dim(d_date_sk),
    wp_access_date_sk   INT          REFERENCES date_dim(d_date_sk),
    wp_autogen_flag     CHAR(1),
    wp_customer_sk      INT          REFERENCES customer(c_customer_sk),
    wp_url              VARCHAR(100),
    wp_type             VARCHAR(50),
    wp_char_count       INT,
    wp_link_count       INT,
    wp_image_count      INT,
    wp_max_ad_count     INT
);

-- web_site: web presence dimension
CREATE TABLE web_site (
    web_site_sk         INT          PRIMARY KEY,
    web_site_id         VARCHAR(16)  NOT NULL,
    web_rec_start_date  DATE,
    web_rec_end_date    DATE,
    web_name            VARCHAR(50),
    web_open_date_sk    INT          REFERENCES date_dim(d_date_sk),
    web_close_date_sk   INT          REFERENCES date_dim(d_date_sk),
    web_class           VARCHAR(50),
    web_manager         VARCHAR(40),
    web_mkt_id          INT,
    web_mkt_class       VARCHAR(50),
    web_mkt_desc        VARCHAR(100),
    web_market_manager  VARCHAR(40),
    web_company_id      INT,
    web_company_name    VARCHAR(50),
    web_street_number   VARCHAR(10),
    web_street_name     VARCHAR(60),
    web_street_type     VARCHAR(15),
    web_suite_number    VARCHAR(10),
    web_city            VARCHAR(60),
    web_county          VARCHAR(30),
    web_state           CHAR(2),
    web_zip             VARCHAR(10),
    web_country         VARCHAR(20),
    web_gmt_offset      NUMERIC(5,2),
    web_tax_percentage  NUMERIC(5,2)
);

-- call_center: telephone sales center dimension
CREATE TABLE call_center (
    cc_call_center_sk   INT          PRIMARY KEY,
    cc_call_center_id   VARCHAR(16)  NOT NULL,
    cc_rec_start_date   DATE,
    cc_rec_end_date     DATE,
    cc_closed_date_sk   INT          REFERENCES date_dim(d_date_sk),
    cc_open_date_sk     INT          REFERENCES date_dim(d_date_sk),
    cc_name             VARCHAR(50),
    cc_class            VARCHAR(50),
    cc_employees        INT,
    cc_sq_ft            INT,
    cc_hours            VARCHAR(20),
    cc_manager          VARCHAR(40),
    cc_mkt_id           INT,
    cc_mkt_class        VARCHAR(50),
    cc_mkt_desc         VARCHAR(100),
    cc_market_manager   VARCHAR(40),
    cc_division         INT,
    cc_division_name    VARCHAR(50),
    cc_company           INT,
    cc_company_name     VARCHAR(50),
    cc_street_number    VARCHAR(10),
    cc_street_name      VARCHAR(60),
    cc_street_type      VARCHAR(15),
    cc_suite_number     VARCHAR(10),
    cc_city             VARCHAR(60),
    cc_county           VARCHAR(30),
    cc_state            CHAR(2),
    cc_zip              VARCHAR(10),
    cc_country          VARCHAR(20),
    cc_gmt_offset       NUMERIC(5,2),
    cc_tax_percentage   NUMERIC(5,2)
);

-- ============================================================================
-- Fact tables
-- ============================================================================

-- store_sales: in-store sales fact
-- Key FK chains for multi-path test:
--   Direct:   ss_customer_sk -> customer.c_customer_sk (1 hop)
--   Indirect: store_returns(sr_ticket_number, sr_item_sk) -> store_sales,
--             then store_returns.sr_customer_sk -> customer (2 hops undirected)
CREATE TABLE store_sales (
    ss_sold_date_sk     INT          REFERENCES date_dim(d_date_sk),
    ss_sold_time_sk     INT          REFERENCES time_dim(t_time_sk),
    ss_item_sk          INT          NOT NULL REFERENCES item(i_item_sk),
    ss_customer_sk      INT          REFERENCES customer(c_customer_sk),
    ss_cdemo_sk         INT          REFERENCES customer_demographics(cd_demo_sk),
    ss_hdemo_sk         INT          REFERENCES household_demographics(hd_demo_sk),
    ss_addr_sk          INT          REFERENCES customer_address(ca_address_sk),
    ss_store_sk         INT          REFERENCES store(s_store_sk),
    ss_promo_sk         INT          REFERENCES promotion(p_promo_sk),
    ss_ticket_number    INT          NOT NULL,
    ss_quantity         INT,
    ss_wholesale_cost   NUMERIC(7,2),
    ss_list_price       NUMERIC(7,2),
    ss_sales_price      NUMERIC(7,2),
    ss_ext_discount_amt NUMERIC(7,2),
    ss_ext_sales_price  NUMERIC(7,2),
    ss_ext_wholesale_cost NUMERIC(7,2),
    ss_ext_list_price   NUMERIC(7,2),
    ss_ext_tax          NUMERIC(7,2),
    ss_coupon_amt       NUMERIC(7,2),
    ss_net_paid         NUMERIC(7,2),
    ss_net_paid_inc_tax NUMERIC(7,2),
    ss_net_profit       NUMERIC(7,2),
    PRIMARY KEY (ss_item_sk, ss_ticket_number)
);

-- store_returns: in-store return fact
-- Composite FK back to store_sales enables the multi-path test:
--   store_returns -> store_sales (via ticket+item) AND
--   store_returns -> customer (via sr_customer_sk)
CREATE TABLE store_returns (
    sr_returned_date_sk INT          REFERENCES date_dim(d_date_sk),
    sr_return_time_sk   INT          REFERENCES time_dim(t_time_sk),
    sr_item_sk          INT          NOT NULL,
    sr_customer_sk      INT          REFERENCES customer(c_customer_sk),
    sr_cdemo_sk         INT          REFERENCES customer_demographics(cd_demo_sk),
    sr_hdemo_sk         INT          REFERENCES household_demographics(hd_demo_sk),
    sr_addr_sk          INT          REFERENCES customer_address(ca_address_sk),
    sr_store_sk         INT          REFERENCES store(s_store_sk),
    sr_reason_sk        INT          REFERENCES reason(r_reason_sk),
    sr_ticket_number    INT          NOT NULL,
    sr_return_quantity  INT,
    sr_return_amt       NUMERIC(7,2),
    sr_return_tax       NUMERIC(7,2),
    sr_return_amt_inc_tax NUMERIC(7,2),
    sr_fee              NUMERIC(7,2),
    sr_return_ship_cost NUMERIC(7,2),
    sr_refunded_cash    NUMERIC(7,2),
    sr_reversed_charge  NUMERIC(7,2),
    sr_store_credit     NUMERIC(7,2),
    sr_net_loss         NUMERIC(7,2),
    PRIMARY KEY (sr_item_sk, sr_ticket_number),
    FOREIGN KEY (sr_item_sk, sr_ticket_number)
        REFERENCES store_sales(ss_item_sk, ss_ticket_number)
);

-- catalog_sales: catalog channel sales fact
-- FK chain for snowflake dimension walk test:
--   catalog_sales.cs_item_sk -> item -> item_brand -> item_category (3 hops)
CREATE TABLE catalog_sales (
    cs_sold_date_sk     INT          REFERENCES date_dim(d_date_sk),
    cs_sold_time_sk     INT          REFERENCES time_dim(t_time_sk),
    cs_ship_date_sk     INT          REFERENCES date_dim(d_date_sk),
    cs_bill_customer_sk INT          REFERENCES customer(c_customer_sk),
    cs_bill_cdemo_sk    INT          REFERENCES customer_demographics(cd_demo_sk),
    cs_bill_hdemo_sk    INT          REFERENCES household_demographics(hd_demo_sk),
    cs_bill_addr_sk     INT          REFERENCES customer_address(ca_address_sk),
    cs_ship_customer_sk INT          REFERENCES customer(c_customer_sk),
    cs_ship_cdemo_sk    INT          REFERENCES customer_demographics(cd_demo_sk),
    cs_ship_hdemo_sk    INT          REFERENCES household_demographics(hd_demo_sk),
    cs_ship_addr_sk     INT          REFERENCES customer_address(ca_address_sk),
    cs_call_center_sk   INT          REFERENCES call_center(cc_call_center_sk),
    cs_ship_mode_sk     INT,
    cs_warehouse_sk     INT          REFERENCES warehouse(w_warehouse_sk),
    cs_item_sk          INT          NOT NULL REFERENCES item(i_item_sk),
    cs_promo_sk         INT          REFERENCES promotion(p_promo_sk),
    cs_order_number     INT          NOT NULL,
    cs_quantity         INT,
    cs_wholesale_cost   NUMERIC(7,2),
    cs_list_price       NUMERIC(7,2),
    cs_sales_price      NUMERIC(7,2),
    cs_ext_discount_amt NUMERIC(7,2),
    cs_ext_sales_price  NUMERIC(7,2),
    cs_ext_wholesale_cost NUMERIC(7,2),
    cs_ext_list_price   NUMERIC(7,2),
    cs_ext_ship_cost    NUMERIC(7,2),
    cs_ext_tax          NUMERIC(7,2),
    cs_coupon_amt       NUMERIC(7,2),
    cs_net_paid         NUMERIC(7,2),
    cs_net_paid_inc_tax NUMERIC(7,2),
    cs_net_paid_inc_ship NUMERIC(7,2),
    cs_net_paid_inc_ship_tax NUMERIC(7,2),
    cs_net_profit       NUMERIC(7,2),
    PRIMARY KEY (cs_item_sk, cs_order_number)
);

-- catalog_returns: catalog channel return fact
CREATE TABLE catalog_returns (
    cr_returned_date_sk INT          REFERENCES date_dim(d_date_sk),
    cr_returned_time_sk INT          REFERENCES time_dim(t_time_sk),
    cr_item_sk          INT          NOT NULL,
    cr_refunded_customer_sk INT      REFERENCES customer(c_customer_sk),
    cr_refunded_cdemo_sk INT         REFERENCES customer_demographics(cd_demo_sk),
    cr_refunded_hdemo_sk INT         REFERENCES household_demographics(hd_demo_sk),
    cr_refunded_addr_sk INT          REFERENCES customer_address(ca_address_sk),
    cr_returning_customer_sk INT     REFERENCES customer(c_customer_sk),
    cr_returning_cdemo_sk INT        REFERENCES customer_demographics(cd_demo_sk),
    cr_returning_hdemo_sk INT        REFERENCES household_demographics(hd_demo_sk),
    cr_returning_addr_sk INT         REFERENCES customer_address(ca_address_sk),
    cr_call_center_sk   INT          REFERENCES call_center(cc_call_center_sk),
    cr_order_number     INT          NOT NULL,
    cr_return_quantity  INT,
    cr_return_amount    NUMERIC(7,2),
    cr_return_tax       NUMERIC(7,2),
    cr_return_amt_inc_tax NUMERIC(7,2),
    cr_fee              NUMERIC(7,2),
    cr_return_ship_cost NUMERIC(7,2),
    cr_refunded_cash    NUMERIC(7,2),
    cr_reversed_charge  NUMERIC(7,2),
    cr_store_credit     NUMERIC(7,2),
    cr_net_loss         NUMERIC(7,2),
    PRIMARY KEY (cr_item_sk, cr_order_number),
    FOREIGN KEY (cr_item_sk, cr_order_number)
        REFERENCES catalog_sales(cs_item_sk, cs_order_number)
);

-- web_sales: web channel sales fact
CREATE TABLE web_sales (
    ws_sold_date_sk     INT          REFERENCES date_dim(d_date_sk),
    ws_sold_time_sk     INT          REFERENCES time_dim(t_time_sk),
    ws_ship_date_sk     INT          REFERENCES date_dim(d_date_sk),
    ws_item_sk          INT          NOT NULL REFERENCES item(i_item_sk),
    ws_bill_customer_sk INT          REFERENCES customer(c_customer_sk),
    ws_bill_cdemo_sk    INT          REFERENCES customer_demographics(cd_demo_sk),
    ws_bill_hdemo_sk    INT          REFERENCES household_demographics(hd_demo_sk),
    ws_bill_addr_sk     INT          REFERENCES customer_address(ca_address_sk),
    ws_ship_customer_sk INT          REFERENCES customer(c_customer_sk),
    ws_ship_cdemo_sk    INT          REFERENCES customer_demographics(cd_demo_sk),
    ws_ship_hdemo_sk    INT          REFERENCES household_demographics(hd_demo_sk),
    ws_ship_addr_sk     INT          REFERENCES customer_address(ca_address_sk),
    ws_web_page_sk      INT          REFERENCES web_page(wp_web_page_sk),
    ws_web_site_sk      INT          REFERENCES web_site(web_site_sk),
    ws_promo_sk         INT          REFERENCES promotion(p_promo_sk),
    ws_warehouse_sk     INT          REFERENCES warehouse(w_warehouse_sk),
    ws_order_number     INT          NOT NULL,
    ws_quantity         INT,
    ws_wholesale_cost   NUMERIC(7,2),
    ws_list_price       NUMERIC(7,2),
    ws_sales_price      NUMERIC(7,2),
    ws_ext_discount_amt NUMERIC(7,2),
    ws_ext_sales_price  NUMERIC(7,2),
    ws_ext_wholesale_cost NUMERIC(7,2),
    ws_ext_list_price   NUMERIC(7,2),
    ws_ext_ship_cost    NUMERIC(7,2),
    ws_ext_tax          NUMERIC(7,2),
    ws_coupon_amt       NUMERIC(7,2),
    ws_net_paid         NUMERIC(7,2),
    ws_net_paid_inc_tax NUMERIC(7,2),
    ws_net_paid_inc_ship NUMERIC(7,2),
    ws_net_paid_inc_ship_tax NUMERIC(7,2),
    ws_net_profit       NUMERIC(7,2),
    PRIMARY KEY (ws_item_sk, ws_order_number)
);

-- web_returns: web channel return fact
CREATE TABLE web_returns (
    wr_returned_date_sk INT          REFERENCES date_dim(d_date_sk),
    wr_returned_time_sk INT          REFERENCES time_dim(t_time_sk),
    wr_item_sk          INT          NOT NULL,
    wr_refunded_customer_sk INT      REFERENCES customer(c_customer_sk),
    wr_refunded_cdemo_sk INT         REFERENCES customer_demographics(cd_demo_sk),
    wr_refunded_hdemo_sk INT         REFERENCES household_demographics(hd_demo_sk),
    wr_refunded_addr_sk INT          REFERENCES customer_address(ca_address_sk),
    wr_returning_customer_sk INT     REFERENCES customer(c_customer_sk),
    wr_returning_cdemo_sk INT        REFERENCES customer_demographics(cd_demo_sk),
    wr_returning_hdemo_sk INT        REFERENCES household_demographics(hd_demo_sk),
    wr_returning_addr_sk INT         REFERENCES customer_address(ca_address_sk),
    wr_web_page_sk      INT          REFERENCES web_page(wp_web_page_sk),
    wr_reason_sk        INT          REFERENCES reason(r_reason_sk),
    wr_order_number     INT          NOT NULL,
    wr_return_quantity  INT,
    wr_return_amt       NUMERIC(7,2),
    wr_return_tax       NUMERIC(7,2),
    wr_return_amt_inc_tax NUMERIC(7,2),
    wr_fee              NUMERIC(7,2),
    wr_return_ship_cost NUMERIC(7,2),
    wr_refunded_cash    NUMERIC(7,2),
    wr_reversed_charge  NUMERIC(7,2),
    wr_account_credit   NUMERIC(7,2),
    wr_net_loss         NUMERIC(7,2),
    PRIMARY KEY (wr_item_sk, wr_order_number),
    FOREIGN KEY (wr_item_sk, wr_order_number)
        REFERENCES web_sales(ws_item_sk, ws_order_number)
);

-- inventory: warehouse inventory snapshot fact
CREATE TABLE inventory (
    inv_date_sk         INT          NOT NULL REFERENCES date_dim(d_date_sk),
    inv_item_sk         INT          NOT NULL REFERENCES item(i_item_sk),
    inv_warehouse_sk    INT          NOT NULL REFERENCES warehouse(w_warehouse_sk),
    inv_quantity_on_hand INT,
    PRIMARY KEY (inv_date_sk, inv_item_sk, inv_warehouse_sk)
);
