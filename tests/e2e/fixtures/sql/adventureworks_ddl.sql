-- AdventureWorks-shaped schema for Pretensor e2e testing
--
-- Modeled on the OLTP schema from `morenoh149/postgresDBSamples` (adventureworks/),
-- specifically commit c381d592a5eb3d0060c1a75c3f0365421f660551, which is itself a
-- Postgres port of Microsoft's "Adventure Works 2014 OLTP Script" (MIT-licensed).
--
-- This file is hand-crafted, NOT a verbatim vendoring of the upstream dump.
-- Reasons:
--   1. The upstream install.sql is a psql script with `\copy` meta-commands and
--      data-conversion logic that requires bundled CSV files; psycopg2 (used by
--      our session fixture) cannot execute psql backslash commands at all.
--   2. The CSV data is not vendored upstream — it's pulled from a CodePlex
--      archive that no longer exists. Even with psql, the dump fails partway
--      through every load.
--   3. Pretensor's graph builder reads pg_catalog metadata only; row data is
--      irrelevant to the audit-column regression this fixture exists for.
--
-- What is preserved from upstream AdventureWorks:
--   * Five schemas: humanresources, person, production, purchasing, sales
--   * ~36 tables across all five (vs Pagila's 8 across two), which stresses
--     multi-schema visibility / cluster discovery / pagination at a non-trivial
--     scale.
--   * `modifieddate` audit column on EVERY table (the audit-column pathology).
--   * The FK chains needed by `tests/e2e/test_adventureworks_traverse.py`:
--       - production.product → sales.customer
--           via product → specialofferproduct → salesorderdetail
--                 → salesorderheader → customer  (4 hops, undirected)
--       - humanresources.employee → sales.salesorderheader
--           via employee → salesperson → salesorderheader  (2 hops)
--       - person.person → production.product
--           shortest real chain is >=5 hops, exceeds traverse default
--           max_depth=4, so the tool must return either no path or only
--           high-confidence paths — never a `modifieddate` coincidence join.
--
-- Re-vendoring this file is a manual decision. If you want to refresh against
-- upstream, re-pin a commit hash above and reconcile table/FK shape by hand.
-- Do not automate.

-- ============================================================================
-- Schemas
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS humanresources;
CREATE SCHEMA IF NOT EXISTS person;
CREATE SCHEMA IF NOT EXISTS production;
CREATE SCHEMA IF NOT EXISTS purchasing;
CREATE SCHEMA IF NOT EXISTS sales;

-- ============================================================================
-- person  (identity, addressing, contact info)
-- ============================================================================

CREATE TABLE person.businessentity (
    businessentityid  SERIAL PRIMARY KEY,
    rowguid           UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE person.person (
    businessentityid  INT          PRIMARY KEY REFERENCES person.businessentity(businessentityid),
    persontype        CHAR(2)      NOT NULL DEFAULT 'IN',
    firstname         VARCHAR(50)  NOT NULL,
    middlename        VARCHAR(50),
    lastname          VARCHAR(50)  NOT NULL,
    suffix            VARCHAR(10),
    rowguid           UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE person.countryregion (
    countryregioncode  VARCHAR(3)   PRIMARY KEY,
    countryname        VARCHAR(50)  NOT NULL,
    modifieddate       TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE person.stateprovince (
    stateprovinceid     SERIAL       PRIMARY KEY,
    stateprovincecode   CHAR(3)      NOT NULL,
    countryregioncode   VARCHAR(3)   NOT NULL REFERENCES person.countryregion(countryregioncode),
    provincename        VARCHAR(50)  NOT NULL,
    modifieddate        TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE person.address (
    addressid         SERIAL       PRIMARY KEY,
    addressline1      VARCHAR(60)  NOT NULL,
    addressline2      VARCHAR(60),
    city              VARCHAR(30)  NOT NULL,
    stateprovinceid   INT          NOT NULL REFERENCES person.stateprovince(stateprovinceid),
    postalcode        VARCHAR(15)  NOT NULL,
    rowguid           UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE person.addresstype (
    addresstypeid     SERIAL       PRIMARY KEY,
    addresstypename   VARCHAR(50)  NOT NULL,
    rowguid           UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE person.businessentityaddress (
    businessentityid  INT          NOT NULL REFERENCES person.businessentity(businessentityid),
    addressid         INT          NOT NULL REFERENCES person.address(addressid),
    addresstypeid     INT          NOT NULL REFERENCES person.addresstype(addresstypeid),
    rowguid           UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (businessentityid, addressid, addresstypeid)
);

CREATE TABLE person.emailaddress (
    businessentityid  INT          NOT NULL REFERENCES person.person(businessentityid),
    emailaddressid    SERIAL       NOT NULL,
    emailaddress_text VARCHAR(50)  NOT NULL,
    rowguid           UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (businessentityid, emailaddressid)
);

CREATE TABLE person.personphone (
    businessentityid  INT          NOT NULL REFERENCES person.person(businessentityid),
    phonenumber       VARCHAR(25)  NOT NULL,
    phonetype         SMALLINT     NOT NULL DEFAULT 1,
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (businessentityid, phonenumber, phonetype)
);

-- ============================================================================
-- humanresources
-- ============================================================================

CREATE TABLE humanresources.department (
    departmentid     SERIAL       PRIMARY KEY,
    departmentname   VARCHAR(50)  NOT NULL,
    groupname        VARCHAR(50)  NOT NULL,
    modifieddate     TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE humanresources.shift (
    shiftid          SERIAL       PRIMARY KEY,
    shiftname        VARCHAR(50)  NOT NULL,
    starttime        TIME         NOT NULL,
    endtime          TIME         NOT NULL,
    modifieddate     TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE humanresources.employee (
    businessentityid     INT          PRIMARY KEY REFERENCES person.person(businessentityid),
    nationalidnumber     VARCHAR(15)  NOT NULL,
    loginid              VARCHAR(256) NOT NULL,
    jobtitle             VARCHAR(50)  NOT NULL,
    birthdate            DATE         NOT NULL,
    hiredate             DATE         NOT NULL,
    vacationhours        SMALLINT     NOT NULL DEFAULT 0,
    sickleavehours       SMALLINT     NOT NULL DEFAULT 0,
    rowguid              UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate         TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE humanresources.employeedepartmenthistory (
    businessentityid  INT       NOT NULL REFERENCES humanresources.employee(businessentityid),
    departmentid      INT       NOT NULL REFERENCES humanresources.department(departmentid),
    shiftid           INT       NOT NULL REFERENCES humanresources.shift(shiftid),
    startdate         DATE      NOT NULL,
    enddate           DATE,
    modifieddate      TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (businessentityid, departmentid, shiftid, startdate)
);

CREATE TABLE humanresources.employeepayhistory (
    businessentityid  INT          NOT NULL REFERENCES humanresources.employee(businessentityid),
    ratechangedate    TIMESTAMP    NOT NULL,
    rate              NUMERIC(8,4) NOT NULL,
    payfrequency      SMALLINT     NOT NULL,
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (businessentityid, ratechangedate)
);

CREATE TABLE humanresources.jobcandidate (
    jobcandidateid    SERIAL       PRIMARY KEY,
    businessentityid  INT          REFERENCES humanresources.employee(businessentityid),
    candidateresume   TEXT,
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- production
-- ============================================================================

CREATE TABLE production.unitmeasure (
    unitmeasurecode  CHAR(3)      PRIMARY KEY,
    unitname         VARCHAR(50)  NOT NULL,
    modifieddate     TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE production.productcategory (
    productcategoryid     SERIAL       PRIMARY KEY,
    categoryname          VARCHAR(50)  NOT NULL,
    rowguid               UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate          TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE production.productsubcategory (
    productsubcategoryid  SERIAL       PRIMARY KEY,
    productcategoryid     INT          NOT NULL REFERENCES production.productcategory(productcategoryid),
    subcategoryname       VARCHAR(50)  NOT NULL,
    rowguid               UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate          TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE production.productmodel (
    productmodelid    SERIAL       PRIMARY KEY,
    modelname         VARCHAR(50)  NOT NULL,
    catalogdescription TEXT,
    rowguid           UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE production.product (
    productid             SERIAL          PRIMARY KEY,
    productnumber         VARCHAR(25)     NOT NULL,
    productname           VARCHAR(50)     NOT NULL,
    productsubcategoryid  INT             REFERENCES production.productsubcategory(productsubcategoryid),
    productmodelid        INT             REFERENCES production.productmodel(productmodelid),
    sizeunitmeasurecode   CHAR(3)         REFERENCES production.unitmeasure(unitmeasurecode),
    weightunitmeasurecode CHAR(3)         REFERENCES production.unitmeasure(unitmeasurecode),
    listprice             NUMERIC(10,4)   NOT NULL,
    standardcost          NUMERIC(10,4)   NOT NULL,
    rowguid               UUID            NOT NULL DEFAULT gen_random_uuid(),
    modifieddate          TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE TABLE production.location (
    locationid       SERIAL          PRIMARY KEY,
    locationname     VARCHAR(50)     NOT NULL,
    costrate         NUMERIC(10,4)   NOT NULL DEFAULT 0,
    availability     NUMERIC(8,2)    NOT NULL DEFAULT 0,
    modifieddate     TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE TABLE production.productinventory (
    productid        INT          NOT NULL REFERENCES production.product(productid),
    locationid       INT          NOT NULL REFERENCES production.location(locationid),
    shelf            VARCHAR(10),
    bin              SMALLINT,
    quantity         SMALLINT     NOT NULL DEFAULT 0,
    rowguid          UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate     TIMESTAMP    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (productid, locationid)
);

CREATE TABLE production.productreview (
    productreviewid   SERIAL       PRIMARY KEY,
    productid         INT          NOT NULL REFERENCES production.product(productid),
    reviewername      VARCHAR(50)  NOT NULL,
    rating            INT          NOT NULL,
    comments          TEXT,
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE production.workorder (
    workorderid       SERIAL       PRIMARY KEY,
    productid         INT          NOT NULL REFERENCES production.product(productid),
    orderqty          INT          NOT NULL,
    scrappedqty       SMALLINT     NOT NULL DEFAULT 0,
    startdate         DATE         NOT NULL,
    enddate           DATE,
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- purchasing
-- ============================================================================

CREATE TABLE purchasing.shipmethod (
    shipmethodid     SERIAL          PRIMARY KEY,
    shipmethodname   VARCHAR(50)     NOT NULL,
    shipbase         NUMERIC(10,4)   NOT NULL DEFAULT 0,
    shiprate         NUMERIC(10,4)   NOT NULL DEFAULT 0,
    rowguid          UUID            NOT NULL DEFAULT gen_random_uuid(),
    modifieddate     TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE TABLE purchasing.vendor (
    businessentityid     INT          PRIMARY KEY REFERENCES person.businessentity(businessentityid),
    accountnumber        VARCHAR(15)  NOT NULL,
    vendorname           VARCHAR(50)  NOT NULL,
    creditrating         SMALLINT     NOT NULL DEFAULT 1,
    preferredvendorflag  BOOLEAN      NOT NULL DEFAULT TRUE,
    activeflag           BOOLEAN      NOT NULL DEFAULT TRUE,
    modifieddate         TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE purchasing.productvendor (
    productid           INT             NOT NULL REFERENCES production.product(productid),
    businessentityid    INT             NOT NULL REFERENCES purchasing.vendor(businessentityid),
    averageleadtime     INT             NOT NULL,
    standardprice       NUMERIC(10,4)   NOT NULL,
    minorderqty         INT             NOT NULL,
    maxorderqty         INT             NOT NULL,
    modifieddate        TIMESTAMP       NOT NULL DEFAULT NOW(),
    PRIMARY KEY (productid, businessentityid)
);

CREATE TABLE purchasing.purchaseorderheader (
    purchaseorderid    SERIAL          PRIMARY KEY,
    employeeid         INT             NOT NULL REFERENCES humanresources.employee(businessentityid),
    vendorid           INT             NOT NULL REFERENCES purchasing.vendor(businessentityid),
    shipmethodid       INT             NOT NULL REFERENCES purchasing.shipmethod(shipmethodid),
    orderdate          DATE            NOT NULL,
    shipdate           DATE,
    subtotal           NUMERIC(12,4)   NOT NULL DEFAULT 0,
    modifieddate       TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE TABLE purchasing.purchaseorderdetail (
    purchaseorderid    INT             NOT NULL REFERENCES purchasing.purchaseorderheader(purchaseorderid),
    purchaseorderdetailid SERIAL       NOT NULL,
    productid          INT             NOT NULL REFERENCES production.product(productid),
    orderqty           SMALLINT        NOT NULL,
    unitprice          NUMERIC(10,4)   NOT NULL,
    modifieddate       TIMESTAMP       NOT NULL DEFAULT NOW(),
    PRIMARY KEY (purchaseorderid, purchaseorderdetailid)
);

-- ============================================================================
-- sales
-- ============================================================================

CREATE TABLE sales.creditcard (
    creditcardid     SERIAL       PRIMARY KEY,
    cardtype         VARCHAR(50)  NOT NULL,
    cardnumber       VARCHAR(25)  NOT NULL,
    expmonth         SMALLINT     NOT NULL,
    expyear          SMALLINT     NOT NULL,
    modifieddate     TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE sales.salesterritory (
    territoryid          SERIAL          PRIMARY KEY,
    territoryname        VARCHAR(50)     NOT NULL,
    countryregioncode    VARCHAR(3)      NOT NULL REFERENCES person.countryregion(countryregioncode),
    salesytd             NUMERIC(14,4)   NOT NULL DEFAULT 0,
    rowguid              UUID            NOT NULL DEFAULT gen_random_uuid(),
    modifieddate         TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE TABLE sales.salesreason (
    salesreasonid    SERIAL       PRIMARY KEY,
    reasonname       VARCHAR(50)  NOT NULL,
    reasontype       VARCHAR(50)  NOT NULL,
    modifieddate     TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE sales.specialoffer (
    specialofferid    SERIAL          PRIMARY KEY,
    offerdescription  VARCHAR(255)    NOT NULL,
    discountpct       NUMERIC(10,4)   NOT NULL DEFAULT 0,
    offertype         VARCHAR(50)     NOT NULL,
    startdate         DATE            NOT NULL,
    enddate           DATE            NOT NULL,
    rowguid           UUID            NOT NULL DEFAULT gen_random_uuid(),
    modifieddate      TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE TABLE sales.specialofferproduct (
    specialofferid    INT          NOT NULL REFERENCES sales.specialoffer(specialofferid),
    productid         INT          NOT NULL REFERENCES production.product(productid),
    rowguid           UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (specialofferid, productid)
);

CREATE TABLE sales.customer (
    customerid       SERIAL       PRIMARY KEY,
    personid         INT          REFERENCES person.person(businessentityid),
    territoryid      INT          REFERENCES sales.salesterritory(territoryid),
    accountnumber    VARCHAR(10)  NOT NULL,
    rowguid          UUID         NOT NULL DEFAULT gen_random_uuid(),
    modifieddate     TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE sales.salesperson (
    businessentityid     INT             PRIMARY KEY REFERENCES humanresources.employee(businessentityid),
    territoryid          INT             REFERENCES sales.salesterritory(territoryid),
    salesquota           NUMERIC(14,4),
    bonus                NUMERIC(14,4)   NOT NULL DEFAULT 0,
    commissionpct        NUMERIC(10,4)   NOT NULL DEFAULT 0,
    salesytd             NUMERIC(14,4)   NOT NULL DEFAULT 0,
    rowguid              UUID            NOT NULL DEFAULT gen_random_uuid(),
    modifieddate         TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE TABLE sales.salesorderheader (
    salesorderid       SERIAL          PRIMARY KEY,
    customerid         INT             NOT NULL REFERENCES sales.customer(customerid),
    salespersonid      INT             REFERENCES sales.salesperson(businessentityid),
    territoryid        INT             REFERENCES sales.salesterritory(territoryid),
    billtoaddressid    INT             NOT NULL REFERENCES person.address(addressid),
    shiptoaddressid    INT             NOT NULL REFERENCES person.address(addressid),
    shipmethodid       INT             NOT NULL REFERENCES purchasing.shipmethod(shipmethodid),
    creditcardid       INT             REFERENCES sales.creditcard(creditcardid),
    orderdate          TIMESTAMP       NOT NULL DEFAULT NOW(),
    duedate            TIMESTAMP       NOT NULL,
    shipdate           TIMESTAMP,
    subtotal           NUMERIC(12,4)   NOT NULL DEFAULT 0,
    rowguid            UUID            NOT NULL DEFAULT gen_random_uuid(),
    modifieddate       TIMESTAMP       NOT NULL DEFAULT NOW()
);

CREATE TABLE sales.salesorderdetail (
    salesorderid          INT             NOT NULL REFERENCES sales.salesorderheader(salesorderid),
    salesorderdetailid    SERIAL          NOT NULL,
    specialofferid        INT             NOT NULL,
    productid             INT             NOT NULL,
    orderqty              SMALLINT        NOT NULL,
    unitprice             NUMERIC(10,4)   NOT NULL,
    rowguid               UUID            NOT NULL DEFAULT gen_random_uuid(),
    modifieddate          TIMESTAMP       NOT NULL DEFAULT NOW(),
    PRIMARY KEY (salesorderid, salesorderdetailid),
    FOREIGN KEY (specialofferid, productid)
        REFERENCES sales.specialofferproduct(specialofferid, productid)
);

CREATE TABLE sales.salestaxrate (
    salestaxrateid    SERIAL       PRIMARY KEY,
    stateprovinceid   INT          NOT NULL REFERENCES person.stateprovince(stateprovinceid),
    taxtype           SMALLINT     NOT NULL DEFAULT 1,
    taxrate           NUMERIC(10,4) NOT NULL DEFAULT 0,
    modifieddate      TIMESTAMP    NOT NULL DEFAULT NOW()
);
