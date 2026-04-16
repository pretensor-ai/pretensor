-- Minimal Pagila-derived schema for E2E testing
-- 2 schemas: public, staff
-- 7 tables in public: film, actor, film_actor, inventory, customer, rental, payment
-- 1 table in staff: staff
-- FK chains: inventory→film, rental→inventory, rental→customer, payment→rental

CREATE SCHEMA IF NOT EXISTS staff;

-- ============================================================
-- public schema
-- ============================================================

CREATE TABLE public.film (
    film_id       SERIAL PRIMARY KEY,
    title         VARCHAR(255)   NOT NULL,
    description   TEXT,
    release_year  SMALLINT,
    rental_rate   NUMERIC(4,2)   NOT NULL DEFAULT 4.99,
    length        SMALLINT,
    rating        VARCHAR(10)    DEFAULT 'G',
    last_update   TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

CREATE TABLE public.actor (
    actor_id    SERIAL PRIMARY KEY,
    first_name  VARCHAR(45)  NOT NULL,
    last_name   VARCHAR(45)  NOT NULL,
    birth_date  DATE,
    active      BOOLEAN      NOT NULL DEFAULT TRUE,
    last_update TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE public.film_actor (
    actor_id    INT          NOT NULL REFERENCES public.actor(actor_id),
    film_id     INT          NOT NULL REFERENCES public.film(film_id),
    role        VARCHAR(50),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (actor_id, film_id)
);

CREATE TABLE public.inventory (
    inventory_id  SERIAL PRIMARY KEY,
    film_id       INT          NOT NULL REFERENCES public.film(film_id),
    store_id      SMALLINT     NOT NULL DEFAULT 1,
    condition     VARCHAR(20)  NOT NULL DEFAULT 'good',
    acquired_at   DATE,
    last_update   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE public.customer (
    customer_id  SERIAL PRIMARY KEY,
    first_name   VARCHAR(45)   NOT NULL,
    last_name    VARCHAR(45)   NOT NULL,
    email        VARCHAR(100),
    active       BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    last_update  TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE TABLE public.rental (
    rental_id     SERIAL PRIMARY KEY,
    inventory_id  INT          NOT NULL REFERENCES public.inventory(inventory_id),
    customer_id   INT          NOT NULL REFERENCES public.customer(customer_id),
    rental_date   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    return_date   TIMESTAMPTZ,
    staff_id      SMALLINT     NOT NULL DEFAULT 1,
    last_update   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE public.payment (
    payment_id   SERIAL PRIMARY KEY,
    rental_id    INT           NOT NULL REFERENCES public.rental(rental_id),
    customer_id  INT           NOT NULL REFERENCES public.customer(customer_id),
    amount       NUMERIC(5,2)  NOT NULL,
    payment_date TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    method       VARCHAR(20)   NOT NULL DEFAULT 'cash'
);

-- ============================================================
-- staff schema (multi-schema coverage)
-- ============================================================

CREATE TABLE staff.staff (
    staff_id    SERIAL PRIMARY KEY,
    first_name  VARCHAR(45)   NOT NULL,
    last_name   VARCHAR(45)   NOT NULL,
    email       VARCHAR(100),
    store_id    SMALLINT      NOT NULL DEFAULT 1,
    active      BOOLEAN       NOT NULL DEFAULT TRUE,
    username    VARCHAR(50)   NOT NULL,
    last_update TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
