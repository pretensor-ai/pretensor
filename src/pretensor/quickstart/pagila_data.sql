-- Minimal Pagila seed for the hello-world quickstart.
-- Curated subset: enough rows for `query` / `context` / `traverse` to return
-- non-empty results, while staying small enough to commit to the repo.

INSERT INTO public.film (title, description, release_year, rental_rate, length, rating) VALUES
    ('Academy Dinosaur', 'A Boring Saga of a Dinosaur', 2006, 0.99, 86, 'PG'),
    ('Ace Goldfinger',   'An Astounding Epistle in Hong Kong', 2006, 4.99, 48, 'G'),
    ('Adaptation Holes', 'A Astounding Reflection of a Lumberjack', 2006, 2.99, 50, 'NC-17'),
    ('Affair Prejudice', 'A Fanciful Documentary of a Frisbee', 2006, 2.99, 117, 'G'),
    ('African Egg',      'A Fast-Paced Documentary of a Pastry Chef', 2006, 2.99, 130, 'G');

INSERT INTO public.actor (first_name, last_name, birth_date) VALUES
    ('Penelope', 'Guiness', '1970-04-12'),
    ('Nick',     'Wahlberg', '1971-01-30'),
    ('Ed',       'Chase',    '1955-07-08'),
    ('Jennifer', 'Davis',    '1980-09-15'),
    ('Johnny',   'Lollobrigida', '1948-11-22');

INSERT INTO public.film_actor (actor_id, film_id, role) VALUES
    (1, 1, 'lead'),
    (2, 1, 'support'),
    (3, 2, 'lead'),
    (4, 3, 'lead'),
    (5, 4, 'support');

INSERT INTO public.inventory (film_id, store_id, condition, acquired_at) VALUES
    (1, 1, 'good', '2024-01-15'),
    (1, 1, 'good', '2024-01-15'),
    (2, 1, 'fair', '2024-02-01'),
    (3, 2, 'good', '2024-02-10'),
    (4, 1, 'good', '2024-03-05');

INSERT INTO public.customer (first_name, last_name, email) VALUES
    ('Mary',    'Smith',   'mary.smith@example.com'),
    ('Patricia','Johnson', 'patricia.j@example.com'),
    ('Linda',   'Williams','linda.w@example.com');

INSERT INTO public.rental (inventory_id, customer_id, rental_date, return_date) VALUES
    (1, 1, '2024-05-24 22:53:30+00', '2024-05-26 22:04:30+00'),
    (2, 2, '2024-05-24 22:54:33+00', '2024-05-28 19:40:33+00'),
    (3, 1, '2024-05-25 11:30:37+00', '2024-06-01 14:25:37+00'),
    (4, 3, '2024-05-26 09:15:00+00', '2024-05-30 12:00:00+00'),
    (5, 2, '2024-05-27 13:42:11+00', NULL);

INSERT INTO public.payment (rental_id, customer_id, amount, method) VALUES
    (1, 1, 2.99, 'cash'),
    (2, 2, 4.99, 'card'),
    (3, 1, 0.99, 'card'),
    (4, 3, 2.99, 'cash'),
    (5, 2, 4.99, 'card');

INSERT INTO staff.staff (first_name, last_name, email, store_id, username) VALUES
    ('Mike', 'Hillyer',  'mike@example.com',  1, 'mike'),
    ('Jon',  'Stephens', 'jon@example.com',   2, 'jon');

-- Reset sequences so subsequent inserts (if any) start past seeded ids.
SELECT setval(pg_get_serial_sequence('public.film',      'film_id'),      (SELECT MAX(film_id)      FROM public.film));
SELECT setval(pg_get_serial_sequence('public.actor',     'actor_id'),     (SELECT MAX(actor_id)     FROM public.actor));
SELECT setval(pg_get_serial_sequence('public.inventory', 'inventory_id'), (SELECT MAX(inventory_id) FROM public.inventory));
SELECT setval(pg_get_serial_sequence('public.customer',  'customer_id'),  (SELECT MAX(customer_id)  FROM public.customer));
SELECT setval(pg_get_serial_sequence('public.rental',    'rental_id'),    (SELECT MAX(rental_id)    FROM public.rental));
SELECT setval(pg_get_serial_sequence('public.payment',   'payment_id'),   (SELECT MAX(payment_id)   FROM public.payment));
SELECT setval(pg_get_serial_sequence('staff.staff',      'staff_id'),     (SELECT MAX(staff_id)     FROM staff.staff));
