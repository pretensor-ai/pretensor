-- Multi-statement maintenance script.
SELECT id FROM users WHERE active = true;

INSERT INTO orders (id, user_id) VALUES (1, 2);

UPDATE accounts SET balance = 0 WHERE closed = true;
