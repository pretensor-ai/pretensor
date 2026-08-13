import sqlite3

conn = sqlite3.connect(":memory:")
conn.execute("INSERT INTO public.orders (id, amount) VALUES (1, 99.9)")
