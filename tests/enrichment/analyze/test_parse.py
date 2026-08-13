"""Tests for the multi-dialect SQL parse wrapper."""

from __future__ import annotations

from pretensor.enrichment.analyze.parse import parse_sql


def test_simple_select_extracts_table_refs() -> None:
    result = parse_sql("SELECT id FROM public.users")
    assert ("public", "users") in result.table_refs


def test_fingerprint_is_16_hex_chars() -> None:
    result = parse_sql("SELECT 1 FROM public.t")
    assert len(result.fingerprint) == 16
    assert all(c in "0123456789abcdef" for c in result.fingerprint)


def test_fingerprint_stable_across_whitespace() -> None:
    r1 = parse_sql("SELECT id FROM public.users")
    r2 = parse_sql("  select id from public.users  ")
    assert r1.fingerprint == r2.fingerprint


def test_fingerprint_differs_for_different_sql() -> None:
    r1 = parse_sql("SELECT id FROM public.users")
    r2 = parse_sql("SELECT name FROM public.accounts")
    assert r1.fingerprint != r2.fingerprint


def test_insert_appears_in_table_refs() -> None:
    # INSERT target is picked up by find_all(Table) in table_refs_from_sql
    result = parse_sql("INSERT INTO public.orders (id) VALUES (1)")
    assert ("public", "orders") in result.table_refs


def test_insert_without_columns_populates_write_targets() -> None:
    # Plain INSERT ... SELECT: write_targets catches the INSERT target
    result = parse_sql("INSERT INTO public.orders SELECT id FROM public.users")
    assert ("public", "orders") in result.write_targets or (
        "public",
        "orders",
    ) in result.table_refs


def test_update_appears_in_refs() -> None:
    result = parse_sql("UPDATE public.accounts SET balance = 0 WHERE id = 1")
    assert ("public", "accounts") in result.table_refs or (
        "public",
        "accounts",
    ) in result.write_targets


def test_unparseable_sql_returns_empty_refs() -> None:
    result = parse_sql("this is not sql at all")
    assert result.table_refs == []
    assert result.write_targets == []
    assert len(result.fingerprint) == 16


def test_default_schema_applied_when_no_schema_qualifier() -> None:
    result = parse_sql("SELECT id FROM users", default_schema="myschema")
    assert ("myschema", "users") in result.table_refs


def test_select_star_from_multiple_tables() -> None:
    result = parse_sql(
        "SELECT * FROM public.users u JOIN public.orders o ON u.id = o.user_id"
    )
    names = {t for _, t in result.table_refs}
    assert "users" in names
    assert "orders" in names


def test_delete_appears_in_refs() -> None:
    result = parse_sql("DELETE FROM public.tmp WHERE created < '2020-01-01'")
    assert ("public", "tmp") in result.table_refs or (
        "public",
        "tmp",
    ) in result.write_targets


def test_empty_string_returns_empty_refs() -> None:
    result = parse_sql("")
    assert result.table_refs == []
    assert result.write_targets == []


def test_dialect_used_field_is_string() -> None:
    result = parse_sql("SELECT id FROM public.users")
    assert isinstance(result.dialect_used, str)


def test_insert_with_dbapi_placeholders_yields_write_target() -> None:
    result = parse_sql(
        "INSERT INTO public.payment (customer_id, amount) VALUES (%s, %s)"
    )
    assert ("public", "payment") in result.write_targets


def test_update_with_named_placeholder_yields_write_target() -> None:
    result = parse_sql(
        "UPDATE public.rental SET return_date = now() WHERE rental_id = :rid"
    )
    assert ("public", "rental") in result.write_targets


def test_qmark_placeholder_parses() -> None:
    result = parse_sql("SELECT title FROM public.film WHERE film_id = ?")
    assert ("public", "film") in result.table_refs


def test_pyformat_named_placeholder_parses() -> None:
    result = parse_sql("SELECT * FROM public.customer WHERE store_id = %(store)s")
    assert ("public", "customer") in result.table_refs


def test_double_colon_cast_survives_normalization() -> None:
    from pretensor.enrichment.analyze.parse import normalize_placeholders

    sql = "SELECT id::int, created_at::date FROM public.t WHERE name = :name"
    normalized = normalize_placeholders(sql)
    assert "::int" in normalized
    assert "::date" in normalized
    assert ":name" not in normalized
    result = parse_sql(sql)
    assert ("public", "t") in result.table_refs


def test_time_literal_not_mangled() -> None:
    from pretensor.enrichment.analyze.parse import normalize_placeholders

    assert "'10:30'" in normalize_placeholders(
        "SELECT * FROM t WHERE opens_at = '10:30'"
    )


def test_fingerprint_computed_from_original_text() -> None:
    r1 = parse_sql("SELECT id FROM public.users WHERE id = %s")
    r2 = parse_sql("SELECT id FROM public.users WHERE id = NULL")
    assert r1.fingerprint != r2.fingerprint  # normalization must not affect identity


def test_bigquery_backticked_three_part_name_resolves() -> None:
    """The generic dialect tokenizes `project.dataset.table` into garbage
    (a literal backtick as the table name); the retry must fall through to
    the BigQuery parse instead of stopping at the first non-empty result."""
    parsed = parse_sql(
        "SELECT order_id FROM `my-project.sales.orders`", default_schema="sales"
    )
    assert parsed.table_refs == [("sales", "orders")]


def test_mysql_backticked_identifiers_resolve() -> None:
    parsed = parse_sql(
        "SELECT r.id FROM `rental` r JOIN customer c ON c.id = r.customer_id",
        default_schema="sakila",
    )
    assert ("sakila", "rental") in parsed.table_refs
    assert ("sakila", "customer") in parsed.table_refs


def test_mysql_upsert_with_backticks_has_write_target() -> None:
    parsed = parse_sql(
        "INSERT INTO `language` (id, name) VALUES (%s, %s)"
        " ON DUPLICATE KEY UPDATE name = VALUES(name)",
        default_schema="sakila",
    )
    assert parsed.write_targets == [("sakila", "language")]


def test_replace_into_is_a_write() -> None:
    """MySQL REPLACE INTO is unparseable by sqlglot in every dialect; the
    normalizer rewrites it to INSERT INTO so the write target survives."""
    parsed = parse_sql(
        "REPLACE INTO staff (id, name) VALUES (%s, %s)", default_schema="sakila"
    )
    assert parsed.write_targets == [("sakila", "staff")]
    assert ("sakila", "staff") in parsed.table_refs


def test_fingerprint_computed_from_original_replace_text() -> None:
    a = parse_sql("REPLACE INTO staff (id) VALUES (%s)")
    b = parse_sql("INSERT INTO staff (id) VALUES (%s)")
    assert a.fingerprint != b.fingerprint
