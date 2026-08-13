"""Tests for the Python AST SQL extractor."""

from __future__ import annotations

from pathlib import Path

from pretensor.enrichment.analyze.extract_python import extract_sql_candidates


def _parse(source: str, tmp_path: Path) -> list:
    f = tmp_path / "src.py"
    f.write_text(source, encoding="utf-8")
    return extract_sql_candidates(f)


def test_assignment_to_query_variable(tmp_path: Path) -> None:
    candidates = _parse('query = "SELECT id FROM public.users"', tmp_path)
    assert len(candidates) == 1
    assert candidates[0].symbol == "query"
    assert candidates[0].text == "SELECT id FROM public.users"


def test_assignment_to_sql_variable(tmp_path: Path) -> None:
    candidates = _parse('sql = "SELECT name FROM public.items"', tmp_path)
    assert len(candidates) == 1
    assert candidates[0].symbol == "sql"


def test_assignment_to_stmt_variable(tmp_path: Path) -> None:
    candidates = _parse(
        "stmt = \"INSERT INTO public.log (msg) VALUES ('x')\"", tmp_path
    )
    assert len(candidates) == 1
    assert candidates[0].symbol == "stmt"


def test_assignment_to_statement_variable(tmp_path: Path) -> None:
    candidates = _parse('statement = "DELETE FROM public.tmp"', tmp_path)
    assert len(candidates) == 1
    assert candidates[0].symbol == "statement"


def test_assignment_case_insensitive(tmp_path: Path) -> None:
    candidates = _parse('SQL = "SELECT 1"', tmp_path)
    assert len(candidates) == 1


def test_method_call_execute(tmp_path: Path) -> None:
    source = 'conn.execute("SELECT * FROM public.orders")'
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 1
    assert candidates[0].symbol == "execute"
    assert candidates[0].text == "SELECT * FROM public.orders"


def test_method_call_executemany(tmp_path: Path) -> None:
    source = 'db.executemany("INSERT INTO public.t (x) VALUES (?)", data)'
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 1
    assert candidates[0].symbol == "executemany"


def test_method_call_query(tmp_path: Path) -> None:
    source = 'cursor.query("SELECT 1")'
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 1
    assert candidates[0].symbol == "query"


def test_method_call_raw(tmp_path: Path) -> None:
    source = 'session.raw("SELECT id FROM public.sessions")'
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 1
    assert candidates[0].symbol == "raw"


def test_method_call_exec_driver_sql(tmp_path: Path) -> None:
    source = 'conn.exec_driver_sql("SELECT col FROM public.data")'
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 1
    assert candidates[0].symbol == "exec_driver_sql"


def test_fstring_with_sql_prefix(tmp_path: Path) -> None:
    source = 'table = "users"\nsql = f"SELECT * FROM public.{table}"'
    candidates = _parse(source, tmp_path)
    fstring_cands = [c for c in candidates if c.kind == "fstring"]
    assert len(fstring_cands) == 1
    assert fstring_cands[0].confidence_bucket == "low"
    assert fstring_cands[0].symbol == "sql"  # the assigned variable, not "fstring"
    assert fstring_cands[0].text.startswith("SELECT")


def test_fstring_without_sql_prefix_not_emitted(tmp_path: Path) -> None:
    source = 'sql = f"WHERE id = {value}"'
    candidates = _parse(source, tmp_path)
    fstring_cands = [c for c in candidates if c.kind == "fstring"]
    assert len(fstring_cands) == 0


def test_noqa_same_line_suppresses(tmp_path: Path) -> None:
    source = 'query = "SELECT id FROM public.t"  # noqa: pretensor-analyze'
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 0


def test_noqa_preceding_line_suppresses(tmp_path: Path) -> None:
    source = '# noqa: pretensor-analyze\nquery = "SELECT id FROM public.t"'
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 0


def test_noqa_does_not_suppress_non_adjacent_line(tmp_path: Path) -> None:
    # noqa on line 1, empty line on line 2, SQL string on line 3 — not suppressed
    source = (
        'query = "SELECT id FROM public.t"  # noqa: pretensor-analyze\n'
        "\n"
        'sql = "SELECT name FROM public.u"\n'
    )
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 1
    assert candidates[0].symbol == "sql"


def test_non_sql_string_assigned_to_query_is_still_extracted(tmp_path: Path) -> None:
    # extract_python emits based on syntactic context only, not SQL content
    source = 'query = "hello world"'
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 1


def test_non_string_value_not_extracted(tmp_path: Path) -> None:
    source = "query = 42"
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 0


def test_syntax_error_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "bad.py"
    f.write_text("def broken(:\n", encoding="utf-8")
    candidates = extract_sql_candidates(f)
    assert candidates == []


def test_line_numbers_are_correct(tmp_path: Path) -> None:
    source = 'x = 1\nquery = "SELECT 1"\ny = 2\n'
    candidates = _parse(source, tmp_path)
    assert len(candidates) == 1
    assert candidates[0].line_start == 2


def test_suffixed_variable_names_are_candidates(tmp_path: Path) -> None:
    src = 'merge_sql = "UPDATE users SET active = false WHERE id = %s"\n'
    f = tmp_path / "m.py"
    f.write_text(src)
    cands = extract_sql_candidates(f)
    assert len(cands) == 1
    assert cands[0].symbol == "merge_sql"


def test_prefixed_variable_names_are_candidates(tmp_path: Path) -> None:
    src = 'sql_fetch_users: str = "SELECT id FROM users"\n'
    f = tmp_path / "p.py"
    f.write_text(src)
    cands = extract_sql_candidates(f)
    assert len(cands) == 1
    assert cands[0].symbol == "sql_fetch_users"


def test_unrelated_names_still_ignored(tmp_path: Path) -> None:
    src = 'sequel = "SELECT id FROM users"\nqueryish = "SELECT id FROM users"\n'
    f = tmp_path / "u.py"
    f.write_text(src)
    assert extract_sql_candidates(f) == []
