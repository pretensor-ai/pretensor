"""Unit tests for the L3 prompt template + hash helpers."""

from __future__ import annotations

from pretensor.benchmark.l3.prompt import (
    build_pretensor_system_prompt,
    build_system_prompt,
    pretensor_prompt_template_hash,
    prompt_hash,
    prompt_template_hash,
    strip_sql_fences,
)

# ---------------------------------------------------------------------------
# template substitution
# ---------------------------------------------------------------------------


def test_build_system_prompt_substitutes_dataset_and_ddl() -> None:
    rendered = build_system_prompt("pagila", "CREATE TABLE film (id int);")
    assert "pagila" in rendered
    assert "CREATE TABLE film" in rendered


def test_build_system_prompt_does_not_swallow_braces_in_ddl() -> None:
    """DDL containing braces must round-trip through the template intact."""
    ddl = "-- hint: see {docs}\nCREATE TABLE t (id int);"
    rendered = build_system_prompt("tpch", ddl)
    assert "{docs}" in rendered


# ---------------------------------------------------------------------------
# template hash stability
# ---------------------------------------------------------------------------


def test_prompt_template_hash_is_a_full_sha256() -> None:
    """64 hex characters means the hash is SHA-256 (not truncated)."""
    h = prompt_template_hash()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_prompt_template_hash_stable_across_calls() -> None:
    assert prompt_template_hash() == prompt_template_hash()


def test_prompt_hash_changes_when_ddl_changes() -> None:
    a = prompt_hash(build_system_prompt("pagila", "CREATE TABLE a (id int);"))
    b = prompt_hash(build_system_prompt("pagila", "CREATE TABLE b (id int);"))
    assert a != b


def test_prompt_hash_changes_when_dataset_name_changes() -> None:
    ddl = "CREATE TABLE t (id int);"
    a = prompt_hash(build_system_prompt("pagila", ddl))
    b = prompt_hash(build_system_prompt("tpch", ddl))
    assert a != b


def test_prompt_hash_stable_for_identical_inputs() -> None:
    rendered = build_system_prompt("pagila", "CREATE TABLE t (id int);")
    assert prompt_hash(rendered) == prompt_hash(rendered)


# ---------------------------------------------------------------------------
# fence stripping
# ---------------------------------------------------------------------------


def test_strip_sql_fences_handles_sql_fence() -> None:
    raw = "```sql\nSELECT 1\n```"
    assert strip_sql_fences(raw) == "SELECT 1"


def test_strip_sql_fences_handles_plain_fence() -> None:
    raw = "```\nSELECT 1\n```"
    assert strip_sql_fences(raw) == "SELECT 1"


def test_strip_sql_fences_handles_postgres_fence_label() -> None:
    raw = "```postgres\nSELECT 1\n```"
    assert strip_sql_fences(raw) == "SELECT 1"


def test_strip_sql_fences_no_fence_returns_input() -> None:
    assert strip_sql_fences("SELECT 1") == "SELECT 1"


def test_strip_sql_fences_tolerates_surrounding_whitespace() -> None:
    raw = "\n  ```sql\nSELECT 1\n```  \n"
    assert strip_sql_fences(raw) == "SELECT 1"


def test_strip_sql_fences_preserves_multiline_body() -> None:
    raw = "```sql\nSELECT a,\n       b\nFROM t\n```"
    assert strip_sql_fences(raw) == "SELECT a,\n       b\nFROM t"


# ---------------------------------------------------------------------------
# pretensor variant
# ---------------------------------------------------------------------------


def test_build_pretensor_system_prompt_mentions_mcp_tools_and_dataset() -> None:
    """Pretensor prompt tells the agent to gather context via MCP tools.

    Asserting the substantive instruction (rather than the literal
    template hash) gives this test useful failure semantics: a regression
    in the wording — e.g. dropping the tool-use instruction or losing
    the dataset hint — fails this case with a comprehensible message,
    not just a hash diff.
    """
    rendered = build_pretensor_system_prompt("pagila")
    assert "pagila" in rendered
    assert "MCP tools" in rendered
    # The pretensor prompt must NOT inline DDL — that's the baseline's
    # job. If the template ever grows a ``{ddl_text}`` slot the
    # pretensor / baseline comparison would no longer be apples-to-apples.
    assert "{ddl_text}" not in rendered
    assert "Schema (DDL):" not in rendered


def test_pretensor_prompt_template_hash_is_stable_and_distinct_from_baseline() -> None:
    """Hash is deterministic across calls and differs from the baseline hash.

    The two hashes diverging is what lets the JSON envelope tell apart
    the runners post-hoc: an auditor reading two L3 results can
    confirm they came from the two different prompts without reading
    the prompts themselves.
    """
    h1 = pretensor_prompt_template_hash()
    h2 = pretensor_prompt_template_hash()
    assert h1 == h2
    assert h1 != prompt_template_hash()
    assert len(h1) == 64  # sha256 hex


def test_build_pretensor_system_prompt_is_invariant_under_dataset() -> None:
    """Different dataset names produce identical hashes for the *template*.

    The rendered prompt is dataset-specific, but the template hash
    (which lives in the JSON envelope) is not. Catching drift in this
    invariant prevents a future refactor from accidentally folding the
    dataset name into the template hash.
    """
    a = build_pretensor_system_prompt("pagila")
    b = build_pretensor_system_prompt("tpch")
    assert a != b
    # Per-rendering hashes differ; the underlying template hash does not.
    assert prompt_hash(a) != prompt_hash(b)
    assert pretensor_prompt_template_hash() == pretensor_prompt_template_hash()
