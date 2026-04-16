#!/usr/bin/env python3
"""Benchmark Graph RAG prompt variants for natural-language-to-SQL (Pagila).

Runs three conditions on the same BM25-selected tables per question:

* **A (baseline_schema):** columns + comments only; join paths omitted from prompt.
* **B (roles_only):** roles/classification + join paths; no cluster or Entity node lines.
* **C (enhanced):** B plus cluster label/description and LLM-indexed entity when present.

Requires a Pretensor graph directory (e.g. from ``scripts/e2e_pagila.py``) and LLM
credentials (same env vars as ``suggest_query``). Results are printed as JSON.

Example::

    uv run python scripts/benchmark_graph_rag_nl2sql.py \\
      --graph-dir ~/pretensor-e2e/pagila --database pagila

"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot.errors import ParseError

from pretensor.core.store import KuzuStore
from pretensor.intelligence.llm_runtime import ChatMessage, strip_json_fence
from pretensor.mcp.service_registry import (
    graph_path_for_entry,
    load_registry,
    open_store_for_entry,
    resolve_registry_entry,
)
from pretensor.mcp.tools.suggest_query import (
    GraphRagPromptMode,
    bm25_table_candidates_for_suggest,
    build_http_client_for_suggest_query,
    build_suggest_query_llm_prompt,
    fallback_tables_for_suggest,
)
from pretensor.validation.query_validator import QueryValidator

_MODES: tuple[tuple[str, GraphRagPromptMode], ...] = (
    ("A_baseline_schema", "baseline_schema"),
    ("B_roles_only", "roles_only"),
    ("C_enhanced", "enhanced"),
)

_DEFAULT_QUESTIONS = (
    Path(__file__).resolve().parent / "data" / "pagila_nl2sql_bench.json"
)


def _approx_tokens(text: str) -> int:
    """Rough token estimate when ``tiktoken`` is unavailable."""
    try:
        import tiktoken  # type: ignore[import-untyped]

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _normalize_sql(sql: str) -> str | None:
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except ParseError:
        return None
    return tree.sql(dialect="postgres", normalize=True)


def _sql_matches_expected(generated: str, expected: str) -> bool:
    g = _normalize_sql(generated)
    e = _normalize_sql(expected)
    if g is None or e is None:
        return False
    return g == e


def _parse_llm_sql(raw: str) -> str | None:
    try:
        data = json.loads(strip_json_fence(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    sql = data.get("sql")
    return str(sql).strip() if isinstance(sql, str) and sql.strip() else None


def _load_cases(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Benchmark JSON must be a list of objects")
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        exp = item.get("expected_sql")
        cid = str(item.get("id", ""))
        if isinstance(q, str) and isinstance(exp, str) and q.strip() and exp.strip():
            out.append(
                {
                    "id": cid or f"q{len(out)}",
                    "question": q.strip(),
                    "expected_sql": exp.strip(),
                }
            )
    return out


async def _run_one_mode(
    *,
    store: KuzuStore,
    question: str,
    database_arg: str,
    connection_name: str,
    database_key: str,
    tables_used: list[str],
    mode: GraphRagPromptMode,
    complete: Any,
) -> dict[str, Any]:
    ctx = build_suggest_query_llm_prompt(
        store,
        question=question,
        database_arg=database_arg,
        connection_name=connection_name,
        database_key=database_key,
        graph_rag_prompt_mode=mode,
        tables_used=tables_used,
    )
    if "error" in ctx:
        return {"error": ctx["error"], "approx_prompt_tokens": 0, "sql_correct": False}

    user = str(ctx["user_prompt"])
    system = str(ctx["system_prompt"])
    approx_tokens = _approx_tokens(system + user)

    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
    raw, _usage = await complete(messages)
    sql = _parse_llm_sql(raw)

    return {
        "approx_prompt_tokens": approx_tokens,
        "raw_sql": sql,
        "parse_error": sql is None,
    }


async def _main_async(args: argparse.Namespace) -> int:
    graph_dir = Path(args.graph_dir).expanduser().resolve()
    reg = load_registry(graph_dir)
    entry = resolve_registry_entry(reg, args.database.strip())
    if entry is None:
        print(json.dumps({"error": "Unknown database for registry"}), file=sys.stderr)
        return 2
    gp = graph_path_for_entry(entry)
    if not gp.exists():
        print(json.dumps({"error": f"Graph missing: {gp}"}), file=sys.stderr)
        return 2

    db_key = str(entry.database)
    cn = str(entry.connection_name)
    db_arg = args.database.strip()

    cases = _load_cases(Path(args.questions))
    if not cases:
        print(json.dumps({"error": "No benchmark cases loaded"}), file=sys.stderr)
        return 2

    client = build_http_client_for_suggest_query()
    if client is None and not args.dry_run:
        print(
            json.dumps(
                {
                    "error": "No LLM API key configured (set OPENAI_API_KEY, "
                    "ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or PRETENSOR_LLM_API_KEY). "
                    "Use --dry-run to only measure prompt size."
                }
            ),
            file=sys.stderr,
        )
        return 2

    store = open_store_for_entry(entry)
    validator = QueryValidator(store, connection_name=cn, database_key=db_key)
    results: list[dict[str, Any]] = []

    async def _noop_complete(
        _m: list[ChatMessage],
    ) -> tuple[str, object]:
        return (
            '{"sql": "SELECT 1", "explanation": "dry", "confidence": 0.5}',
            object(),
        )

    try:
        for case in cases:
            q = case["question"]
            exp_sql = case["expected_sql"]
            cid = case["id"]

            tables_used = bm25_table_candidates_for_suggest(
                graph_dir,
                question=q,
                connection_name=cn,
                database_key=db_key,
            )
            if not tables_used:
                tables_used = fallback_tables_for_suggest(
                    store,
                    connection_name=cn,
                    database_key=db_key,
                    question=q,
                )

            row: dict[str, Any] = {
                "id": cid,
                "question": q,
                "tables_used": tables_used,
                "modes": {},
            }

            if args.dry_run:
                llm_call = _noop_complete
            else:
                assert client is not None
                llm_call = client.complete

            for label, mode in _MODES:
                mode_out = await _run_one_mode(
                    store=store,
                    question=q,
                    database_arg=db_arg,
                    connection_name=cn,
                    database_key=db_key,
                    tables_used=tables_used,
                    mode=mode,
                    complete=llm_call,
                )
                sql = mode_out.get("raw_sql")
                valid = False
                if sql:
                    valid = bool(validator.validate(sql).valid)
                correct = bool(sql and _sql_matches_expected(sql, exp_sql))
                row["modes"][label] = {
                    "approx_prompt_tokens": mode_out.get("approx_prompt_tokens"),
                    "graph_validation_pass": valid,
                    "sql_matches_expected_ast": correct,
                    "parse_error": mode_out.get("parse_error"),
                }
                if args.verbose and sql:
                    row["modes"][label]["generated_sql"] = sql

            results.append(row)

    finally:
        store.close()
        if client is not None:
            await client.aclose()

    summary = {
        "database": db_arg,
        "dry_run": args.dry_run,
        "per_question": results,
        "totals": {},
    }
    for label, _ in _MODES:
        tot_tokens = sum(
            int(r["modes"][label]["approx_prompt_tokens"] or 0) for r in results
        )
        tot_match = sum(
            1 for r in results if r["modes"][label].get("sql_matches_expected_ast")
        )
        tot_valid = sum(
            1 for r in results if r["modes"][label].get("graph_validation_pass")
        )
        summary["totals"][label] = {
            "questions": len(results),
            "sum_approx_prompt_tokens": tot_tokens,
            "count_sql_matches_expected_ast": tot_match,
            "count_graph_validation_pass": tot_valid,
        }

    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Pretensor state directory containing registry.json and graphs/",
    )
    parser.add_argument(
        "--database",
        type=str,
        required=True,
        help="Registry connection name or logical database key (e.g. pagila)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=_DEFAULT_QUESTIONS,
        help=f"JSON benchmark file (default: {_DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; still reports approximate prompt token sizes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include generated SQL in per-mode output",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
