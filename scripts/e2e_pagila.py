#!/usr/bin/env python3
"""
End-to-end smoke test for Pretensor Graph against a local Pagila (PostgreSQL) database.

Usage:
    # Minimal (no LLM):
    uv run python scripts/e2e_pagila.py --dsn "postgresql://postgres:postgres@localhost:5432/pagila"

    # With LLM (Anthropic):
    ANTHROPIC_API_KEY=sk-... uv run python scripts/e2e_pagila.py --dsn "postgresql://..." --llm

    # Custom state dir (keeps graph separate from project state):
    uv run python scripts/e2e_pagila.py --dsn "postgresql://..." --state-dir /tmp/pagila-test

    # Skip re-indexing (reuse existing graph):
    uv run python scripts/e2e_pagila.py --dsn "postgresql://..." --skip-index

Sections:
    1. Index          — build the Kuzu graph from Pagila schema (always heuristic; LLM optional)
    2. List           — verify the registry entry
    3. MCP tools      — exercise list_databases, query, context, cypher, suggest_query
    4. MCP resources  — databases, overview, metrics markdown resources
    5. suggest_query  — no-LLM path (returns tables/join_paths); metric template short-circuit
    6. LLM (optional) — suggest_query with real SQL generation if --llm is set
    7. Reindex        — dry-run diff and live reindex cycle
    8. MCP config     — print the mcpServers snippet for Claude / Cursor
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(title: str) -> None:
    width = 70
    print(f"\n{BOLD}{CYAN}{'━' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'━' * width}{RESET}")


def step(label: str) -> None:
    print(f"\n{BOLD}▶ {label}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def dump(obj: object, indent: int = 4) -> None:
    """Pretty-print a dict/list, truncating long string values."""
    def _clip(o: object, depth: int = 0) -> object:
        if isinstance(o, dict):
            return {k: _clip(v, depth + 1) for k, v in o.items()}
        if isinstance(o, list):
            clipped = [_clip(i, depth + 1) for i in o[:5]]
            if len(o) > 5:
                clipped.append(f"... ({len(o) - 5} more)")
            return clipped
        if isinstance(o, str) and len(o) > 200:
            return o[:200] + "…"
        return o

    text = json.dumps(_clip(obj), indent=indent, default=str)
    for line in text.splitlines():
        print(f"  {DIM}{line}{RESET}")


# ---------------------------------------------------------------------------
# CLI runner helper
# ---------------------------------------------------------------------------

def run_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["uv", "run", "pretensor", *args]
    info("$ " + " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=False,  # let output stream to terminal
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    if check and result.returncode != 0:
        fail(f"Command exited {result.returncode}")
        sys.exit(result.returncode)
    return result


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pretensor Graph end-to-end smoke test against Pagila."
    )
    p.add_argument(
        "--dsn",
        default=os.environ.get("PAGILA_DSN", "postgresql://postgres:postgres@localhost:5432/pagila"),
        help="PostgreSQL DSN for Pagila (default: env PAGILA_DSN or localhost:5432/pagila)",
    )
    p.add_argument(
        "--state-dir",
        default=os.path.expanduser("~/pretensor-e2e/pagila"),
        help="State directory for this test run (default: ~/pretensor-e2e/pagila). "
             "Kept outside the project so skill files and graph data don't mix with "
             "the core .pretensor/, .claude/skills/, and .cursor/rules/.",
    )
    p.add_argument(
        "--name",
        default="pagila",
        help="Connection name to register (default: pagila)",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="Run LLM-backed indexing after the heuristic pass (requires API key).",
    )
    p.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing — reuse an existing graph in --state-dir.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete --state-dir before starting (full fresh run).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# MCP tool helpers (call Python API directly, no stdio server needed)
# ---------------------------------------------------------------------------

def _graph_dir(state_dir: str) -> Path:
    return Path(state_dir).resolve()


def run_list_databases(graph_dir: Path) -> list[dict]:
    from pretensor.mcp.service import list_databases_payload
    result = list_databases_payload(graph_dir)
    # payload returns {"databases": [...]}
    return result.get("databases", result) if isinstance(result, dict) else result


def run_query(graph_dir: Path, question: str, db: str | None = None) -> list[dict]:
    from pretensor.mcp.service import query_payload
    result = query_payload(graph_dir, q=question, db=db)
    return result.get("results", result) if isinstance(result, dict) else result


def run_context(graph_dir: Path, table: str, db: str | None = None, detail: str = "standard") -> dict:
    from pretensor.mcp.service import context_payload
    return context_payload(graph_dir, table=table, db=db, detail=detail)


def run_cypher(graph_dir: Path, query: str, database: str) -> dict:
    from pretensor.mcp.service import cypher_payload
    return cypher_payload(graph_dir, query=query, database=database)


def run_suggest_query(graph_dir: Path, question: str, database: str) -> dict:
    from pretensor.mcp.service import suggest_query_payload
    return asyncio.run(suggest_query_payload(graph_dir, question=question, database=database))


def run_databases_resource(graph_dir: Path) -> str:
    from pretensor.mcp.service import databases_resource_markdown
    return databases_resource_markdown(graph_dir)


def run_overview_resource(graph_dir: Path, name: str) -> str:
    from pretensor.mcp.service import db_overview_resource_markdown
    return db_overview_resource_markdown(graph_dir, name)


def run_metrics_resource(graph_dir: Path, name: str) -> str:
    from pretensor.mcp.service import metrics_resource_markdown
    return metrics_resource_markdown(graph_dir, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    graph_dir = _graph_dir(args.state_dir)
    state_dir = args.state_dir
    dsn = args.dsn
    name = args.name

    print(f"\n{BOLD}Pretensor Graph — End-to-End Smoke Test{RESET}")
    print(f"  DSN        : {dsn}")
    print(f"  State dir  : {graph_dir}")
    print(f"  Connection : {name}")
    print(f"  LLM        : {'yes' if args.llm else 'no (heuristic only)'}")

    # ── Clean ────────────────────────────────────────────────────────────────
    if args.clean and graph_dir.exists():
        step("Cleaning state directory")
        shutil.rmtree(graph_dir)
        ok(f"Removed {graph_dir}")

    # =========================================================================
    header("1 · Index — build Kuzu graph from Pagila schema")
    # =========================================================================
    #
    # What happens without --llm:
    #   • Connector introspects schema → SchemaTable, SchemaColumn, FK edges
    #   • Heuristic table classification (dim/fact signals from column names/FKs)
    #   • Leiden clustering → Cluster nodes
    #   • FK + BFS join-path precomputation → JoinPath nodes
    #   • MetricTemplateBuilder → MetricTemplate nodes from fact-classified tables
    #   • Skill file written to --state-dir/../.claude/skills/pretensor-pagila.md
    #
    # What --llm adds on top:
    #   • LLM relationship inference (inferred join edges)
    #   • LLM entity extraction → Entity nodes + REPRESENTS edges
    #   • LLM cluster labeling (richer display names and descriptions)
    #   • LLM table classification (overrides heuristic role where confident)

    if not args.skip_index:
        step("Running heuristic index (no LLM)")
        # Write the skill file inside the state dir so it stays isolated from the
        # project's own .claude/skills/ and .cursor/rules/ directories.
        # (--skills-target with a file path writes a single file at that path.)
        skill_file = str(graph_dir / f"skills/pretensor-{name}.md")
        index_args = [
            "index", dsn,
            "--name", name,
            "--state-dir", state_dir,
            "--skills-target", skill_file,
        ]
        run_cli(*index_args)
        ok("Heuristic index complete")
        info(f"Skill file: {skill_file}")

        if args.llm:
            step("Re-running with --llm for entity extraction and richer labels")
            llm_args = [
                "index", dsn,
                "--name", name,
                "--state-dir", state_dir,
                "--llm",
                "--skills-target", skill_file,
            ]
            run_cli(*llm_args)
            ok("LLM-backed index complete")
    else:
        warn("Skipping index — using existing graph in state dir")

    # =========================================================================
    header("2 · List — verify registry")
    # =========================================================================

    step("pretensor list")
    run_cli("list", "--state-dir", state_dir)

    step("Python API: list_databases_payload")
    databases = run_list_databases(graph_dir)
    if databases:
        for db in databases:
            ok(f"{db.get('name')} — {db.get('table_count', '?')} tables, "
               f"indexed: {db.get('last_indexed', '?')}, stale={db.get('is_stale')}")
    else:
        fail("No databases returned from list_databases_payload")
        sys.exit(1)

    # =========================================================================
    header("3 · MCP Tools (no LLM required)")
    # =========================================================================

    # ── 3a: query (BM25 search) ──────────────────────────────────────────────
    step("query — BM25 search: 'film rental inventory'")
    hits = run_query(graph_dir, "film rental inventory", db=name)
    if hits:
        ok(f"Got {len(hits)} search hits")
        for h in hits[:4]:
            info(f"  {h.get('node_type')} · {h.get('name')} (score={h.get('score', 0):.3f})")
    else:
        warn("No BM25 hits — check that the FTS index was built")

    # ── 3b: context ──────────────────────────────────────────────────────────
    step("context — 'film' table (standard detail)")
    ctx = run_context(graph_dir, "film", db=name, detail="standard")
    if "error" in ctx:
        fail(f"context error: {ctx['error']}")
    else:
        ok(f"Table: {ctx.get('qualified_name')} — {len(ctx.get('columns', []))} columns")
        cols = ctx.get("columns", [])[:5]
        for c in cols:
            pk = " [PK]" if c.get("is_primary_key") else ""
            fk = " [FK]" if c.get("is_foreign_key") else ""
            null = " nullable" if c.get("nullable") else " NOT NULL"
            info(f"    {c.get('column_name')}: {c.get('data_type')}{pk}{fk}{null}")
        if ctx.get("role"):
            ok(f"Classifier role: {ctx['role']} (confidence={ctx.get('role_confidence', 0):.2f})")
        if ctx.get("cluster"):
            ok(f"Cluster: {ctx['cluster'].get('label', '?')}")
        if ctx.get("relationships"):
            ok(f"Relationships: {len(ctx['relationships'])} (FK + inferred)")

    step("context — 'rental' table (full detail)")
    ctx_full = run_context(graph_dir, "rental", db=name, detail="full")
    if "error" not in ctx_full:
        ok(f"Full detail: {ctx_full.get('qualified_name')} — "
           f"{len(ctx_full.get('columns', []))} columns, "
           f"{len(ctx_full.get('relationships', []))} relationships")
        if ctx_full.get("classification_signals"):
            info(f"  signals: {ctx_full['classification_signals'][:3]}")
    else:
        fail(str(ctx_full.get("error")))

    # ── 3c: cypher ───────────────────────────────────────────────────────────
    step("cypher — MATCH (t:SchemaTable) RETURN t.table_name LIMIT 10")
    cypher_result = run_cypher(
        graph_dir,
        "MATCH (t:SchemaTable) RETURN t.table_name ORDER BY t.table_name LIMIT 10",
        database=name,
    )
    if "rows" in cypher_result:
        rows = cypher_result["rows"]
        ok(f"Cypher returned {len(rows)} rows")
        for r in rows:
            info(f"  {r.get('t.table_name')}")
    else:
        fail(f"cypher error: {cypher_result.get('error')}")

    step("cypher — inspect MetricTemplate nodes")
    metric_cypher = run_cypher(
        graph_dir,
        "MATCH (m:MetricTemplate) RETURN m.name, m.validated, m.dialect ORDER BY m.name LIMIT 10",
        database=name,
    )
    if "rows" in metric_cypher:
        mrows = metric_cypher["rows"]
        if mrows:
            ok(f"MetricTemplate nodes: {len(mrows)}")
            for r in mrows:
                ok(f"  {r.get('m.name')} — validated={r.get('m.validated')} dialect={r.get('m.dialect')}")
        else:
            warn("No MetricTemplate nodes found — tables may not be classified as 'fact' yet")
    else:
        fail(f"cypher error: {metric_cypher.get('error')}")

    step("cypher — mutation blocked (safety check)")
    bad = run_cypher(graph_dir, "CREATE (n:Foo {x: 1})", database=name)
    if "error" in bad:
        ok(f"Mutation correctly rejected: {bad['error'][:80]}")
    else:
        fail("Mutation was NOT rejected — read-only guard missing")

    # ── 3d: suggest_query without LLM ────────────────────────────────────────
    step("suggest_query — 'total revenue' (no LLM: expects metric_template hit or table context)")
    sq_revenue = run_suggest_query(graph_dir, "total revenue", database=name)
    if sq_revenue.get("source") == "metric_template":
        ok(f"MetricTemplate short-circuit! SQL: {sq_revenue.get('sql')}")
        ok(f"  metric: {sq_revenue.get('metric_name')} | stale={sq_revenue.get('metric_stale', False)}")
    elif sq_revenue.get("error") and "LLM" in sq_revenue.get("error", ""):
        ok("No LLM configured — got expected error with table/join context")
        info(f"  tables_used: {sq_revenue.get('tables_used', [])}")
        info(f"  join_paths:  {len(sq_revenue.get('join_paths', []))} paths available")
    elif sq_revenue.get("error"):
        warn(f"suggest_query returned error: {sq_revenue['error']}")
    else:
        ok(f"Got SQL: {sq_revenue.get('sql')}")

    step("suggest_query — 'list all films with their categories' (no LLM)")
    sq_films = run_suggest_query(graph_dir, "list all films with their categories", database=name)
    if sq_films.get("source") == "metric_template":
        ok(f"MetricTemplate hit: {sq_films.get('sql')}")
    elif "tables_used" in sq_films:
        ok(f"Tables identified: {sq_films.get('tables_used', [])}")
        info(f"  join_paths: {len(sq_films.get('join_paths', []))} paths")
        if sq_films.get("sql"):
            ok(f"  SQL: {sq_films['sql'][:120]}")
    else:
        info(str(sq_films))

    # =========================================================================
    header("4 · MCP Resources")
    # =========================================================================

    step("pretensor://databases — registry overview")
    db_md = run_databases_resource(graph_dir)
    lines = [line for line in db_md.splitlines() if line.strip()]
    ok(f"Databases resource: {len(lines)} non-empty lines")
    for line in lines[:6]:
        info(f"  {line}")

    step(f"pretensor://db/{name}/overview")
    ov_md = run_overview_resource(graph_dir, name)
    lines = [line for line in ov_md.splitlines() if line.strip()]
    ok(f"Overview resource: {len(lines)} lines")
    for line in lines[:8]:
        info(f"  {line}")

    step(f"pretensor://db/{name}/metrics")
    m_md = run_metrics_resource(graph_dir, name)
    if "MetricTemplate" in m_md and "No `MetricTemplate`" not in m_md:
        # Count ## headers = number of templates
        n_templates = m_md.count("\n## `")
        ok(f"Metrics resource: {n_templates} MetricTemplate(s) listed")
    else:
        warn("No MetricTemplate nodes in metrics resource (need fact-classified tables)")
    for line in m_md.splitlines()[:12]:
        info(f"  {line}")

    # =========================================================================
    header("5 · Reindex — dry-run then live")
    # =========================================================================

    step("reindex --dry-run (no writes)")
    run_cli("reindex", dsn, "--database", name, "--state-dir", state_dir, "--dry-run")

    step("reindex (live — schema should be unchanged, produces no diff)")
    skill_file = str(graph_dir / f"skills/pretensor-{name}.md")
    run_cli("reindex", dsn, "--database", name, "--state-dir", state_dir,
            "--skills-target", skill_file)
    ok("Reindex complete")

    step("pretensor list (post-reindex)")
    run_cli("list", "--state-dir", state_dir)

    # =========================================================================
    header("6 · LLM suggest_query (skipped if no API key)" if not args.llm else
           "6 · LLM suggest_query")
    # =========================================================================

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("PRETENSOR_LLM_API_KEY")
    )

    if not api_key:
        warn("No LLM API key found in environment — skipping LLM suggest_query test.")
        warn("Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY to enable.")
    else:
        # Import the LLM runtime directly so we can drive suggest_query with a real client.
        from pretensor.cli.llm_options import (
            effective_llm_base_url,
            effective_llm_model,
            effective_llm_provider,
        )
        from pretensor.intelligence.llm_runtime import (
            build_http_completion,
            env_api_key_for_provider,
        )
        from pretensor.mcp.service import suggest_query_payload

        provider = effective_llm_provider(None)
        model = effective_llm_model(None)
        base_url = effective_llm_base_url(None)
        key = env_api_key_for_provider(provider)

        client = build_http_completion(
            provider=provider,
            api_key=key or "",
            model=model,
            base_url=base_url,
        )

        questions = [
            "List the top 10 films by rental count",
            "Which customers have rented more than 30 films?",
            "Show total payments per store",
        ]

        for q in questions:
            step(f"suggest_query (LLM): {q!r}")
            out = asyncio.run(
                suggest_query_payload(graph_dir, question=q, database=name, client=client)
            )
            if out.get("sql"):
                ok(f"SQL [{out.get('validation_status')}]: {out['sql'][:120]}")
                info(f"  confidence={out.get('confidence')} explanation={out.get('explanation','')[:80]}")
            elif out.get("source") == "metric_template":
                ok(f"MetricTemplate: {out.get('sql')}")
            else:
                warn(f"No SQL: {out.get('error','?')[:120]}")

        asyncio.run(client.aclose()) if hasattr(client, "aclose") else None

    # =========================================================================
    header("7 · MCP Server config (for Claude / Cursor)")
    # =========================================================================

    step("pretensor serve --config-only")
    result = subprocess.run(
        ["uv", "run", "pretensor", "serve", "--config-only", "--graph-dir", state_dir],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    if result.returncode == 0:
        ok("mcpServers config snippet:")
        for line in result.stdout.splitlines():
            print(f"  {CYAN}{line}{RESET}")
    else:
        warn("Could not generate MCP config")
        info(result.stderr[:300])

    # =========================================================================
    header("8 · Kuzu Explorer — visual graph browser")
    # =========================================================================
    #
    # Kuzu Explorer is an official web UI that connects directly to the .kuzu
    # file and lets you run Cypher queries with a visual graph renderer.

    kuzu_file = graph_dir / "graphs" / f"{name}.kuzu"
    explorer_port = 8888
    explorer_url = f"http://localhost:{explorer_port}"

    if not kuzu_file.exists():
        warn(f"Graph file not found at {kuzu_file} — skipping Explorer launch.")
    elif shutil.which("docker") is None:
        warn("docker not found in PATH — cannot launch Kuzu Explorer automatically.")
        info("Install Docker Desktop and re-run, or launch manually:")
        info(f"  docker run -p {explorer_port}:{explorer_port} \\")
        info(f"    -v {kuzu_file}:/database --rm kuzudb/explorer:latest")
    else:
        step("Checking for running Kuzu Explorer container")
        existing = subprocess.run(
            ["docker", "ps", "--filter", f"publish={explorer_port}", "--format", "{{.ID}}"],
            capture_output=True, text=True,
        )
        if existing.stdout.strip():
            ok(f"Kuzu Explorer already running → {explorer_url}")
        else:
            step(f"Launching Kuzu Explorer on port {explorer_port} (background)")
            info(f"  Graph: {kuzu_file}")
            # Mount the parent graphs/ directory and tell Explorer which file to open
            # via KUZU_FILE — mounting the .kuzu file directly doesn't work.
            subprocess.Popen(
                [
                    "docker", "run", "--rm",
                    "-p", f"{explorer_port}:8000",
                    "-e", f"KUZU_FILE={kuzu_file.name}",
                    "-v", f"{kuzu_file.parent}:/database",
                    "--name", "pretensor-kuzu-explorer",
                    "kuzudb/explorer:latest",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait briefly for the container to start
            import time
            time.sleep(5)
            ok(f"Kuzu Explorer running → {BOLD}{explorer_url}{RESET}")

        print(f"""
  {CYAN}Useful starter queries in the Explorer:{RESET}

  {DIM}-- Schema: tables and FK connections{RESET}
  MATCH (a:SchemaTable)-[r:FK_REFERENCES]->(b:SchemaTable)
  RETURN a, r, b LIMIT 60

  {DIM}-- Clusters: which tables belong together{RESET}
  MATCH (t:SchemaTable)-[:IN_CLUSTER]->(c:Cluster)
  RETURN t, c LIMIT 80

  {DIM}-- Metric templates and their source tables{RESET}
  MATCH (m:MetricTemplate)-[:METRIC_DEPENDS]->(t:SchemaTable)
  RETURN m, t

  {DIM}-- Columns for a specific table{RESET}
  MATCH (t:SchemaTable {{table_name: 'film'}})-[:HAS_COLUMN]->(c:SchemaColumn)
  RETURN t, c

  {DIM}-- Stop the container when done:{RESET}
  docker stop pretensor-kuzu-explorer
""")

    # =========================================================================
    header("Done")
    # =========================================================================
    print(f"\n{GREEN}{BOLD}All sections complete.{RESET}")
    print(textwrap.dedent(f"""
    Next steps:
      • Visual graph  : {explorer_url}  (Kuzu Explorer — stop with: docker stop pretensor-kuzu-explorer)
      • MCP config    : add the mcpServers snippet above to your Claude / Cursor config
      • Graph dir     : {state_dir}
      • Try 'suggest_query' from an agent with a real question about your data.
      • Run with --llm for richer entity links and cluster labels.
    """))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    main()
