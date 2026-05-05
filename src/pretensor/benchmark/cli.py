"""``pretensor benchmark`` command group — L1 / L2 / L3 dispatch.

Contract: see ``docs/specs/benchmark/spec.md`` §CLI contract. stdout carries
only the JSON document when ``--out`` is absent; all human-facing text goes
to stderr.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from pretensor.benchmark.results import (
    ComparisonError,
    compare,
    read_json,
)
from pretensor.benchmark.runner import (
    Dataset,
    RunnerKind,
    run_l1,
    run_l2,
    run_l3,
)

__all__ = ["register_benchmark_command"]


# Exit 1 covers both stub run failures and detected regressions — the spec
# §CLI contract collapses them under a single "release-gating assertion
# failed" code; downstream scripts inspect stderr to disambiguate.
_EXIT_FAILED = 1
_EXIT_BAD_INPUT = 2
_DEFAULT_GRAPH_DIR = Path(".pretensor")
_DATASET_HELP = (
    "Fixture key (pagila, tpch, analytics_dwh, adversarial, "
    "saas_multitenant, adventureworks)."
)
_OUT_HELP = "Write JSON output to this path; stdout when omitted."
_GRAPH_DIR_HELP = "Existing indexed graph directory. Defaults to .pretensor."
_BASELINE_HELP = "Path to the baseline benchmark result JSON."
_CURRENT_HELP = (
    "Path to the current benchmark result JSON to compare against the baseline."
)


def register_benchmark_command(app: typer.Typer) -> None:
    """Register ``pretensor benchmark`` onto ``app``.

    The benchmark commands route all human output to a dedicated stderr
    console (the spec reserves stdout for the JSON document). Unlike most
    ``register_*`` callers, this function therefore does not accept the
    shared stdout console from ``cli.main``.
    """
    err_console = Console(stderr=True)

    benchmark_app = typer.Typer(
        name="benchmark",
        help=(
            "Run the 3-level benchmark harness (L1 graph quality, "
            "L2 MCP tool quality, L3 agent task success)."
        ),
        no_args_is_help=True,
    )

    def _handle_not_implemented(exc: NotImplementedError) -> None:
        err_console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(_EXIT_FAILED) from exc

    @benchmark_app.command("l1")
    def l1_command(
        dataset: Dataset = typer.Option(
            ...,
            "--dataset",
            help=_DATASET_HELP,
            case_sensitive=False,
        ),
        out: Path | None = typer.Option(
            None,
            "--out",
            help=_OUT_HELP,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
        graph_dir: Path = typer.Option(
            _DEFAULT_GRAPH_DIR,
            "--graph-dir",
            help=_GRAPH_DIR_HELP,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        embeddings: bool = typer.Option(
            False,
            "--embeddings/--no-embeddings",
            help="Run the Layer-A embeddings path (requires the "
            "'embeddings' optional extra).",
        ),
    ) -> None:
        """Run L1 graph-quality metrics against a fixture."""
        try:
            run_l1(dataset, out, graph_dir, embeddings=embeddings)
        except NotImplementedError as exc:
            _handle_not_implemented(exc)

    @benchmark_app.command("l2")
    def l2_command(
        dataset: Dataset = typer.Option(
            ...,
            "--dataset",
            help=_DATASET_HELP,
            case_sensitive=False,
        ),
        out: Path | None = typer.Option(
            None,
            "--out",
            help=_OUT_HELP,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
        graph_dir: Path = typer.Option(
            _DEFAULT_GRAPH_DIR,
            "--graph-dir",
            help=_GRAPH_DIR_HELP,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        embeddings: bool = typer.Option(
            False,
            "--embeddings/--no-embeddings",
            help="Include the semantic_search Recall@K metric "
            "(requires the 'embeddings' optional extra).",
        ),
    ) -> None:
        """Run L2 MCP-tool-quality metrics against a fixture."""
        try:
            run_l2(dataset, out, graph_dir, embeddings=embeddings)
        except NotImplementedError as exc:
            _handle_not_implemented(exc)

    @benchmark_app.command("l3")
    def l3_command(
        dataset: Dataset = typer.Option(
            ...,
            "--dataset",
            help=_DATASET_HELP,
            case_sensitive=False,
        ),
        out: Path | None = typer.Option(
            None,
            "--out",
            help=_OUT_HELP,
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
        graph_dir: Path = typer.Option(
            _DEFAULT_GRAPH_DIR,
            "--graph-dir",
            help=_GRAPH_DIR_HELP,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        runner: RunnerKind = typer.Option(
            ...,
            "--runner",
            help="Which L3 runner to execute: baseline (agent + raw "
            "schema) or pretensor (agent + MCP).",
            case_sensitive=False,
        ),
        model: str = typer.Option(
            ...,
            "--model",
            help="LLM model identifier (e.g. claude-haiku-4-5).",
        ),
        seed: int | None = typer.Option(
            None,
            "--seed",
            help="Optional integer seed for reproducible agent runs.",
        ),
    ) -> None:
        """Run L3 agent-task-success evaluation against a fixture."""
        try:
            run_l3(
                dataset,
                out,
                graph_dir,
                runner=runner,
                model=model,
                seed=seed,
            )
        except NotImplementedError as exc:
            _handle_not_implemented(exc)

    @benchmark_app.command("compare")
    def compare_command(
        baseline: Path = typer.Option(
            ...,
            "--baseline",
            help=_BASELINE_HELP,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
        current: Path = typer.Option(
            ...,
            "--current",
            help=_CURRENT_HELP,
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
        tolerance: float = typer.Option(
            1e-6,
            "--tolerance",
            help="Numeric tolerance below which metric drift is treated as noise.",
        ),
    ) -> None:
        """Compare two benchmark result JSONs and exit non-zero on regression."""
        try:
            baseline_result = read_json(baseline)
        except (json.JSONDecodeError, KeyError) as exc:
            err_console.print(f"[red]Failed to read baseline: {exc}[/red]")
            raise typer.Exit(_EXIT_BAD_INPUT) from exc
        try:
            current_result = read_json(current)
        except (json.JSONDecodeError, KeyError) as exc:
            err_console.print(f"[red]Failed to read current: {exc}[/red]")
            raise typer.Exit(_EXIT_BAD_INPUT) from exc
        try:
            report = compare(baseline_result, current_result, tolerance=tolerance)
        except ComparisonError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(_EXIT_BAD_INPUT) from exc
        err_console.print(report.format_diff())
        if report.has_regression:
            raise typer.Exit(_EXIT_FAILED)

    app.add_typer(benchmark_app)
