"""Individual wizard steps.

Every function takes a ``Prompter`` and returns a value. None of them writes to
disk, so the wizard controls all side effects in one place.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pretensor.cli.init.inference import (
    DIALECTS,
    InferredDsn,
    InferredRepo,
    assemble_bigquery_dsn,
    assemble_mysql_dsn,
    assemble_postgres_dsn,
    assemble_snowflake_dsn,
)
from pretensor.cli.init.mcp_clients import McpClient
from pretensor.cli.prompts import Prompter
from pretensor.enrichment.analyze.walker import supported_extensions
from pretensor.introspection.models.dsn import redact_dsn

__all__ = [
    "ask_add_another_database",
    "ask_compute_embeddings",
    "choose_clients",
    "collect_dsn",
    "collect_repository",
    "repo_skip_message",
]

_ENTRY_OPTIONS = ("Paste a full connection URL", "Enter the details one by one")


def collect_dsn(
    prompter: Prompter,
    inferred: InferredDsn | None,
    *,
    dialect: str | None = None,
) -> InferredDsn:
    """Return the DSN to use, offering the inferred value first.

    ``dialect``, when given (typically from ``--dialect``), pre-selects the
    build-from-parts dialect and skips the "Which database?" question.
    """
    if inferred is not None:
        source = f" (from {inferred.env_var})" if inferred.env_var else ""
        question = f"Use {redact_dsn(inferred.dsn)}{source}?"
        if prompter.confirm(question, default=True):
            return inferred

    if prompter.choose("How do you want to connect?", _ENTRY_OPTIONS) == 0:
        return InferredDsn(dsn=prompter.text("Connection URL"), env_var=None)

    if dialect is not None:
        chosen_dialect = dialect
    else:
        index = prompter.choose("Which database?", DIALECTS)
        chosen_dialect = DIALECTS[index]

    return InferredDsn(
        dsn=_build_dsn_from_parts(prompter, chosen_dialect), env_var=None
    )


def _build_dsn_from_parts(prompter: Prompter, dialect: str) -> str:
    """Collect fields for one dialect and assemble its DSN.

    Field sets and DSN shapes mirror the parsers in
    ``introspection.models.dsn`` exactly (``_config_from_postgres_url``,
    ``_config_from_mysql_url``, ``_config_from_snowflake_url``,
    ``_config_from_bigquery_url``).
    """
    if dialect == "postgres":
        host = prompter.text("Host", default="localhost")
        port = prompter.text("Port", default="5432")
        user = prompter.text("User")
        database = prompter.text("Database")
        password = prompter.secret("Password")
        return assemble_postgres_dsn(
            user=user, password=password, host=host, port=port, database=database
        )
    if dialect == "mysql":
        host = prompter.text("Host", default="localhost")
        port = prompter.text("Port", default="3306")
        user = prompter.text("User")
        database = prompter.text("Database")
        password = prompter.secret("Password")
        return assemble_mysql_dsn(
            user=user, password=password, host=host, port=port, database=database
        )
    if dialect == "snowflake":
        account = prompter.text("Account")
        user = prompter.text("User")
        password = prompter.secret("Password")
        database = prompter.text("Database")
        warehouse = prompter.text("Warehouse (optional)", default="")
        role = prompter.text("Role (optional)", default="")
        return assemble_snowflake_dsn(
            account=account,
            user=user,
            password=password,
            database=database,
            warehouse=warehouse,
            role=role,
        )
    if dialect == "bigquery":
        project = prompter.text("Project")
        dataset = _prompt_bigquery_dataset(prompter)
        location = prompter.text("Location (optional)", default="")
        return assemble_bigquery_dsn(
            project=project, dataset=dataset, location=location
        )

    msg = f"Unknown dialect: {dialect!r}; expected one of: {', '.join(DIALECTS)}"
    raise ValueError(msg)


def _prompt_bigquery_dataset(prompter: Prompter) -> str:
    """Ask for the BigQuery dataset, a required field, re-asking once on blank.

    ``_config_from_bigquery_url`` rejects a DSN without a dataset, and a
    blank answer here would otherwise waste the rest of the wizard flow
    (dataset, location, repo linking, client selection, plan confirmation)
    before that failure surfaces. A single re-ask catches the common
    "typed enter too fast" case; a second blank answer is returned as-is
    and falls through to the existing "must include project and dataset"
    error at connection time.
    """
    dataset = prompter.text("Dataset")
    if dataset:
        return dataset
    return prompter.text("BigQuery requires a dataset. Dataset")


def repo_skip_message(path: Path) -> str:
    """Explain that a repository has nothing analyzable, without sounding broken."""
    languages = ", ".join(supported_extensions())
    return (
        f"No supported source files found under {path}.\n"
        f"`analyze` currently scans {languages}. Skipping repository linking."
    )


def collect_repository(
    prompter: Prompter,
    inferred: InferredRepo | None,
    cwd: Path,
) -> Path | None:
    """Return a repository path to analyze, or None to skip the step."""
    if inferred is not None:
        question = (
            f"Link {inferred.path} ({inferred.file_count} source files) "
            "so agents can see which code reads each table?"
        )
        if prompter.confirm(question, default=True):
            return inferred.path

    if not prompter.confirm("Link a different code repository?", default=False):
        return None
    answer = prompter.text("Repository path", default=str(cwd)).strip()
    return Path(answer).expanduser().resolve() if answer else None


def choose_clients(
    prompter: Prompter,
    candidates: Sequence[McpClient],
) -> tuple[McpClient, ...]:
    """Ask per client whether to register pretensor with it."""
    chosen: list[McpClient] = []
    for client in candidates:
        if prompter.confirm(f"Register pretensor with {client.label}?", default=True):
            chosen.append(client)
    return tuple(chosen)


def ask_compute_embeddings(prompter: Prompter) -> bool:
    """Ask once, in the guided flow, whether to compute table embeddings.

    Only called when the ``[embeddings]`` extra is installed; the caller is
    responsible for that gate, and for skipping this question entirely under
    ``--yes``, non-interactive, and the express-accept fast path.
    """
    return prompter.confirm(
        "Compute table embeddings for semantic search?", default=True
    )


def ask_add_another_database(prompter: Prompter) -> bool:
    """Ask whether to index one more database connection."""
    return prompter.confirm("Add another database?", default=False)
