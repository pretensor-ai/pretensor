"""System-prompt template + hashing for the L3 NL-to-SQL agent.

The prompt is locked into source so its hash is reproducible across runs:
two invocations with the same dataset and DDL produce the same
``prompt_hash``. Any drift — a wording tweak, a new DDL revision,
a different dataset name — flips the hash and is auditable from the
JSON envelope.
"""

from __future__ import annotations

import hashlib
import re

__all__ = [
    "MAX_DDL_CHARS",
    "build_pretensor_system_prompt",
    "build_system_prompt",
    "pretensor_prompt_template_hash",
    "prompt_hash",
    "prompt_template_hash",
    "strip_sql_fences",
]


_SYSTEM_PROMPT_TEMPLATE = """You are a senior SQL engineer. The schema below describes a {dataset_name} PostgreSQL database. Read the schema carefully, then translate the user's question into a single PostgreSQL SELECT statement that runs against this schema and answers the question.

Rules:
- Return only the SQL query — no markdown, no code fences, no commentary.
- Emit a single statement; do not include trailing semicolons or explanatory text.
- Use only objects (tables, columns, schemas) that appear in the schema.
- Prefer fully-qualified table names (schema.table) when the schema names them that way.
- Read-only access only: SELECT or WITH ... SELECT statements only. Never DDL or DML.

Schema (DDL):
{ddl_text}
"""

MAX_DDL_CHARS = 400_000
"""Soft warning threshold for the DDL passed to the model.

Beyond this size the runner emits a ``notes[]`` warning so an auditor can
tell whether a low success rate might be a context-window artefact rather
than agent quality. The runner never silently truncates — silent
truncation would produce a fake baseline.
"""


def build_system_prompt(dataset_name: str, ddl_text: str) -> str:
    """Render the locked template with the given dataset name and DDL.

    The dataset name is interpolated verbatim — give the agent a hint about
    the schema's domain (e.g. ``pagila`` is a video-rental catalog, ``tpch``
    is a decision-support benchmark) at no cost. The DDL is also verbatim;
    comments inside the DDL carry semantic hints we want preserved.
    """
    return _SYSTEM_PROMPT_TEMPLATE.format(dataset_name=dataset_name, ddl_text=ddl_text)


_PRETENSOR_SYSTEM_PROMPT_TEMPLATE = """You are a senior SQL engineer. The {dataset_name} PostgreSQL database is indexed by Pretensor, and you have access to MCP tools that describe the schema, search for relevant tables, and traverse relationships. Use the tools to gather just enough context, then translate the user's question into a single PostgreSQL SELECT statement.

Rules:
- Return only the SQL query — no markdown, no code fences, no commentary.
- Emit a single statement; do not include trailing semicolons or explanatory text.
- Use only objects (tables, columns, schemas) that the tools confirm exist.
- Prefer fully-qualified table names (schema.table) when the schema names them that way.
- Read-only access only: SELECT or WITH ... SELECT statements only. Never DDL or DML.
- Stop calling tools and emit the SQL as soon as you have enough context. Do not call tools for general curiosity.
"""


def build_pretensor_system_prompt(dataset_name: str) -> str:
    """Render the pretensor-runner system prompt for ``dataset_name``.

    The pretensor prompt is intentionally close to the baseline's: same
    voice, same SQL-emission rules. The only structural change is that
    schema knowledge is delivered through MCP tools rather than inlined
    DDL — so the template tells the model to gather context via tools
    before answering. Keeping the prompts otherwise identical keeps the
    side-by-side comparison about Pretensor's tooling, not about prompt
    engineering.
    """
    return _PRETENSOR_SYSTEM_PROMPT_TEMPLATE.format(dataset_name=dataset_name)


def pretensor_prompt_template_hash() -> str:
    """SHA-256 of the literal pretensor prompt template (no substitutions).

    Mirrors :func:`prompt_template_hash` so the JSON envelope can record
    which prompt the run used; flips when the wording is edited in source.
    """
    return hashlib.sha256(_PRETENSOR_SYSTEM_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def prompt_template_hash() -> str:
    """SHA-256 of the literal template (no substitutions).

    Stable across runs; flips only when the prompt wording changes in
    source. Pair this with ``prompt_hash`` in the JSON envelope so an
    auditor can tell apart "DDL changed but template didn't" from
    "template changed but DDL didn't".
    """
    return hashlib.sha256(_SYSTEM_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def prompt_hash(rendered_prompt: str) -> str:
    """SHA-256 of the rendered prompt (template + dataset + DDL substituted)."""
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()


# Match a leading ```sql ... ``` or ``` ... ``` fence. We intentionally
# only strip the outermost wrapping fence; if the model returns multiple
# code blocks we want to surface that as an SQL parse failure downstream
# rather than silently merge them.
_FENCE_RE = re.compile(
    r"^\s*```(?:sql|postgres|postgresql)?\s*\n(?P<body>.*?)\n```\s*$",
    re.IGNORECASE | re.DOTALL,
)


def strip_sql_fences(raw: str) -> str:
    """Best-effort cleanup of model output that ignored the no-fence rule.

    Some models wrap SQL in ```sql ... ``` despite the explicit
    instruction. This strips one leading fence; everything else is
    returned verbatim so downstream parsing still surfaces real garbage
    as a parse failure.
    """
    match = _FENCE_RE.match(raw)
    if match is None:
        return raw
    return match.group("body")
