from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator
from ruamel.yaml import YAML

from .base import PretensorModel

ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} references with their environment variable values."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(f"Environment variable '{var_name}' is not set")
        return env_value

    return ENV_VAR_PATTERN.sub(_replace, value)


class DatabaseType(StrEnum):
    POSTGRES = "postgres"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"
    DUCKDB = "duckdb"


class PrivacyMode(StrEnum):
    SCHEMA_ONLY = "schema_only"
    SYNTHETIC_SAMPLES = "synthetic_samples"
    REAL_SAMPLES = "real_samples"


class SchemaFilter(PretensorModel):
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class ConnectionConfig(PretensorModel):
    # Mutable config (env resolution, optional connector-specific fields).
    model_config = ConfigDict(frozen=False)

    name: str
    type: DatabaseType
    host: str | None = None
    port: int | str | None = None
    database: str | None = None
    user: str | None = None
    password: str | None = None
    schema_filter: SchemaFilter = Field(default_factory=SchemaFilter)
    metadata_extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_env_vars(self) -> ConnectionConfig:
        for field_name in self.__class__.model_fields:
            raw = getattr(self, field_name)
            if isinstance(raw, str) and ENV_VAR_PATTERN.search(raw):
                object.__setattr__(self, field_name, _resolve_env_vars(raw))
        # Also resolve env vars inside metadata_extra string values.
        for key, val in self.metadata_extra.items():
            if isinstance(val, str) and ENV_VAR_PATTERN.search(val):
                self.metadata_extra[key] = _resolve_env_vars(val)
        # Coerce port back to int after env var substitution (${PORT} → "5432").
        raw_port = getattr(self, "port")
        if isinstance(raw_port, str):
            object.__setattr__(self, "port", int(raw_port))
        return self


class LLMProvider(StrEnum):
    """Registered LLM backends (see `LLMConfig.provider`)."""

    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class LLMStepKey(StrEnum):
    """Pipeline step identifiers for per-step LLM overrides (`llm_steps`)."""

    SEMANTIC = "semantic"
    PLANNER = "planner"
    CODEGEN_QUERIES = "codegen_queries"
    CODEGEN_PAGES = "codegen_pages"
    VALIDATION = "validation"
    REPAIR = "repair"


class StepLLMOverride(PretensorModel):
    """Partial LLM settings merged over the global `llm` block for one step.

    Omitted fields keep the global default. ``http_headers`` replaces the
    global header map when set (non-``None``).
    """

    provider: LLMProvider | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    base_url: str | None = None
    http_headers: dict[str, str] | None = None


class LLMConfig(PretensorModel):
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096
    base_url: str | None = None
    http_headers: dict[str, str] = Field(default_factory=dict)

    def merged_with(self, override: StepLLMOverride | None) -> LLMConfig:
        """Return a copy with non-``None`` fields from ``override`` applied."""
        if override is None:
            return self.model_copy(deep=True)
        data = self.model_dump()
        if override.provider is not None:
            data["provider"] = override.provider
        if override.model is not None:
            data["model"] = override.model
        if override.temperature is not None:
            data["temperature"] = override.temperature
        if override.max_tokens is not None:
            data["max_tokens"] = override.max_tokens
        if override.base_url is not None:
            data["base_url"] = override.base_url
        if override.http_headers is not None:
            data["http_headers"] = dict(override.http_headers)
        return LLMConfig.model_validate(data)


class SyntheticOptions(PretensorModel):
    hash_strings: bool = True
    randomize_numbers: bool = True
    preserve_magnitude: bool = True


class PrivacyConfig(PretensorModel):
    mode: PrivacyMode = PrivacyMode.SCHEMA_ONLY
    synthetic: SyntheticOptions = Field(default_factory=SyntheticOptions)


class MCPConfig(PretensorModel):
    enabled: bool = False
    url: str = "https://evidence.studio/mcp"


class EvidenceConfig(PretensorModel):
    mcp: MCPConfig = Field(default_factory=MCPConfig)


class RepairConfig(PretensorModel):
    max_code_retries: int = 3
    max_plan_retries: int = 2
    max_total_iterations: int = 10


class StagingConfig(PretensorModel):
    annotations_enabled: bool = True
    staging_env_var: str = "PRETENSOR_STAGING"


class ValidationConfig(PretensorModel):
    threshold: int = 7
    top_components_check: int = 2
    repair: RepairConfig = Field(default_factory=RepairConfig)
    staging: StagingConfig = Field(default_factory=StagingConfig)
    # Layer 1 — skip LLM self-review when SQL lint, page markdown, and Evidence
    # build (if run) produced no ERROR issues; still run when any deterministic
    # step reports ERROR.
    l1_skip_llm_self_review_on_clean_deterministic: bool = True
    # Layer 3 — deterministic structure vs plan + optional presentation hints.
    l3_structural_checks_enabled: bool = True
    l3_orphan_chart_severity: Literal["error", "warning"] = "error"
    l3_structural_presentation_warnings: bool = True


class ProjectConfig(PretensorModel):
    project_name: str | None = None
    connections: list[ConnectionConfig] | dict[str, ConnectionConfig] = Field(
        default_factory=list
    )
    llm: LLMConfig = Field(default_factory=LLMConfig)
    llm_steps: dict[LLMStepKey, StepLLMOverride] = Field(default_factory=dict)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @model_validator(mode="before")
    @classmethod
    def _preprocess_config(cls, data: dict) -> dict:
        """Handle legacy and shorthand config formats."""
        if not isinstance(data, dict):
            return data

        # Infer project_name if missing
        if not data.get("project_name"):
            data["project_name"] = Path.cwd().name

        # Handle connections as dict
        connections = data.get("connections")
        if isinstance(connections, dict):
            connection_list = []
            for name, config in connections.items():
                if isinstance(config, dict):
                    config["name"] = name
                    connection_list.append(config)
            data["connections"] = connection_list

        return data

    def resolved_llm_for_step(self, step: LLMStepKey) -> LLMConfig:
        """Global ``llm`` merged with ``llm_steps[step]`` when present."""
        override = self.llm_steps.get(step)
        return self.llm.merged_with(override)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> ProjectConfig:
        yaml = YAML()
        data = yaml.load(yaml_str)
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: Path) -> ProjectConfig:
        """Load project config from a YAML file on disk."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return cls.from_yaml(path.read_text())
