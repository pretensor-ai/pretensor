"""CLI config-file loading and option precedence helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from click.core import ParameterSource
from ruamel.yaml import YAML

from pretensor.config import GraphConfig

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "CliConfigError",
    "ConnectionDefaults",
    "LlmDefaults",
    "PretensorCliConfig",
    "SourceConfig",
    "VisibilityDefaults",
    "get_cli_config",
    "load_cli_config",
    "resolve_optional_path_option",
    "resolve_optional_str_option",
    "resolve_path_option",
]

DEFAULT_CONFIG_PATH = Path(".pretensor") / "config.yaml"


class CliConfigError(ValueError):
    """Raised when the CLI config file is unreadable or invalid."""


@dataclass(frozen=True, slots=True)
class LlmDefaults:
    """LLM defaults accepted by config for Cloud-compatible workflows."""

    model: str | None = None
    budget_usd: float | None = None


@dataclass(frozen=True, slots=True)
class VisibilityDefaults:
    """Visibility defaults for ``index`` and ``serve``."""

    path: Path | None = None
    profile: str | None = None


@dataclass(frozen=True, slots=True)
class ConnectionDefaults:
    """Default connection settings for DSN-driven commands."""

    name: str | None = None
    dialect: str | None = None


_SOURCE_ALLOWED_KEYS = frozenset(
    {
        "dialect",
        "host",
        "port",
        "user",
        "password",
        "database",
        # Snowflake
        "account",
        "schema",
        "warehouse",
        "role",
        # BigQuery
        "project",
        "dataset",
        "location",
    }
)


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Declarative database source definition from ``sources:`` config block."""

    dialect: str
    host: str | None = None
    port: int | None = None
    user: str | None = None
    password: str | None = None
    database: str | None = None
    # Snowflake
    account: str | None = None
    schema: str | None = None
    warehouse: str | None = None
    role: str | None = None
    # BigQuery
    project: str | None = None
    dataset: str | None = None
    location: str | None = None


@dataclass(frozen=True, slots=True)
class PretensorCliConfig:
    """Normalized runtime config loaded from YAML (or empty defaults)."""

    source_path: Path | None = None
    state_dir: Path | None = None
    graph: GraphConfig = GraphConfig()
    llm: LlmDefaults = LlmDefaults()
    visibility: VisibilityDefaults = VisibilityDefaults()
    connection_defaults: ConnectionDefaults = ConnectionDefaults()
    sources: dict[str, SourceConfig] = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        """Whether this config came from a file on disk."""
        return self.source_path is not None


def _as_mapping(raw: Any, *, field: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise CliConfigError(f"`{field}` must be a mapping in config YAML")
    return raw


def _resolved_path(raw: Any, *, field: str, base_dir: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise CliConfigError(f"`{field}` must be a non-empty string path")
    value = Path(raw.strip()).expanduser()
    if not value.is_absolute():
        value = (base_dir / value).resolve()
    else:
        value = value.resolve()
    return value


def _resolved_optional_path(raw: Any, *, field: str, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    return _resolved_path(raw, field=field, base_dir=base_dir)


def _normalized_optional_str(raw: Any, *, field: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise CliConfigError(f"`{field}` must be a string when provided")
    value = raw.strip()
    return value or None


def _normalized_optional_float(raw: Any, *, field: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise CliConfigError(f"`{field}` must be a number when provided")
    if isinstance(raw, (int, float)):
        return float(raw)
    raise CliConfigError(f"`{field}` must be a number when provided")


def _graph_config_from_mapping(raw: Any) -> GraphConfig:
    data = _as_mapping(raw, field="graph")
    allowed = {
        "stale_index_warning_days",
        "clustering_resolution_override",
        "join_path_max_depth",
        "min_cluster_size_merge",
        "collapse_shadow_aliases",
        "lineage_in_max_for_alias",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise CliConfigError(f"Unknown `graph` config key(s): {joined}")
    try:
        return GraphConfig(**data)
    except TypeError as exc:
        raise CliConfigError(f"Invalid `graph` config: {exc}") from exc


def _source_config_from_mapping(
    name: str,
    raw: Any,
) -> SourceConfig:
    """Validate and build a :class:`SourceConfig` from a raw YAML mapping."""
    if not isinstance(raw, dict):
        raise CliConfigError(f"Source `{name}` must be a mapping")
    data: dict[str, Any] = dict(raw)
    unknown = sorted(set(data) - _SOURCE_ALLOWED_KEYS)
    if unknown:
        joined = ", ".join(unknown)
        raise CliConfigError(f"Unknown key(s) in source `{name}`: {joined}")
    dialect = data.get("dialect")
    if not isinstance(dialect, str) or not dialect.strip():
        raise CliConfigError(f"Source `{name}` requires a `dialect` string")
    port = data.get("port")
    if port is not None:
        if isinstance(port, bool) or not isinstance(port, int):
            raise CliConfigError(f"Source `{name}`: `port` must be an integer")
    str_fields = {
        "host",
        "user",
        "password",
        "database",
        "account",
        "schema",
        "warehouse",
        "role",
        "project",
        "dataset",
        "location",
    }
    kwargs: dict[str, Any] = {"dialect": dialect.strip().lower(), "port": port}
    for key in str_fields:
        val = data.get(key)
        if val is None:
            kwargs[key] = None
        elif not isinstance(val, str):
            raise CliConfigError(f"Source `{name}`: `{key}` must be a string")
        else:
            kwargs[key] = val.strip() or None
    return SourceConfig(**kwargs)


def _parse_sources(
    raw: Any,
    secrets_raw: dict[str, Any] | None,
) -> dict[str, SourceConfig]:
    """Parse the ``sources:`` block and merge optional secrets overlay."""
    mapping = _as_mapping(raw, field="sources")
    if not mapping:
        return {}
    sources: dict[str, SourceConfig] = {}
    for name, block in mapping.items():
        if not isinstance(name, str) or not name.strip():
            raise CliConfigError("Source names must be non-empty strings")
        merged = dict(block) if isinstance(block, dict) else block
        if secrets_raw and name in secrets_raw:
            secret_block = secrets_raw[name]
            if not isinstance(secret_block, dict):
                raise CliConfigError(
                    f"Secrets entry for source `{name}` must be a mapping"
                )
            if isinstance(merged, dict):
                merged = {**merged, **secret_block}
        sources[name.strip()] = _source_config_from_mapping(name.strip(), merged)
    return sources


def _load_secrets_file(base_dir: Path) -> dict[str, Any] | None:
    """Load the optional ``sources.secrets.yaml`` sibling to the config file."""
    secrets_path = base_dir / "sources.secrets.yaml"
    if not secrets_path.exists():
        return None
    return _parse_yaml_file(secrets_path)


def _parse_yaml_file(path: Path) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliConfigError(f"Cannot read config file {path}: {exc}") from exc
    except Exception as exc:
        raise CliConfigError(f"Invalid YAML in config file {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise CliConfigError("Config file root must be a mapping")
    return dict(loaded)


def load_cli_config(config_path: Path | None) -> PretensorCliConfig:
    """Load and normalize CLI defaults from YAML."""
    target = (config_path or DEFAULT_CONFIG_PATH).expanduser()
    explicit = config_path is not None
    if not target.exists():
        if explicit:
            raise CliConfigError(f"Config file not found: {target}")
        return PretensorCliConfig()

    source = target.resolve()
    base_dir = source.parent
    raw = _parse_yaml_file(source)

    state_dir = _resolved_optional_path(
        raw.get("state_dir"),
        field="state_dir",
        base_dir=base_dir,
    )
    graph = _graph_config_from_mapping(raw.get("graph"))

    llm_raw = _as_mapping(raw.get("llm"), field="llm")
    llm = LlmDefaults(
        model=_normalized_optional_str(llm_raw.get("model"), field="llm.model"),
        budget_usd=_normalized_optional_float(
            llm_raw.get("budget_usd"),
            field="llm.budget_usd",
        ),
    )

    visibility_raw = _as_mapping(raw.get("visibility"), field="visibility")
    visibility = VisibilityDefaults(
        path=_resolved_optional_path(
            visibility_raw.get("path"),
            field="visibility.path",
            base_dir=base_dir,
        ),
        profile=_normalized_optional_str(
            visibility_raw.get("profile"),
            field="visibility.profile",
        ),
    )

    conn_raw = _as_mapping(raw.get("connection_defaults"), field="connection_defaults")
    connection_defaults = ConnectionDefaults(
        name=_normalized_optional_str(
            conn_raw.get("name"),
            field="connection_defaults.name",
        ),
        dialect=_normalized_optional_str(
            conn_raw.get("dialect"),
            field="connection_defaults.dialect",
        ),
    )

    secrets_raw = _load_secrets_file(base_dir)
    sources = _parse_sources(raw.get("sources"), secrets_raw)

    return PretensorCliConfig(
        source_path=source,
        state_dir=state_dir,
        graph=graph,
        llm=llm,
        visibility=visibility,
        connection_defaults=connection_defaults,
        sources=sources,
    )


def get_cli_config(ctx: typer.Context | None) -> PretensorCliConfig:
    """Return config stored in Typer context, or an empty config."""
    if ctx is None:
        return PretensorCliConfig()
    obj = ctx.obj
    if isinstance(obj, PretensorCliConfig):
        return obj
    if isinstance(obj, dict):
        cfg = obj.get("config")
        if isinstance(cfg, PretensorCliConfig):
            return cfg
    return PretensorCliConfig()


def _is_user_override(ctx: typer.Context | None, param_name: str) -> bool:
    if ctx is None:
        return False
    try:
        source = ctx.get_parameter_source(param_name)
    except Exception:
        return False
    if source is None:
        return False
    return source is not ParameterSource.DEFAULT


def resolve_path_option(
    ctx: typer.Context | None,
    *,
    param_name: str,
    cli_value: Path,
    config_value: Path | None,
) -> Path:
    """Return effective path option with ``CLI > config > default`` precedence."""
    if _is_user_override(ctx, param_name):
        return cli_value
    return config_value or cli_value


def resolve_optional_path_option(
    ctx: typer.Context | None,
    *,
    param_name: str,
    cli_value: Path | None,
    config_value: Path | None,
) -> Path | None:
    """Return effective optional path with ``CLI > config > default`` precedence."""
    if _is_user_override(ctx, param_name):
        return cli_value
    return config_value if config_value is not None else cli_value


def resolve_optional_str_option(
    ctx: typer.Context | None,
    *,
    param_name: str,
    cli_value: str | None,
    config_value: str | None,
) -> str | None:
    """Return effective optional string with ``CLI > config > default`` precedence."""
    if _is_user_override(ctx, param_name):
        return cli_value
    return config_value if config_value is not None else cli_value


def resolve_graph_config(ctx: typer.Context | None) -> GraphConfig:
    """Return graph config from context, falling back to OSS defaults."""
    return get_cli_config(ctx).graph
