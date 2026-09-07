"""CLI config-file loading and option precedence helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from click.core import ParameterSource
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from pretensor.config import GraphConfig
from pretensor.errors import PretensorError

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "CliConfigError",
    "ConnectionDefaults",
    "LlmDefaults",
    "PretensorCliConfig",
    "RepositoryConfig",
    "SourceConfig",
    "VisibilityDefaults",
    "get_cli_config",
    "load_cli_config",
    "record_repository",
    "record_source",
    "resolve_aliased_path_option",
    "resolve_optional_path_option",
    "resolve_optional_str_option",
    "resolve_path_option",
]

DEFAULT_CONFIG_PATH = Path(".pretensor") / "config.yaml"


class CliConfigError(PretensorError, ValueError):
    """Raised when the CLI config file is unreadable or invalid."""


@dataclass(frozen=True, slots=True)
class LlmDefaults:
    """LLM defaults accepted by config for compatibility with external workflows."""

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
        "url",
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
        "private_key_path",
        "private_key_passphrase",
        # BigQuery
        "project",
        "dataset",
        "location",
    }
)

# Field-based keys that are mutually exclusive with `url`. `dialect` is
# deliberately excluded: it remains a valid override alongside `url` (when
# absent, the URL scheme infers the dialect).
_SOURCE_FIELD_KEYS = frozenset(
    {
        "host",
        "port",
        "user",
        "password",
        "database",
        "account",
        "schema",
        "warehouse",
        "role",
        "private_key_path",
        "private_key_passphrase",
        "project",
        "dataset",
        "location",
    }
)


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Declarative database source definition from ``sources:`` config block.

    Either the field-based form (``host``/``user``/``password``/... or the
    dialect-specific equivalents) or the whole-DSN ``url`` form is used, never
    both. ``dialect`` is required for the field-based form; it is optional
    (an override) alongside ``url``, where the URL scheme infers it when
    omitted.
    """

    dialect: str | None = None
    url: str | None = None
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
    private_key_path: str | None = None
    private_key_passphrase: str | None = None
    # BigQuery
    project: str | None = None
    dataset: str | None = None
    location: str | None = None


_REPOSITORY_ALLOWED_KEYS = frozenset({"path", "service", "connection"})


@dataclass(frozen=True, slots=True)
class RepositoryConfig:
    """A code repository linked to one indexed connection."""

    path: Path
    service: str
    connection: str


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
    repositories: tuple[RepositoryConfig, ...] = ()

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
    *,
    overlay_used: bool = False,
) -> SourceConfig:
    """Validate and build a :class:`SourceConfig` from a raw YAML mapping.

    *overlay_used* indicates whether ``raw`` was merged with a
    ``sources.secrets.yaml`` overlay entry for this source, so a
    mutual-exclusion error can point the user at both files rather than just
    ``config.yaml``.
    """
    if not isinstance(raw, dict):
        raise CliConfigError(f"Source `{name}` must be a mapping")
    data: dict[str, Any] = dict(raw)
    unknown = sorted(set(data) - _SOURCE_ALLOWED_KEYS)
    if unknown:
        joined = ", ".join(unknown)
        raise CliConfigError(f"Unknown key(s) in source `{name}`: {joined}")

    url = data.get("url")
    if url is not None:
        if not isinstance(url, str) or not url.strip():
            raise CliConfigError(f"Source `{name}`: `url` must be a non-empty string")
        url = url.strip()
        conflicting = sorted(
            key for key in _SOURCE_FIELD_KEYS if data.get(key) is not None
        )
        if conflicting:
            joined = ", ".join(conflicting)
            hint = (
                " The conflicting key(s) may come from `sources.secrets.yaml`; "
                "check both `config.yaml` and `sources.secrets.yaml` for this "
                "source."
                if overlay_used
                else ""
            )
            raise CliConfigError(
                f"Source `{name}`: `url` cannot be combined with {joined}.{hint}"
            )

    dialect = data.get("dialect")
    if dialect is None:
        if url is None:
            raise CliConfigError(f"Source `{name}` requires a `dialect` string")
        dialect_value: str | None = None
    else:
        if not isinstance(dialect, str) or not dialect.strip():
            raise CliConfigError(f"Source `{name}`: `dialect` must be a string")
        dialect_value = dialect.strip().lower()

    if url is not None:
        return SourceConfig(dialect=dialect_value, url=url)

    port = data.get("port")
    if port is not None:
        if isinstance(port, bool) or not isinstance(port, int):
            raise CliConfigError(f"Source `{name}`: `port` must be an integer")
    kwargs: dict[str, Any] = {"dialect": dialect_value, "port": port}
    for key in _SOURCE_FIELD_KEYS - {"port"}:
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
        overlay_used = False
        if secrets_raw and name in secrets_raw:
            secret_block = secrets_raw[name]
            if not isinstance(secret_block, dict):
                raise CliConfigError(
                    f"Secrets entry for source `{name}` must be a mapping"
                )
            if isinstance(merged, dict):
                merged = {**merged, **secret_block}
                overlay_used = True
        sources[name.strip()] = _source_config_from_mapping(
            name.strip(), merged, overlay_used=overlay_used
        )
    return sources


def _parse_repositories(raw: Any, base_dir: Path) -> tuple[RepositoryConfig, ...]:
    """Parse the ``repositories:`` block used by ``analyze --all``."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise CliConfigError("`repositories` must be a list in config YAML")
    parsed: list[RepositoryConfig] = []
    for index, entry in enumerate(raw):
        label = f"repositories[{index}]"
        if not isinstance(entry, dict):
            raise CliConfigError(f"{label} must be a mapping")
        unknown = sorted(set(entry) - _REPOSITORY_ALLOWED_KEYS)
        if unknown:
            joined = ", ".join(unknown)
            raise CliConfigError(f"Unknown key(s) in {label}: {joined}")
        path = _resolved_path(
            entry.get("path"), field=f"{label}.path", base_dir=base_dir
        )
        connection = _normalized_optional_str(
            entry.get("connection"), field=f"{label}.connection"
        )
        if not connection:
            raise CliConfigError(f"{label} requires a `connection` string")
        service = (
            _normalized_optional_str(entry.get("service"), field=f"{label}.service")
            or path.name
        )
        parsed.append(
            RepositoryConfig(path=path, service=service, connection=connection)
        )
    return tuple(parsed)


def _round_trip_yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # avoid rewrapping long lines (e.g. paths)
    return yaml


def _backup_config_file(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    shutil.copy2(path, backup)
    return backup


def _atomic_write_yaml(path: Path, yaml: YAML, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            yaml.dump(data, stream)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def record_repository(
    config_path: Path,
    *,
    path: Path,
    service: str,
    connection: str,
) -> None:
    """Append or update one entry under ``repositories:`` in *config_path*.

    Loads (and re-writes) the YAML with the round-trip loader so existing
    comments and key order survive. Entries are deduped by resolved path: a
    call for a path already present updates its ``service``/``connection``
    instead of appending a duplicate. The file is parsed *before* it is
    backed up (mirroring ``mcp_clients.register_client``): a malformed
    pre-existing file must raise without ever being touched, so the backup
    is only taken once loading has proven the file readable.
    """
    config_path = Path(config_path)
    resolved_repo_path = str(Path(path).resolve())

    yaml = _round_trip_yaml()

    file_existed = config_path.exists()
    if file_existed:
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.load(stream)
        if data is None:
            data = CommentedMap()
    else:
        data = CommentedMap()

    repos = data.get("repositories")
    if not isinstance(repos, list):
        repos = CommentedSeq()
        data["repositories"] = repos

    entry: Any = None
    for item in repos:
        if not isinstance(item, dict):
            continue
        existing_path = item.get("path")
        if not isinstance(existing_path, str):
            continue
        try:
            if str(Path(existing_path).resolve()) == resolved_repo_path:
                entry = item
                break
        except OSError:
            continue

    if entry is not None:
        entry["service"] = service
        entry["connection"] = connection
    else:
        new_entry = CommentedMap()
        new_entry["path"] = resolved_repo_path
        new_entry["service"] = service
        new_entry["connection"] = connection
        repos.append(new_entry)

    # Only back up once the file has been proven readable and valid: a
    # failed record must not strand a `.bak` copy behind.
    if file_existed:
        _backup_config_file(config_path)
    _atomic_write_yaml(config_path, yaml, data)


def record_source(
    config_path: Path,
    *,
    name: str,
    url_reference: str,
    dialect: str | None = None,
) -> None:
    """Write or update one entry under ``sources:`` in *config_path*.

    Mirrors ``record_repository``'s safety pattern: parse first with the
    round-trip loader (so existing comments and key order survive), back up
    only after a successful parse, atomic write. ``sources:`` is a mapping
    keyed by name, so this call is naturally idempotent: a second call with
    the same *name* updates that entry's ``url``/``dialect`` in place instead
    of appending a duplicate.

    An existing entry for *name* may be field-based (``host``/``user``/
    ``password``/...); rewriting it as ``url``-based clears every
    ``_SOURCE_FIELD_KEYS`` member from that entry first, since ``url`` and
    the field-based form are mutually exclusive and leaving both would make
    the entry unloadable. ``dialect`` is the one exception: when *dialect*
    is given it is set, otherwise any existing ``dialect`` on the entry is
    left untouched (it is a valid override alongside ``url``).

    *url_reference* must be the literal ``${VAR}`` reference (e.g.
    ``"${DATABASE_URL}"``), never a resolved DSN. Callers are responsible for
    only calling this when the DSN came from a single environment variable.
    """
    config_path = Path(config_path)
    yaml = _round_trip_yaml()

    file_existed = config_path.exists()
    if file_existed:
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.load(stream)
        if data is None:
            data = CommentedMap()
    else:
        data = CommentedMap()

    sources = data.get("sources")
    if not isinstance(sources, dict):
        sources = CommentedMap()
        data["sources"] = sources

    entry = sources.get(name)
    if not isinstance(entry, dict):
        entry = CommentedMap()
        sources[name] = entry

    for key in _SOURCE_FIELD_KEYS:
        if key in entry:
            del entry[key]
    entry["url"] = url_reference
    if dialect is not None:
        entry["dialect"] = dialect

    # Only back up once the file has been proven readable and valid: a
    # failed record must not strand a `.bak` copy behind.
    if file_existed:
        _backup_config_file(config_path)
    _atomic_write_yaml(config_path, yaml, data)


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
    repositories = _parse_repositories(raw.get("repositories"), base_dir)

    return PretensorCliConfig(
        source_path=source,
        state_dir=state_dir,
        graph=graph,
        llm=llm,
        visibility=visibility,
        connection_defaults=connection_defaults,
        sources=sources,
        repositories=repositories,
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


def resolve_aliased_path_option(
    ctx: typer.Context | None,
    *,
    primary_param: str,
    primary_value: Path,
    alias_param: str,
    alias_value: Path | None,
    config_value: Path | None,
) -> Path:
    """Resolve a path option that has a hidden, deprecated alias flag.

    Precedence: explicit ``primary_param`` > explicit ``alias_param`` > config
    > ``primary_value`` default. If both flags are passed explicitly, the
    primary (canonical) one wins.
    """
    if _is_user_override(ctx, primary_param):
        return primary_value
    if _is_user_override(ctx, alias_param) and alias_value is not None:
        return alias_value
    return config_value if config_value is not None else primary_value


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
