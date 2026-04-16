"""YAML visibility rules: hidden schemas/tables/columns and named profiles."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

__all__ = [
    "VisibilityProfileRule",
    "VisibilityConfig",
    "load_visibility_config",
    "default_visibility_path",
    "merge_profile_into_base",
]


def default_visibility_path(graph_dir: Path) -> Path:
    """Default location for visibility rules under the graph state directory."""
    return graph_dir / "visibility.yml"


class VisibilityProfileRule(BaseModel):
    """Overlay merged onto base visibility when ``--profile`` is set."""

    model_config = ConfigDict(extra="ignore")

    hidden_schemas: list[str] = Field(default_factory=list)
    hidden_tables: list[str] = Field(default_factory=list)
    hidden_columns: list[str] = Field(default_factory=list)
    hidden_table_types: list[str] = Field(default_factory=list)
    allowed_schemas: list[str] = Field(default_factory=list)
    allowed_tables: list[str] = Field(default_factory=list)


class VisibilityConfig(BaseModel):
    """Parsed ``visibility.yml`` (base section)."""

    model_config = ConfigDict(extra="ignore")

    hidden_schemas: list[str] = Field(default_factory=list)
    hidden_tables: list[str] = Field(default_factory=list)
    hidden_columns: list[str] = Field(default_factory=list)
    hidden_table_types: list[str] = Field(default_factory=list)
    allowed_schemas: list[str] = Field(default_factory=list)
    allowed_tables: list[str] = Field(default_factory=list)
    profiles: dict[str, VisibilityProfileRule] = Field(default_factory=dict)


def load_visibility_config(path: Path) -> VisibilityConfig:
    """Load and validate visibility YAML from disk.

    Args:
        path: File to read (typically ``.pretensor/visibility.yml``).

    Returns:
        Parsed config. Missing file yields an empty config.

    Raises:
        ValueError: If YAML is present but invalid.
    """
    if not path.exists():
        return VisibilityConfig()
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return VisibilityConfig()
    yaml = YAML(typ="safe")
    try:
        data = yaml.load(raw_text)
    except Exception as exc:
        raise ValueError(f"Invalid visibility YAML: {path}: {exc}") from exc
    if data is None:
        return VisibilityConfig()
    if not isinstance(data, dict):
        raise ValueError(f"visibility.yml must be a mapping at root: {path}")
    profiles_raw = data.get("profiles")
    base_data = {k: v for k, v in data.items() if k != "profiles"}
    cfg = VisibilityConfig.model_validate(base_data)
    if profiles_raw is None:
        return cfg
    if not isinstance(profiles_raw, dict):
        raise ValueError(f"`profiles` must be a mapping in {path}")
    profiles: dict[str, VisibilityProfileRule] = {}
    for name, body in profiles_raw.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if body is None:
            profiles[name.strip()] = VisibilityProfileRule()
        elif isinstance(body, dict):
            profiles[name.strip()] = VisibilityProfileRule.model_validate(body)
        else:
            raise ValueError(f"Profile {name!r} must be a mapping in {path}")
    return cfg.model_copy(update={"profiles": profiles})


def merge_profile_into_base(
    base: VisibilityConfig,
    profile_name: str | None,
) -> VisibilityConfig:
    """Return a new config: base lists + optional profile overlay."""
    if not profile_name or not profile_name.strip():
        return base
    key = profile_name.strip()
    prof = base.profiles.get(key)
    if prof is None:
        raise ValueError(f"Unknown visibility profile: {key!r}")
    merged_allowed = (
        [*base.allowed_schemas, *prof.allowed_schemas]
        if prof.allowed_schemas
        else list(base.allowed_schemas)
    )
    merged_allowed_tables = (
        [*base.allowed_tables, *prof.allowed_tables]
        if prof.allowed_tables
        else list(base.allowed_tables)
    )
    return VisibilityConfig(
        hidden_schemas=[*base.hidden_schemas, *prof.hidden_schemas],
        hidden_tables=[*base.hidden_tables, *prof.hidden_tables],
        hidden_columns=[*base.hidden_columns, *prof.hidden_columns],
        hidden_table_types=[*base.hidden_table_types, *prof.hidden_table_types],
        allowed_schemas=merged_allowed,
        allowed_tables=merged_allowed_tables,
        profiles=base.profiles,
    )
