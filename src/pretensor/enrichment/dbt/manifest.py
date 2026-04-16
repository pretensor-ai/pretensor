"""Load and parse dbt ``manifest.json`` artifacts for graph enrichment."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_SUPPORTED_SCHEMA_MARKERS = ("manifest/v9", "manifest/v10", "/v9.json", "/v10.json")


class DbtManifestError(Exception):
    """Raised when a dbt manifest file is missing or cannot be parsed."""


@dataclass(frozen=True, slots=True)
class DbtModel:
    """A dbt model node from ``manifest.json`` ``nodes``."""

    unique_id: str
    name: str
    description: str | None = None
    tags: tuple[str, ...] = ()
    package_name: str | None = None
    database: str | None = None
    schema_name: str | None = None
    alias: str | None = None
    column_descriptions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DbtSource:
    """A dbt source from ``manifest.json`` ``sources``."""

    unique_id: str
    name: str
    source_name: str | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()
    package_name: str | None = None
    database: str | None = None
    schema_name: str | None = None
    identifier: str | None = None
    column_descriptions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DbtExposure:
    """A dbt exposure from ``manifest.json`` ``exposures``."""

    unique_id: str
    name: str
    description: str | None = None
    tags: tuple[str, ...] = ()
    package_name: str | None = None
    depends_on_nodes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DbtTest:
    """A dbt test node (typically under ``nodes`` with ``resource_type`` test)."""

    unique_id: str
    name: str | None = None
    package_name: str | None = None
    attached_node: str | None = None


def _is_supported_schema_version(version: str | None) -> bool:
    if version is None:
        return True
    lowered = version.lower()
    return any(marker in lowered for marker in _SUPPORTED_SCHEMA_MARKERS)


def _warn_if_unsupported_schema(version: str | None) -> None:
    if version is not None and not _is_supported_schema_version(version):
        logger.warning(
            "dbt manifest schema version may be unsupported: %s "
            "(expected v9 or v10 URL in metadata.dbt_schema_version)",
            version,
        )


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            s = _as_str(item)
            if s is not None:
                out.append(s)
        return tuple(out)
    return ()


def _depends_on_nodes(raw: Mapping[str, Any]) -> tuple[str, ...]:
    depends_on = raw.get("depends_on")
    if not isinstance(depends_on, dict):
        return ()
    nodes = depends_on.get("nodes")
    if not isinstance(nodes, list):
        return ()
    return tuple(n for n in (_as_str(x) for x in nodes) if n is not None)


def _column_descriptions_from_raw(raw: Mapping[str, Any]) -> dict[str, str]:
    """Map column name -> non-empty inline ``description`` from manifest ``columns``."""
    cols = raw.get("columns")
    if not isinstance(cols, dict):
        return {}
    out: dict[str, str] = {}
    for key, col_raw in cols.items():
        col_name = _as_str(key)
        if col_name is None or not isinstance(col_raw, dict):
            continue
        desc = _as_str(col_raw.get("description"))
        if desc is None:
            continue
        stripped = desc.strip()
        if stripped:
            out[col_name] = stripped
    return out


def _parse_model(unique_id: str, raw: Mapping[str, Any]) -> DbtModel:
    return DbtModel(
        unique_id=unique_id,
        name=_as_str(raw.get("name")) or unique_id,
        description=_as_str(raw.get("description")),
        tags=_as_str_tuple(raw.get("tags")),
        package_name=_as_str(raw.get("package_name")),
        database=_as_str(raw.get("database")),
        schema_name=_as_str(raw.get("schema")),
        alias=_as_str(raw.get("alias")),
        column_descriptions=_column_descriptions_from_raw(raw),
    )


def _parse_source(unique_id: str, raw: Mapping[str, Any]) -> DbtSource:
    return DbtSource(
        unique_id=unique_id,
        name=_as_str(raw.get("name")) or unique_id,
        source_name=_as_str(raw.get("source_name")),
        description=_as_str(raw.get("description")),
        tags=_as_str_tuple(raw.get("tags")),
        package_name=_as_str(raw.get("package_name")),
        database=_as_str(raw.get("database")),
        schema_name=_as_str(raw.get("schema")),
        identifier=_as_str(raw.get("identifier")),
        column_descriptions=_column_descriptions_from_raw(raw),
    )


def _parse_exposure(unique_id: str, raw: Mapping[str, Any]) -> DbtExposure:
    return DbtExposure(
        unique_id=unique_id,
        name=_as_str(raw.get("name")) or unique_id,
        description=_as_str(raw.get("description")),
        tags=_as_str_tuple(raw.get("tags")),
        package_name=_as_str(raw.get("package_name")),
        depends_on_nodes=_depends_on_nodes(raw),
    )


def _parse_test(unique_id: str, raw: Mapping[str, Any]) -> DbtTest:
    return DbtTest(
        unique_id=unique_id,
        name=_as_str(raw.get("name")),
        package_name=_as_str(raw.get("package_name")),
        attached_node=_as_str(raw.get("attached_node")),
    )


def _parse_parent_map(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[str]] = {}
    for key, value in raw.items():
        kid = _as_str(key)
        if kid is None:
            continue
        if not isinstance(value, list):
            result[kid] = []
            continue
        parents: list[str] = []
        for item in value:
            sid = _as_str(item)
            if sid is not None:
                parents.append(sid)
        result[kid] = parents
    return result


@dataclass(frozen=True, slots=True)
class DbtManifest:
    """Parsed dbt ``manifest.json`` (v9/v10-oriented; unknown fields ignored)."""

    schema_version: str | None
    nodes: dict[str, DbtModel]
    sources: dict[str, DbtSource]
    exposures: dict[str, DbtExposure]
    tests: dict[str, DbtTest]
    parent_map: dict[str, list[str]]

    @classmethod
    def load(cls, path: Path) -> DbtManifest:
        """Load a manifest from ``path``.

        Args:
            path: Filesystem path to ``manifest.json``.

        Returns:
            Parsed manifest.

        Raises:
            DbtManifestError: If the file is missing or JSON is invalid.
        """
        if not path.is_file():
            msg = f"dbt manifest not found or not a file: {path}"
            raise DbtManifestError(msg)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"cannot read dbt manifest: {path}"
            raise DbtManifestError(msg) from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"dbt manifest is not valid JSON: {path}"
            raise DbtManifestError(msg) from exc
        if not isinstance(payload, dict):
            msg = f"dbt manifest root must be a JSON object: {path}"
            raise DbtManifestError(msg)

        metadata = payload.get("metadata")
        schema_version: str | None = None
        if isinstance(metadata, dict):
            schema_version = _as_str(metadata.get("dbt_schema_version"))
        _warn_if_unsupported_schema(schema_version)

        nodes_out: dict[str, DbtModel] = {}
        tests_out: dict[str, DbtTest] = {}
        raw_nodes = payload.get("nodes")
        if isinstance(raw_nodes, dict):
            for node_id, node_raw in raw_nodes.items():
                nid = _as_str(node_id)
                if nid is None or not isinstance(node_raw, dict):
                    continue
                rtype = _as_str(node_raw.get("resource_type"))
                if rtype == "model":
                    nodes_out[nid] = _parse_model(nid, node_raw)
                elif rtype == "test":
                    tests_out[nid] = _parse_test(nid, node_raw)

        sources_out: dict[str, DbtSource] = {}
        raw_sources = payload.get("sources")
        if isinstance(raw_sources, dict):
            for source_id, source_raw in raw_sources.items():
                sid = _as_str(source_id)
                if sid is None or not isinstance(source_raw, dict):
                    continue
                sources_out[sid] = _parse_source(sid, source_raw)

        exposures_out: dict[str, DbtExposure] = {}
        raw_exposures = payload.get("exposures")
        if isinstance(raw_exposures, dict):
            for exposure_id, exposure_raw in raw_exposures.items():
                eid = _as_str(exposure_id)
                if eid is None or not isinstance(exposure_raw, dict):
                    continue
                exposures_out[eid] = _parse_exposure(eid, exposure_raw)

        parent_map = _parse_parent_map(payload.get("parent_map"))

        return cls(
            schema_version=schema_version,
            nodes=nodes_out,
            sources=sources_out,
            exposures=exposures_out,
            tests=tests_out,
            parent_map=parent_map,
        )
