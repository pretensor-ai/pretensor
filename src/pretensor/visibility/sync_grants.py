"""Merge DB SELECT grants into ``visibility.yml`` profile ``allowed_tables``."""

from __future__ import annotations

import copy
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from ruamel.yaml import YAML

from pretensor.connectors.base import BaseConnector, TableGrant

__all__ = [
    "allowed_table_patterns_for_grants",
    "merge_grant_profiles_into_visibility_doc",
    "write_visibility_yaml_document",
]


def allowed_table_patterns_for_grants(
    grants: list[TableGrant],
    connection_name: str,
) -> dict[str, list[str]]:
    """Group grants by grantee; each value is sorted ``connection::schema.table`` patterns."""
    by_grantee: dict[str, set[str]] = defaultdict(set)
    for row in grants:
        g = row.grantee.strip()
        if not g:
            continue
        schema = row.schema_name.strip()
        table = row.table_name.strip()
        if not schema or not table:
            continue
        by_grantee[g].add(f"{connection_name}::{schema}.{table}")
    return {k: sorted(v) for k, v in sorted(by_grantee.items(), key=lambda kv: kv[0])}


def _is_grant_only_profile(body: Mapping[str, Any]) -> bool:
    """True if the profile body only carries ``allowed_tables`` (grant-sync shape)."""
    keys = frozenset(body.keys())
    return keys == frozenset({"allowed_tables"})


def merge_grant_profiles_into_visibility_doc(
    *,
    existing_root: Mapping[str, Any] | None,
    grantee_to_tables: Mapping[str, list[str]],
    roles_filter: frozenset[str] | None,
) -> dict[str, Any]:
    """Return a YAML-root mapping: base keys preserved, ``profiles`` merged for grant sync.

    Profiles not targeted by this sync pass are copied verbatim. For each targeted
    grantee, ``allowed_tables`` is set to the computed list; other keys on that profile
    are preserved. Grantees with no tables omit ``allowed_tables`` unless the profile
    has other keys (then ``allowed_tables`` is removed to avoid an empty allowlist).

    Args:
        existing_root: Parsed root mapping from the current file, or ``None``.
        grantee_to_tables: Grantee name -> ``allowed_tables`` patterns (sorted).
        roles_filter: When set, only these grantees receive grant-driven updates.

    Returns:
        New root dict suitable for YAML serialization.
    """
    root: dict[str, Any] = (
        copy.deepcopy(dict(existing_root)) if existing_root else {}
    )
    profiles_any = root.get("profiles")
    profiles: dict[str, Any] = (
        copy.deepcopy(dict(profiles_any)) if isinstance(profiles_any, dict) else {}
    )

    grant_keys = frozenset(grantee_to_tables.keys())
    if roles_filter is None:
        stale = [
            name
            for name, body in list(profiles.items())
            if name not in grant_keys
            and isinstance(body, dict)
            and _is_grant_only_profile(body)
        ]
        for name in stale:
            profiles.pop(name, None)

    if roles_filter is not None:
        targets = sorted(roles_filter)
    else:
        targets = sorted(grant_keys)

    for grantee in targets:
        tables = list(grantee_to_tables.get(grantee, []))
        existing_body = profiles.get(grantee)
        body: dict[str, Any] = (
            copy.deepcopy(dict(existing_body))
            if isinstance(existing_body, dict)
            else {}
        )
        other_keys = {k for k in body if k != "allowed_tables"}
        if tables:
            body["allowed_tables"] = tables
        else:
            body.pop("allowed_tables", None)
            if not other_keys:
                profiles.pop(grantee, None)
                continue
        profiles[grantee] = body

    if profiles:
        root["profiles"] = profiles
    else:
        root.pop("profiles", None)

    return root


def write_visibility_yaml_document(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically write a mapping to ``path`` using block-style YAML.

    Writes to a sibling ``.tmp`` file first and renames on success, so an
    interrupted run (Ctrl-C, crash, disk full) never leaves ``visibility.yml``
    truncated — the file is either the previous content or the new content,
    never a partial write. ``os.replace`` is atomic on POSIX and Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            yaml.dump(dict(document), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up partial temp file so a retry starts from a clean slate.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def load_visibility_yaml_root(path: Path) -> dict[str, Any] | None:
    """Load the raw YAML root mapping, or ``None`` if missing/empty."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    yaml = YAML(typ="safe")
    data = yaml.load(text)
    if data is None:
        return None
    if not isinstance(data, dict):
        msg = f"visibility.yml must be a mapping at root: {path}"
        raise ValueError(msg)
    return data


def run_sync_grants(
    connector: BaseConnector,
    *,
    output_path: Path,
    connection_name: str,
    roles_filter: frozenset[str] | None,
) -> int:
    """Read grants from ``connector``, merge into ``output_path``, return grantee count.

    The connector must already be connected. The file on disk is replaced atomically
    with merged content; base visibility keys and non-target profiles are preserved.

    Returns the number of distinct grantees for which ``allowed_tables`` was written
    (0 if the connector returned no grants). Callers that need the parsed
    :class:`VisibilityConfig` should call :func:`load_visibility_config` afterward.
    """
    grants = connector.get_table_grants()
    grantee_to_tables = allowed_table_patterns_for_grants(grants, connection_name)
    existing_root = load_visibility_yaml_root(output_path)
    merged = merge_grant_profiles_into_visibility_doc(
        existing_root=existing_root,
        grantee_to_tables=grantee_to_tables,
        roles_filter=roles_filter,
    )
    write_visibility_yaml_document(output_path, merged)
    return len(grantee_to_tables)
