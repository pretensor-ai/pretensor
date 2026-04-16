"""Deterministic synthetic dbt ``manifest.json`` generator.

Produces a ~1,500-model manifest that mimics warehouse-scale shape — varied
fan-out, ``alias``/``identifier`` overrides, partial description coverage,
exposures spread across the DAG, dangling refs — without depending on any
external dataset. Targets ``manifest/v10`` (the newest version supported by
``src/pretensor/enrichment/dbt/manifest.py``).

Used by ``tests/e2e/test_dbt_warehouse_scale_enrichment.py`` to regression-guard
the dbt enrichment pipeline at warehouse scale. See ``README.md`` for refresh
policy and the rationale for shipping a synthetic stand-in instead of a
vendored ``gitlab-data/analytics`` snapshot.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CONNECTION_NAME",
    "DATABASE_NAME",
    "KnownExposure",
    "SyntheticManifestSpec",
    "write_synthetic_manifest",
]

CONNECTION_NAME = "synthetic_warehouse"
DATABASE_NAME = "warehouse"
SCHEMA_VERSION_URL = "https://schemas.getdbt.com/dbt/manifest/v10.json"
PACKAGE_NAME = "synthetic"

# Generation knobs — stable; bump only when refreshing the fixture.
TARGET_MODEL_COUNT = 1500
TARGET_SOURCE_COUNT = 150
EXPOSURE_COUNT = 12
KNOWN_EXPOSURE_COUNT = 6  # subset asserted on by the test
TEST_COUNT = 50
DESCRIPTION_COVERAGE = 0.85
ALIAS_OVERRIDE_RATIO = 0.10
SOURCE_IDENTIFIER_OVERRIDE_RATIO = 0.25
DANGLING_PARENT_RATIO = 0.05
MAX_PARENTS_PER_MODEL = 6


@dataclass(frozen=True, slots=True)
class KnownExposure:
    """A hand-picked exposure the test asserts ``has_external_consumers`` on.

    ``expected_tables`` is the set of ``(schema, physical_name)`` pairs the
    enrichment writer should mark, computed at generation time so the test
    doesn't have to re-derive them.
    """

    unique_id: str
    expected_tables: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SyntheticManifestSpec:
    """Reference metadata returned to the test fixture.

    Decouples the generator from the assertions by recording, at generation
    time, every value the assertions need to verify against. Refreshing the
    fixture (changing the seed or knobs) automatically updates these counts.
    """

    connection_name: str
    database: str
    physical_tables: tuple[tuple[str, str], ...]
    """Unique ``(schema, physical_name)`` pairs to materialize as DDL."""
    expected_lineage_edges: int
    """Upper bound on lineage edges the writer should emit (resolvable parents)."""
    documented_resolvable_models: tuple[tuple[str, str], ...]
    """Models that have a description AND resolve to a synthetic warehouse table."""
    known_exposures: tuple[KnownExposure, ...]


_SCHEMA_PREFIXES = (
    "analytics",
    "marts_finance",
    "marts_product",
    "marts_growth",
    "marts_marketing",
    "intermediate",
    "staging",
    "reporting",
    "core",
    "metrics",
)
_MODEL_PREFIXES = ("dim", "fct", "stg", "int", "rpt", "mart")
_NOUNS = (
    "account",
    "user",
    "session",
    "order",
    "invoice",
    "payment",
    "subscription",
    "product",
    "campaign",
    "ticket",
    "lead",
    "opportunity",
    "shipment",
    "refund",
    "device",
    "event",
    "page_view",
    "click",
    "signup",
    "trial",
    "renewal",
    "feature_use",
    "cohort",
    "channel",
    "experiment",
    "variant",
    "region",
    "country",
    "currency",
    "category",
    "vendor",
    "warehouse_loc",
    "shipment_leg",
    "fulfillment",
    "address",
    "contact",
)


def _gen_schemas(rnd: random.Random) -> list[str]:
    """Build ~30 model schemas. Names are stable across runs for the same seed."""
    schemas: list[str] = []
    for prefix in _SCHEMA_PREFIXES:
        # Each prefix expands into 2-4 sub-schemas (e.g. analytics, analytics_v2).
        for i in range(rnd.randint(2, 4)):
            schemas.append(prefix if i == 0 else f"{prefix}_v{i + 1}")
    return schemas


def _gen_source_schemas(rnd: random.Random) -> list[str]:
    raw_schemas = [
        "raw_app",
        "raw_billing",
        "raw_crm",
        "raw_events",
        "raw_marketing",
        "raw_support",
        "raw_finance",
        "raw_product",
        "raw_logistics",
        "raw_legacy",
    ]
    rnd.shuffle(raw_schemas)
    return raw_schemas


def _gen_model_name(rnd: random.Random, used: set[str]) -> str:
    """Build a unique model name like ``fct_orders_v3``."""
    while True:
        prefix = rnd.choice(_MODEL_PREFIXES)
        noun = rnd.choice(_NOUNS)
        suffix = rnd.choice(("", "", "", "_daily", "_v2", "_history", "_summary"))
        name = f"{prefix}_{noun}{suffix}"
        if name not in used:
            used.add(name)
            return name
        # Collision: append a counter.
        for n in range(2, 100):
            cand = f"{name}_{n}"
            if cand not in used:
                used.add(cand)
                return cand


def _gen_source_name(rnd: random.Random, used: set[str]) -> str:
    while True:
        noun = rnd.choice(_NOUNS)
        suffix = rnd.choice(("", "", "_raw", "_log", "_stream"))
        name = f"{noun}{suffix}"
        if name not in used:
            used.add(name)
            return name
        for n in range(2, 100):
            cand = f"{name}_{n}"
            if cand not in used:
                used.add(cand)
                return cand


def _build_models(
    rnd: random.Random, schemas: list[str]
) -> tuple[dict[str, dict[str, object]], dict[str, tuple[str, str]]]:
    """Return (raw nodes dict for manifest JSON, mapping unique_id -> (schema, physical_name))."""
    raw_nodes: dict[str, dict[str, object]] = {}
    physical_by_id: dict[str, tuple[str, str]] = {}
    used_per_schema: dict[str, set[str]] = {s: set() for s in schemas}
    for i in range(TARGET_MODEL_COUNT):
        schema = schemas[i % len(schemas)]
        name = _gen_model_name(rnd, used_per_schema[schema])
        unique_id = f"model.{PACKAGE_NAME}.{name}"
        # Some models alias to a different physical name.
        if rnd.random() < ALIAS_OVERRIDE_RATIO:
            alias_used = used_per_schema[schema]
            alias = _gen_model_name(rnd, alias_used)
        else:
            alias = name
        physical_by_id[unique_id] = (schema, alias)
        node: dict[str, object] = {
            "resource_type": "model",
            "name": name,
            "package_name": PACKAGE_NAME,
            "database": DATABASE_NAME,
            "schema": schema,
            "alias": alias,
            "tags": [rnd.choice(("daily", "hourly", "core", "exp"))],
        }
        if rnd.random() < DESCRIPTION_COVERAGE:
            node["description"] = (
                f"Synthetic model {name} for warehouse-scale enrichment regression test."
            )
        raw_nodes[unique_id] = node
    return raw_nodes, physical_by_id


def _build_sources(
    rnd: random.Random, source_schemas: list[str]
) -> tuple[dict[str, dict[str, object]], dict[str, tuple[str, str]]]:
    raw_sources: dict[str, dict[str, object]] = {}
    physical_by_id: dict[str, tuple[str, str]] = {}
    used_per_schema: dict[str, set[str]] = {s: set() for s in source_schemas}
    for i in range(TARGET_SOURCE_COUNT):
        schema = source_schemas[i % len(source_schemas)]
        name = _gen_source_name(rnd, used_per_schema[schema])
        # Source names are scoped per source group; we use schema as the group.
        unique_id = f"source.{PACKAGE_NAME}.{schema}.{name}"
        if rnd.random() < SOURCE_IDENTIFIER_OVERRIDE_RATIO:
            identifier_used = used_per_schema[schema]
            identifier = _gen_source_name(rnd, identifier_used)
        else:
            identifier = name
        physical_by_id[unique_id] = (schema, identifier)
        raw_sources[unique_id] = {
            "resource_type": "source",
            "name": name,
            "source_name": schema,
            "package_name": PACKAGE_NAME,
            "database": DATABASE_NAME,
            "schema": schema,
            "identifier": identifier,
            "tags": [],
        }
    return raw_sources, physical_by_id


def _build_parent_map(
    rnd: random.Random,
    model_ids: list[str],
    source_ids: list[str],
) -> tuple[dict[str, list[str]], int]:
    """Wire each model to 0-6 parents from earlier models + sources.

    Returns ``(parent_map, resolvable_edge_count)``. The resolvable count
    excludes dangling parents (intentionally injected to exercise the
    resolver's silent-skip path).
    """
    parent_map: dict[str, list[str]] = {sid: [] for sid in source_ids}
    resolvable = 0
    for idx, mid in enumerate(model_ids):
        # Earlier models in the list are eligible parents for later ones.
        # Power-law weighting: occasional models become hubs.
        n_parents = min(rnd.randint(0, MAX_PARENTS_PER_MODEL), MAX_PARENTS_PER_MODEL)
        candidate_pool: list[str] = []
        candidate_pool.extend(source_ids)
        candidate_pool.extend(model_ids[: max(0, idx)])
        if not candidate_pool:
            parent_map[mid] = []
            continue
        # Bias toward sources to create source hubs with many children.
        weights = [3.0 if pid.startswith("source.") else 1.0 for pid in candidate_pool]
        # rnd.choices does not deduplicate; we trim duplicates afterward.
        picks = rnd.choices(candidate_pool, weights=weights, k=n_parents)
        unique_picks: list[str] = []
        seen: set[str] = set()
        for p in picks:
            if p in seen:
                continue
            seen.add(p)
            unique_picks.append(p)
        # Inject a dangling parent ~5% of the time so the silent-skip path runs.
        if rnd.random() < DANGLING_PARENT_RATIO:
            unique_picks.append(f"model.{PACKAGE_NAME}.does_not_exist_{idx}")
        parent_map[mid] = unique_picks
        resolvable += sum(
            1
            for p in unique_picks
            if p.startswith("source.") or p.startswith("model.")
            if not p.endswith(f"does_not_exist_{idx}")
        )
    return parent_map, resolvable


def _build_exposures(
    rnd: random.Random,
    model_ids: list[str],
    physical_by_id: dict[str, tuple[str, str]],
) -> tuple[dict[str, dict[str, object]], tuple[KnownExposure, ...]]:
    raw: dict[str, dict[str, object]] = {}
    known: list[KnownExposure] = []
    # Bias toward late-DAG models so the BFS upstream walks several levels.
    eligible = model_ids[len(model_ids) // 2 :]
    for i in range(EXPOSURE_COUNT):
        deps_count = rnd.randint(2, 4)
        deps = rnd.sample(eligible, deps_count)
        unique_id = f"exposure.{PACKAGE_NAME}.dashboard_{i}"
        raw[unique_id] = {
            "resource_type": "exposure",
            "name": f"dashboard_{i}",
            "package_name": PACKAGE_NAME,
            "depends_on": {"nodes": deps},
            "tags": [],
        }
        if i < KNOWN_EXPOSURE_COUNT:
            expected = tuple(physical_by_id[d] for d in deps if d in physical_by_id)
            known.append(KnownExposure(unique_id=unique_id, expected_tables=expected))
    return raw, tuple(known)


def _build_tests(
    rnd: random.Random, model_ids: list[str]
) -> dict[str, dict[str, object]]:
    raw: dict[str, dict[str, object]] = {}
    for i in range(TEST_COUNT):
        attached = rnd.choice(model_ids)
        unique_id = f"test.{PACKAGE_NAME}.not_null_{i}"
        raw[unique_id] = {
            "resource_type": "test",
            "name": f"not_null_{i}",
            "package_name": PACKAGE_NAME,
            "attached_node": attached,
        }
    return raw


def write_synthetic_manifest(
    out_path: Path, *, seed: int = 0
) -> SyntheticManifestSpec:
    """Generate a synthetic ``manifest.json`` at ``out_path``.

    Args:
        out_path: Where to write the manifest. Parent directories must exist.
        seed: PRNG seed. Use a fixed value (default ``0``) for deterministic
            test runs; bump only when refreshing the fixture intentionally.

    Returns:
        Reference metadata used by the test assertions.
    """
    rnd = random.Random(seed)

    schemas = _gen_schemas(rnd)
    source_schemas = _gen_source_schemas(rnd)

    raw_models, model_phys = _build_models(rnd, schemas)
    raw_sources, source_phys = _build_sources(rnd, source_schemas)

    model_ids = list(raw_models.keys())
    source_ids = list(raw_sources.keys())
    physical_by_id: dict[str, tuple[str, str]] = {**model_phys, **source_phys}

    parent_map, resolvable_edges = _build_parent_map(rnd, model_ids, source_ids)
    raw_exposures, known_exposures = _build_exposures(rnd, model_ids, physical_by_id)
    raw_tests = _build_tests(rnd, model_ids)

    # Combine model + test nodes under "nodes" (manifest convention).
    nodes_payload: dict[str, dict[str, object]] = {**raw_models, **raw_tests}

    payload: dict[str, object] = {
        "metadata": {"dbt_schema_version": SCHEMA_VERSION_URL},
        "nodes": nodes_payload,
        "sources": raw_sources,
        "exposures": raw_exposures,
        "parent_map": parent_map,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    # Reference metadata for assertions.
    physical_tables: set[tuple[str, str]] = set(physical_by_id.values())
    documented_resolvable: list[tuple[str, str]] = []
    for mid, raw in raw_models.items():
        desc = raw.get("description")
        if not isinstance(desc, str) or not desc:
            continue
        documented_resolvable.append(physical_by_id[mid])

    return SyntheticManifestSpec(
        connection_name=CONNECTION_NAME,
        database=DATABASE_NAME,
        physical_tables=tuple(sorted(physical_tables)),
        expected_lineage_edges=resolvable_edges,
        documented_resolvable_models=tuple(documented_resolvable),
        known_exposures=known_exposures,
    )
