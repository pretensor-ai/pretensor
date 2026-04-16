"""Deterministic synthetic dbt manifest fixture for warehouse-scale e2e tests.

See ``generator.py`` and ``README.md``.
"""

from tests.fixtures.dbt.synthetic_warehouse.generator import (
    KnownExposure,
    SyntheticManifestSpec,
    write_synthetic_manifest,
)

__all__ = [
    "KnownExposure",
    "SyntheticManifestSpec",
    "write_synthetic_manifest",
]
