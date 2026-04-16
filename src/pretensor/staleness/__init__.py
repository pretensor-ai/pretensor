"""Schema drift: snapshots, graph patching, and impact analysis."""

from __future__ import annotations

from pretensor.staleness.graph_patcher import GraphPatcher, PatchResult
from pretensor.staleness.impact_analyzer import ImpactAnalyzer, ImpactReport
from pretensor.staleness.snapshot_store import SnapshotStore

__all__ = [
    "GraphPatcher",
    "ImpactAnalyzer",
    "ImpactReport",
    "PatchResult",
    "SnapshotStore",
]
