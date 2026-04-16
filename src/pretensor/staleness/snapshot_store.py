"""Persist schema snapshots next to the graph registry for drift detection."""

from __future__ import annotations

from pathlib import Path

from pretensor.connectors.models import SchemaSnapshot

__all__ = ["SnapshotStore"]


class SnapshotStore:
    """Read/write ``SchemaSnapshot`` YAML under ``.pretensor/snapshots/``."""

    def __init__(self, state_dir: Path) -> None:
        self._dir = state_dir / "snapshots"

    @property
    def snapshots_dir(self) -> Path:
        return self._dir

    def path_for(self, connection_name: str) -> Path:
        safe = connection_name.replace("/", "_")
        return self._dir / f"{safe}.yaml"

    def load(self, connection_name: str) -> SchemaSnapshot | None:
        """Return the last snapshot or None if missing."""
        path = self.path_for(connection_name)
        if not path.exists():
            return None
        return SchemaSnapshot.from_yaml(path.read_text(encoding="utf-8"))

    def save(self, connection_name: str, snapshot: SchemaSnapshot) -> Path:
        """Write snapshot YAML; creates parent directories."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(connection_name)
        path.write_text(snapshot.to_yaml(), encoding="utf-8")
        return path
