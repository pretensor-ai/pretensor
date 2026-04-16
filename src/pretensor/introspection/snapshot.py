"""Snapshot diff — re-exported from ``pretensor.connectors.snapshot``."""

from pathlib import Path

from pretensor.connectors.models import SchemaSnapshot
from pretensor.connectors.snapshot import (
    ChangeTarget,
    ChangeType,
    SchemaChange,
    diff_snapshots,
)


def save_snapshot(snapshot: SchemaSnapshot, base_path: Path) -> Path:
    """Serialize a snapshot to YAML and write it to disk.

    Writes to ``{base_path}/{connection_name}.yaml``.

    Returns:
        The path to the written file.
    """
    base_path.mkdir(parents=True, exist_ok=True)
    file_path = base_path / f"{snapshot.connection_name}.yaml"
    file_path.write_text(snapshot.to_yaml())
    return file_path


def load_snapshot(path: Path) -> SchemaSnapshot:
    """Load a schema snapshot from a YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    return SchemaSnapshot.from_yaml(path.read_text())


__all__ = [
    "ChangeTarget",
    "ChangeType",
    "SchemaChange",
    "diff_snapshots",
    "load_snapshot",
    "save_snapshot",
]
