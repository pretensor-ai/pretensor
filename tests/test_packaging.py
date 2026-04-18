"""Packaging smoke tests — verify PEP 561 typing marker is distributed."""

from __future__ import annotations

import importlib.resources


def test_py_typed_marker_is_shipped() -> None:
    marker = importlib.resources.files("pretensor").joinpath("py.typed")
    assert marker.is_file()
