"""Packaging smoke tests — verify package data files are distributed."""

from __future__ import annotations

import importlib.resources


def test_py_typed_marker_is_shipped() -> None:
    marker = importlib.resources.files("pretensor").joinpath("py.typed")
    assert marker.is_file()


def test_quickstart_assets_are_shipped() -> None:
    """`pretensor quickstart` needs the compose file and pagila SQL fixtures
    to land in the wheel; otherwise a `pip install pretensor` cannot run the
    one-command quickstart."""
    pkg = importlib.resources.files("pretensor.quickstart")
    assert pkg.joinpath("docker-compose.yml").is_file()
    assert pkg.joinpath("pagila_ddl.sql").is_file()
    assert pkg.joinpath("pagila_data.sql").is_file()


def test_compose_path_resolver_returns_real_file() -> None:
    """The resolver used by `pretensor quickstart` must return a real path on
    disk — both in editable installs and in pip-installed wheels."""
    from pretensor.cli.commands.quickstart import _compose_path

    assert _compose_path().is_file()
