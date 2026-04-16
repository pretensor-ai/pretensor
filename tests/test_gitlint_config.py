"""``.gitlint`` and ``.gitattributes`` contract checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GITLINT_CONFIG = REPO_ROOT / ".gitlint"
GITATTRIBUTES = REPO_ROOT / ".gitattributes"


def _gitlint_cmd() -> list[str] | None:
    if shutil.which("uv"):
        return [
            "uv",
            "run",
            "--extra",
            "dev",
            "gitlint",
            "--config",
            str(GITLINT_CONFIG),
        ]
    exe = shutil.which("gitlint")
    if exe:
        return [exe, "--config", str(GITLINT_CONFIG)]
    return None


@pytest.fixture(scope="module")
def gitlint_available() -> list[str]:
    cmd = _gitlint_cmd()
    if cmd is None:
        pytest.skip("Need `uv` on PATH or a `gitlint` executable to run gitlint checks")
    return cmd


def test_gitlint_config_exists() -> None:
    assert GITLINT_CONFIG.is_file(), ".gitlint must exist at repo root"


def test_gitattributes_rules() -> None:
    assert GITATTRIBUTES.is_file(), ".gitattributes must exist at repo root"
    text = GITATTRIBUTES.read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in text
    assert "uv.lock linguist-generated=true" in text


def test_gitlint_self_validation(gitlint_available: list[str]) -> None:
    proc = subprocess.run(
        [*gitlint_available, "--debug"],
        cwd=REPO_ROOT,
        input="feat: validate gitlint config\n\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_gitlint_rejects_wip(gitlint_available: list[str]) -> None:
    proc = subprocess.run(
        gitlint_available,
        cwd=REPO_ROOT,
        input="wip\n\n",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode != 0


@pytest.mark.parametrize(
    "message",
    [
        "feat: add thing\n\n",
        "PROJ-123: feat(mcp): add retry\n\n",
    ],
)
def test_gitlint_accepts_r5_examples(
    gitlint_available: list[str], message: str
) -> None:
    proc = subprocess.run(
        gitlint_available,
        cwd=REPO_ROOT,
        input=message,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
