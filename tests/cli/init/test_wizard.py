"""Tests for wizard orchestration."""

from __future__ import annotations

from pathlib import Path

from pretensor.cli.init.wizard import InitPlan, build_plan, render_plan


def test_build_plan_uses_environment_dsn(tmp_path: Path) -> None:
    plan = build_plan(
        env={"DATABASE_URL": "postgresql://u:p@h:5432/app"},
        cwd=tmp_path,
        home=tmp_path,
        state_dir=tmp_path / ".pretensor",
    )
    assert plan.dsn == "postgresql://u:p@h:5432/app"
    assert plan.dsn_env_var == "DATABASE_URL"
    assert plan.name == "app"


def test_build_plan_without_environment_dsn(tmp_path: Path) -> None:
    plan = build_plan(env={}, cwd=tmp_path, home=tmp_path, state_dir=tmp_path)
    assert plan.dsn is None


def test_render_plan_redacts_password(tmp_path: Path) -> None:
    plan = InitPlan(
        dsn="postgresql://u:secret@h:5432/app",
        dsn_env_var="DATABASE_URL",
        name="app",
        state_dir=tmp_path,
        repo=None,
        clients=(),
    )
    rendered = render_plan(plan)
    assert "secret" not in rendered
    assert "DATABASE_URL" in rendered
    assert "—" not in rendered
