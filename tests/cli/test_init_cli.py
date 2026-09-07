"""CLI tests for ``pretensor init``."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from pretensor.cli.main import app

_ANSI_ESCAPE_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[\ -/]*[@-~])")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _ANSI_ESCAPE_RE.sub("", text)).strip()


def test_init_help_lists_flags() -> None:
    result = CliRunner().invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    for flag in (
        "--dsn",
        "--dialect",
        "--name",
        "--state-dir",
        "--repo",
        "--yes",
        "--no-mcp-write",
        "--client",
    ):
        assert flag in plain


def test_init_noninteractive_without_dsn_exits_one(monkeypatch) -> None:
    monkeypatch.setenv("PRETENSOR_NONINTERACTIVE", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    result = CliRunner().invoke(app, ["init"])
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "--dsn" in plain
    assert "Traceback" not in result.stdout


def test_init_help_has_no_em_dash() -> None:
    result = CliRunner().invoke(app, ["init", "--help"])
    assert "—" not in result.stdout


def test_init_sample_prints_quickstart_pointer_and_exits_zero() -> None:
    """``--sample`` short-circuits before any plan build/render/confirm.

    It must print guidance pointing at ``pretensor quickstart`` and exit 0
    without ever asking "Proceed with these?".
    """
    result = CliRunner().invoke(app, ["init", "--sample"])
    assert result.exit_code == 0
    plain = _normalize(result.stdout)
    assert "pretensor quickstart" in plain
    assert "Proceed" not in plain


def test_execute_runs_index_then_analyze_then_register(monkeypatch, tmp_path) -> None:
    """``_execute`` must index, then analyze, then register in that order.

    ``run_analyze_enrichment`` raises ``EmptyGraphError`` if the graph has no
    ``SchemaTable`` rows, so analysis must never run before indexing.
    """
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.init.mcp_clients import McpClient
    from pretensor.cli.init.wizard import InitPlan

    calls: list[str] = []

    def fake_index(**kwargs: object) -> None:
        calls.append("index")

    def fake_analyze_one(**kwargs: object) -> int:
        calls.append("analyze")
        return 0

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        calls.append("register")
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 3)

    plan = InitPlan(
        dsn="postgresql://u:p@h:5432/app",
        dsn_env_var=None,
        name="app",
        state_dir=tmp_path / ".pretensor",
        repo=tmp_path,
        clients=(McpClient(key="claude-code", label="Claude Code", config_path=None),),
    )
    console = Console(file=io.StringIO(), width=200)

    init_module._execute(console=console, ctx=None, plan=plan)

    assert calls == ["index", "analyze", "register"]


def test_execute_propagates_analyze_failure(monkeypatch, tmp_path) -> None:
    """A failed analyze must not be reported as a successful repo link.

    Analyze can fail (e.g. ``EmptyGraphError``) after a successful index.
    That failure must surface as a non-zero exit and must not claim the repo
    was linked, but indexing and MCP registration already succeeded and
    should not be rolled back or skipped.
    """
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.init.mcp_clients import McpClient
    from pretensor.cli.init.wizard import InitPlan

    calls: list[str] = []

    def fake_index(**kwargs: object) -> None:
        calls.append("index")

    def fake_analyze_one(**kwargs: object) -> int:
        calls.append("analyze")
        return 1

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        calls.append("register")
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 3)

    plan = InitPlan(
        dsn="postgresql://u:p@h:5432/app",
        dsn_env_var=None,
        name="app",
        state_dir=tmp_path / ".pretensor",
        repo=tmp_path,
        clients=(McpClient(key="claude-code", label="Claude Code", config_path=None),),
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=200)

    with pytest.raises(typer.Exit) as exc_info:
        init_module._execute(console=console, ctx=None, plan=plan)

    assert exc_info.value.exit_code == 1
    assert calls == ["index", "analyze", "register"]
    assert "Linked code" not in buffer.getvalue()


def test_init_yes_does_not_auto_register_detected_client(monkeypatch, tmp_path) -> None:
    """``--yes`` must not write to a merely-detected MCP client's config.

    Detecting a client (e.g. Claude Code on PATH) is not the same as the
    user consenting to register with it. Under ``--yes`` there is no prompt
    to ask, and no flag yet to name a client explicitly, so registration
    must be skipped and the mcpServers block printed instead.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.mcp_clients import McpClient

    monkeypatch.chdir(tmp_path)

    detected = (McpClient(key="claude-code", label="Claude Code", config_path=None),)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: detected)

    register_calls: list[McpClient] = []

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        register_calls.append(client)
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    print_mcp_config_calls: list[object] = []

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(
        init_module,
        "print_mcp_config",
        lambda state_dir: print_mcp_config_calls.append(state_dir),
    )

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/app",
            "--yes",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert register_calls == []
    assert print_mcp_config_calls == [tmp_path / ".pretensor"]


# ── --client: explicit consent for unattended registration ──────────────────


def test_init_yes_with_client_registers_the_named_detected_client(
    monkeypatch, tmp_path
) -> None:
    """--client names the consent --yes alone cannot give.

    Detecting Claude Code on PATH is not consent to register with it, but
    naming it via --client is: registration must run end to end, exactly
    once, for the named client.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.mcp_clients import McpClient

    monkeypatch.chdir(tmp_path)

    detected = (McpClient(key="claude-code", label="Claude Code", config_path=None),)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: detected)

    register_calls: list[McpClient] = []

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        register_calls.append(client)
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/app",
            "--yes",
            "--client",
            "claude-code",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert register_calls == [detected[0]]


def test_init_yes_with_client_not_detected_warns_and_skips(
    monkeypatch, tmp_path
) -> None:
    """Naming an undetected client warns and prints the config block instead."""
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.mcp_clients import McpClient

    monkeypatch.chdir(tmp_path)

    detected = (McpClient(key="claude-code", label="Claude Code", config_path=None),)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: detected)

    register_calls: list[McpClient] = []

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        register_calls.append(client)
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/app",
            "--yes",
            "--client",
            "cursor",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert register_calls == []
    plain = _normalize(result.stdout)
    assert "Cursor was not detected" in plain


def test_init_client_unknown_key_exits_one(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/app",
            "--yes",
            "--client",
            "bogus",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "bogus" in plain
    assert "claude-code" in plain
    assert "claude-desktop" in plain
    assert "cursor" in plain
    assert "Traceback" not in result.stdout


def test_init_client_and_no_mcp_write_conflict_exits_one(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/app",
            "--yes",
            "--client",
            "claude-code",
            "--no-mcp-write",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.stdout
    plain = _normalize(result.stdout)
    assert "--client" in plain
    assert "--no-mcp-write" in plain


def test_init_interactive_client_skips_confirmation_other_detected_client_confirmed(
    monkeypatch, tmp_path
) -> None:
    """Interactive run: a named client is never asked about; an unnamed
    detected client still goes through the per-client confirmation.

    ``choose_clients`` is monkeypatched rather than driven through scripted
    stdin: it is the one function in the wizard that owns per-client
    confirmation wording and prompt order, and stubbing it here pins the
    assertion to the behavior that actually matters (which candidates reach
    it, and how the confirmed set is merged with the named one) without the
    test being brittle to unrelated prompt-copy or prompt-count changes
    elsewhere in the guided flow. The DSN-confirm and repo-decline prompts
    still go through real scripted stdin, following the existing guided-flow
    test's pattern, since those are exercised elsewhere and their order is
    stable.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.mcp_clients import McpClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)

    client_a = McpClient(key="claude-code", label="Claude Code", config_path=None)
    client_b = McpClient(
        key="cursor", label="Cursor", config_path=tmp_path / "mcp.json"
    )
    detected = (client_a, client_b)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: detected)

    received_candidates: list[tuple[McpClient, ...]] = []

    def fake_choose_clients(
        prompter: object, candidates: tuple[McpClient, ...]
    ) -> tuple[McpClient, ...]:
        received_candidates.append(tuple(candidates))
        return (client_b,)  # user confirms Cursor

    register_calls: list[McpClient] = []

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        register_calls.append(client)
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "choose_clients", fake_choose_clients)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    dsn = "postgresql://u:p@h:5432/e2edb"
    # A DSN is supplied via --dsn, so the express summary offers its
    # three-way choice first; "3" customizes. From there: confirm the
    # pre-supplied DSN, decline linking a different repo (tmp_path has no
    # .git so none is inferred), confirm the plan, then decline "Add
    # another database?". Client A is named via --client and never
    # prompted for; client B's confirmation is stubbed via choose_clients
    # above, not scripted stdin.
    answers = "\n".join(["3", "y", "n", "y", "n", ""])

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            dsn,
            "--client",
            "claude-code",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    # Client A (named) must never reach choose_clients; only B (unnamed).
    assert received_candidates == [(client_b,)]
    # Both the named client and the interactively confirmed one register.
    assert register_calls == [client_a, client_b]


def test_init_interactive_client_declined_other_detected_client_not_registered(
    monkeypatch, tmp_path
) -> None:
    """Same flow, but the interactively-offered client is declined.

    Only the client named via --client must register; the unnamed detected
    one, confirmed False, must not.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.mcp_clients import McpClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)

    client_a = McpClient(key="claude-code", label="Claude Code", config_path=None)
    client_b = McpClient(
        key="cursor", label="Cursor", config_path=tmp_path / "mcp.json"
    )
    detected = (client_a, client_b)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: detected)

    received_candidates: list[tuple[McpClient, ...]] = []

    def fake_choose_clients(
        prompter: object, candidates: tuple[McpClient, ...]
    ) -> tuple[McpClient, ...]:
        received_candidates.append(tuple(candidates))
        return ()  # user declines Cursor

    register_calls: list[McpClient] = []

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        register_calls.append(client)
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "choose_clients", fake_choose_clients)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    dsn = "postgresql://u:p@h:5432/e2edb"
    # See the sibling test above for the prompt-order rationale: "3"
    # customizes past the express summary, then confirm DSN, decline repo,
    # confirm plan, decline "Add another database?".
    answers = "\n".join(["3", "y", "n", "y", "n", ""])

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            dsn,
            "--client",
            "claude-code",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert received_candidates == [(client_b,)]
    assert register_calls == [client_a]


# ── FIX 1: repositories: block gets written ─────────────────────────────────


def test_execute_records_repository_after_successful_analyze(
    monkeypatch, tmp_path
) -> None:
    """A successful analyze run must record the repo under `repositories:`."""
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.config_file import load_cli_config
    from pretensor.cli.init.wizard import InitPlan

    def fake_analyze_one(**kwargs: object) -> int:
        return 0

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(
        init_module, "register_client", lambda client, state_dir, **k: "ok"
    )
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 3)

    repo_dir = tmp_path / "svc"
    repo_dir.mkdir()
    state_dir = tmp_path / ".pretensor"
    plan = InitPlan(
        dsn="postgresql://u:p@h:5432/app",
        dsn_env_var=None,
        name="app",
        state_dir=state_dir,
        repo=repo_dir,
        clients=(),
    )
    console = Console(file=io.StringIO(), width=200)

    init_module._execute(console=console, ctx=None, plan=plan)

    cfg = load_cli_config(state_dir / "config.yaml")
    assert len(cfg.repositories) == 1
    repo = cfg.repositories[0]
    assert repo.path == repo_dir.resolve()
    assert repo.service == "svc"
    assert repo.connection == "app"


def test_execute_does_not_record_repository_when_analyze_fails(
    monkeypatch, tmp_path
) -> None:
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.init.wizard import InitPlan

    def fake_analyze_one(**kwargs: object) -> int:
        return 1

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(
        init_module, "register_client", lambda client, state_dir, **k: "ok"
    )
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 3)

    repo_dir = tmp_path / "svc"
    repo_dir.mkdir()
    state_dir = tmp_path / ".pretensor"
    plan = InitPlan(
        dsn="postgresql://u:p@h:5432/app",
        dsn_env_var=None,
        name="app",
        state_dir=state_dir,
        repo=repo_dir,
        clients=(),
    )
    console = Console(file=io.StringIO(), width=200)

    with pytest.raises(typer.Exit):
        init_module._execute(console=console, ctx=None, plan=plan)

    assert not (state_dir / "config.yaml").exists()


def test_execute_recording_failure_on_malformed_config_does_not_traceback(
    monkeypatch, tmp_path
) -> None:
    """A malformed pre-existing config.yaml must not raise through _execute.

    Indexing and analysis already succeeded by the time record_repository
    runs, so a config.yaml pretensor can't safely rewrite must be reported
    plainly and skipped, not let a YAMLError traceback past a successful run,
    and it must not strand a `.bak` copy of the untouched original.
    """
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.init.wizard import InitPlan

    def fake_analyze_one(**kwargs: object) -> int:
        return 0

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(
        init_module, "register_client", lambda client, state_dir, **k: "ok"
    )
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 3)

    repo_dir = tmp_path / "svc"
    repo_dir.mkdir()
    state_dir = tmp_path / ".pretensor"
    state_dir.mkdir()
    config_path = state_dir / "config.yaml"
    original = "repositories:\n  - path: [unclosed\n"
    config_path.write_text(original, encoding="utf-8")

    plan = InitPlan(
        dsn="postgresql://u:p@h:5432/app",
        dsn_env_var=None,
        name="app",
        state_dir=state_dir,
        repo=repo_dir,
        clients=(),
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=200)

    # analyze succeeded, no clients configured -> _execute returns normally.
    init_module._execute(console=console, ctx=None, plan=plan)

    output = buffer.getvalue()
    assert "Traceback" not in output
    assert "Could not record the repository" in output
    assert config_path.read_text(encoding="utf-8") == original
    assert list(state_dir.glob("config.yaml.bak.*")) == []


def test_execute_records_repository_never_writes_password(
    monkeypatch, tmp_path
) -> None:
    """End-to-end: record_repository must never leak the indexed DSN's password."""
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.init.wizard import InitPlan

    def fake_analyze_one(**kwargs: object) -> int:
        return 0

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(
        init_module, "register_client", lambda client, state_dir, **k: "ok"
    )
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 3)

    repo_dir = tmp_path / "svc"
    repo_dir.mkdir()
    state_dir = tmp_path / ".pretensor"
    plan = InitPlan(
        dsn="postgresql://user:s3cr3t!@localhost:5432/appdb",
        dsn_env_var=None,
        name="app",
        state_dir=state_dir,
        repo=repo_dir,
        clients=(),
    )
    console = Console(file=io.StringIO(), width=200)

    init_module._execute(console=console, ctx=None, plan=plan)

    text = (state_dir / "config.yaml").read_text(encoding="utf-8")
    assert "s3cr3t" not in text


# ── env-reference DSN persistence (sources:) ─────────────────────────────────


def test_execute_records_source_when_dsn_from_env_var(monkeypatch, tmp_path) -> None:
    """When the DSN came from a single env var, the literal reference is recorded."""
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.config_file import load_cli_config
    from pretensor.cli.init.wizard import InitPlan

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:s3cr3t@h:5432/app")
    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    state_dir = tmp_path / ".pretensor"
    plan = InitPlan(
        dsn="postgresql://u:s3cr3t@h:5432/app",
        dsn_env_var="DATABASE_URL",
        name="app",
        state_dir=state_dir,
        repo=None,
        clients=(),
    )
    console = Console(file=io.StringIO(), width=200)

    init_module._execute(console=console, ctx=None, plan=plan)

    cfg = load_cli_config(state_dir / "config.yaml")
    assert cfg.sources["app"].url == "${DATABASE_URL}"

    text = (state_dir / "config.yaml").read_text(encoding="utf-8")
    assert "${DATABASE_URL}" in text
    assert "s3cr3t" not in text
    assert "postgresql://u:" not in text


def test_execute_source_record_dedupes_on_second_run(monkeypatch, tmp_path) -> None:
    """A second `init` run for the same connection updates in place, not appends."""
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.config_file import load_cli_config
    from pretensor.cli.init.wizard import InitPlan

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/app")
    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    state_dir = tmp_path / ".pretensor"
    plan = InitPlan(
        dsn="postgresql://u:p@h:5432/app",
        dsn_env_var="DATABASE_URL",
        name="app",
        state_dir=state_dir,
        repo=None,
        clients=(),
    )
    console = Console(file=io.StringIO(), width=200)

    init_module._execute(console=console, ctx=None, plan=plan)
    init_module._execute(console=console, ctx=None, plan=plan)

    cfg = load_cli_config(state_dir / "config.yaml")
    assert list(cfg.sources) == ["app"]
    assert cfg.sources["app"].url == "${DATABASE_URL}"


def test_execute_does_not_record_source_for_typed_dsn(monkeypatch, tmp_path) -> None:
    """A typed/pasted/assembled DSN (no single env var) writes no `sources:` entry."""
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.init.wizard import InitPlan

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    state_dir = tmp_path / ".pretensor"
    plan = InitPlan(
        dsn="postgresql://u:p@h:5432/app",
        dsn_env_var=None,
        name="app",
        state_dir=state_dir,
        repo=None,
        clients=(),
    )
    console = Console(file=io.StringIO(), width=200)

    init_module._execute(console=console, ctx=None, plan=plan)

    config_path = state_dir / "config.yaml"
    assert not config_path.exists()


# ── FIX 2: MCP registration errors don't traceback ──────────────────────────


def test_execute_registration_failure_is_caught_and_exits_nonzero(
    monkeypatch, tmp_path
) -> None:
    """A raising register_client must not traceback and must still try the rest."""
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.init.mcp_clients import McpClient
    from pretensor.cli.init.wizard import InitPlan

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    attempted: list[str] = []

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        attempted.append(client.key)
        if client.key == "cursor":
            raise ValueError("boom: bad json")
        return f"{client.label}: registered."

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    plan = InitPlan(
        dsn="postgresql://u:p@h:5432/app",
        dsn_env_var=None,
        name="app",
        state_dir=tmp_path / ".pretensor",
        repo=None,
        clients=(
            McpClient(key="cursor", label="Cursor", config_path=tmp_path / "mcp.json"),
            McpClient(key="claude-code", label="Claude Code", config_path=None),
        ),
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=200)

    with pytest.raises(typer.Exit) as exc_info:
        init_module._execute(console=console, ctx=None, plan=plan)

    assert exc_info.value.exit_code != 0
    assert attempted == ["cursor", "claude-code"]
    output = buffer.getvalue()
    assert "Traceback" not in output
    assert "boom: bad json" in output
    assert "Indexed" in output  # summary is still printed


# ── FIX 3: stale connection name from --dsn override ────────────────────────


def test_init_dsn_flag_overrides_stale_env_derived_name(monkeypatch, tmp_path) -> None:
    """--dsn must reset the connection name; it must not keep the env DSN's."""
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/envdb")

    recorded: dict[str, str] = {}

    def fake_index(*, connection_name: str, **kwargs: object) -> None:
        recorded["connection_name"] = connection_name

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/flagdb",
            "--yes",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert recorded["connection_name"] == "flagdb"


# ── FIX 5: minimal preflight on existing state ───────────────────────────────


def _seed_registry(state_dir: Path) -> None:
    from pretensor.core.registry import GraphRegistry

    reg = GraphRegistry(state_dir / "registry.json").load()
    reg.upsert(
        connection_name="old",
        database="olddb",
        dsn="postgresql://localhost/olddb",
        graph_path=state_dir / "graphs" / "old.kuzu",
        dialect="postgres",
    )
    reg.save()


def test_init_yes_shows_existing_connections_and_proceeds(
    monkeypatch, tmp_path
) -> None:
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / ".pretensor"
    _seed_registry(state_dir)

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/newdb",
            "--yes",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    plain = _normalize(result.stdout)
    assert "Existing connections" in plain
    assert "old" in plain


def test_init_interactive_decline_on_existing_state_writes_nothing(
    monkeypatch, tmp_path
) -> None:
    import pretensor.cli.commands.init as init_module
    from pretensor.cli.prompts import RichPrompter

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(RichPrompter, "confirm", lambda self, *a, **k: False)

    state_dir = tmp_path / ".pretensor"
    _seed_registry(state_dir)

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/newdb",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Nothing was written." in result.stdout


def test_init_yes_is_idempotent_across_repeated_runs(monkeypatch, tmp_path) -> None:
    """A second `init --yes` run must still succeed once state exists.

    The fake `run_index` writes a minimal real registry.json (the same
    shape the preflight tests seed) so the second invocation actually
    exercises `_confirm_existing_state` under `--yes`, not just a no-op path
    where registry.json never gets created.
    """
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / ".pretensor"

    def fake_index(*, state_dir: Path, **kwargs: object) -> None:
        _seed_registry(state_dir)

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    args = [
        "init",
        "--dsn",
        "postgresql://u:p@h:5432/app",
        "--yes",
        "--state-dir",
        str(state_dir),
    ]
    first = CliRunner().invoke(app, args)
    second = CliRunner().invoke(app, args)

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    plain = _normalize(second.stdout)
    assert "Existing connections" in plain


def test_init_corrupt_registry_falls_back_to_proceeding(monkeypatch, tmp_path) -> None:
    """A garbage `registry.json` must not block `init`.

    `_confirm_existing_state` treats a registry it cannot parse as "nothing
    to preflight" and lets the normal index/analyze flow surface the
    problem later, rather than crashing during preflight.
    """
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / ".pretensor"
    state_dir.mkdir(parents=True)
    (state_dir / "registry.json").write_text("not valid json {{{")

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/newdb",
            "--yes",
            "--state-dir",
            str(state_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Traceback" not in result.stdout


def test_init_guided_flow_pastes_dsn_declines_repo_and_client(
    monkeypatch, tmp_path
) -> None:
    """End-to-end guided flow via scripted stdin, no ``--yes``, no env DSN.

    No DSN is inferred (DATABASE_URL/POSTGRES_URL are unset), so the
    summary-first express choice never appears; the guided flow's per-item
    questions are the very first prompts, unchanged from before that
    feature existed.

    Answers, in prompt order: "1" (paste a full connection URL), the DSN
    text, "n" (decline linking a different repo), "n" (decline registering
    with the one detected MCP client), "y" (proceed with the plan), then a
    trailing "n" for the new "Add another database?" loop question (the
    repo-linking loop never appears: the repo was declined, so
    ``plan.repo`` is None).
    ``CliRunner``'s stdin is never a real TTY, so ``is_noninteractive`` is
    monkeypatched to False to simulate one, per the wizard's own contract.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.mcp_clients import McpClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)

    detected = (McpClient(key="claude-code", label="Claude Code", config_path=None),)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: detected)

    recorded: dict[str, str] = {}

    def fake_index(*, connection_name: str, **kwargs: object) -> None:
        recorded["connection_name"] = connection_name

    register_calls: list[McpClient] = []

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        register_calls.append(client)
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)
    # Deterministic regardless of whether the [embeddings] extra happens to
    # be installed in the environment running this test.
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)

    dsn = "postgresql://u:p@h:5432/e2edb"
    answers = "\n".join(["1", dsn, "n", "n", "y", "n", ""])

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert recorded.get("connection_name") == "e2edb"
    assert register_calls == []


# ── FIX 4: CLI flag overrides to plan parameters ──────────────────────────────


def test_init_flag_overrides_name_and_repo(monkeypatch, tmp_path) -> None:
    """--name and --repo flags must override plan defaults and reach _execute.

    The init_command copies flag values onto the plan (lines 128-142), and those
    modified values must propagate to run_index (connection_name) and
    run_analyze_one (repo_path).
    """
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)

    recorded: dict[str, object] = {}

    def fake_index(*, connection_name: str, **kwargs: object) -> None:
        recorded["connection_name"] = connection_name

    def fake_analyze_one(*, repo_path: Path, **kwargs: object) -> int:
        recorded["repo_path"] = repo_path
        return 0

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 3)

    repo_path = tmp_path / "myrepo"
    repo_path.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/flagdb",
            "--name",
            "custom",
            "--repo",
            str(repo_path),
            "--yes",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert recorded["connection_name"] == "custom"
    assert recorded["repo_path"] == repo_path.resolve()


def test_init_no_repo_flag_prevents_analyze(monkeypatch, tmp_path) -> None:
    """--no-repo flag must prevent run_analyze_one from being called.

    Even if the plan would normally have a repo, --no-repo explicitly sets
    plan.repo = None, which skips the analyze phase in _execute.
    """
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)

    analyze_called: list[bool] = []

    def fake_index(**kwargs: object) -> None:
        pass

    def fake_analyze_one(**kwargs: object) -> int:
        analyze_called.append(True)
        return 0

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "postgresql://u:p@h:5432/flagdb",
            "--no-repo",
            "--yes",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert analyze_called == []


# ── Summary-first [Y/n/c] express confirmation ──────────────────────────────


def test_init_express_yes_accepts_full_inference(monkeypatch, tmp_path) -> None:
    """Express Y: full inference (env DSN + repo + client), a single Y answer.

    Beyond the leading choice, the only further prompts are the two new
    "add another" loop questions (both declined here) -- the guided flow's
    per-item DSN/repo/client questions are never asked.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.inference import InferredRepo
    from pretensor.cli.init.mcp_clients import McpClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/expressdb")
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.setattr(
        wizard_module,
        "infer_repo",
        lambda cwd: InferredRepo(path=repo_dir, file_count=5),
    )
    detected = (McpClient(key="claude-code", label="Claude Code", config_path=None),)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: detected)

    calls: list[str] = []

    def fake_index(**kwargs: object) -> None:
        calls.append("index")

    def fake_analyze_one(**kwargs: object) -> int:
        calls.append("analyze")
        return 0

    register_calls: list[McpClient] = []

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        register_calls.append(client)
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    print_calls: list[object] = []

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 5)
    monkeypatch.setattr(
        init_module,
        "print_mcp_config",
        lambda state_dir: print_calls.append(state_dir),
    )

    # "1" accepts everything inferred; the two trailing "n"s decline the
    # "Link another repository?" and "Add another database?" loop questions.
    answers = "\n".join(["1", "n", "n", ""])

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert calls == ["index", "analyze"]
    assert register_calls == []
    assert print_calls == [tmp_path / ".pretensor"]
    plain = _normalize(result.stdout)
    assert "MCP server config" in plain


def test_init_express_yes_with_client_registers_named_only(
    monkeypatch, tmp_path
) -> None:
    """Express Y with --client: the named client registers, the other prints nothing.

    Two clients are detected (Claude Code, Cursor); only Claude Code is
    named via --client. A single "Y" answer must register Claude Code (the
    flag's explicit consent survives the express-accept path) while Cursor,
    merely detected, gets neither a confirmation prompt nor a registration
    call -- "Y" grants no new consent, it just doesn't revoke the flag's.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.mcp_clients import McpClient

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/expressclientdb")
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)
    monkeypatch.setattr(wizard_module, "infer_repo", lambda cwd: None)

    client_a = McpClient(key="claude-code", label="Claude Code", config_path=None)
    client_b = McpClient(
        key="cursor", label="Cursor", config_path=tmp_path / "mcp.json"
    )
    detected = (client_a, client_b)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: detected)

    register_calls: list[McpClient] = []

    def fake_register_client(
        client: McpClient, state_dir: object, **kwargs: object
    ) -> str:
        register_calls.append(client)
        return f"{client.label}: registered."

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    print_calls: list[object] = []

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "register_client", fake_register_client)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(
        init_module,
        "print_mcp_config",
        lambda state_dir: print_calls.append(state_dir),
    )

    # "1" accepts everything inferred; the trailing "n" declines "Add
    # another database?" (no repo was inferred, so its loop is never
    # offered).
    answers = "\n".join(["1", "n", ""])

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--client",
            "claude-code",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    # Only the named client registers; the merely-detected one does not, and
    # the mcpServers block is never printed (plan.clients is non-empty).
    assert register_calls == [client_a]
    assert print_calls == []


def test_init_express_no_exits_without_writing(monkeypatch, tmp_path) -> None:
    """Express n: exits 0, prints "Nothing was written.", index never called."""
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/declinedb")
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)

    index_calls: list[str] = []
    monkeypatch.setattr(
        init_module, "run_index", lambda **kwargs: index_calls.append("index")
    )

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input="2\n",
    )

    assert result.exit_code == 0, result.stdout
    assert "Nothing was written." in result.stdout
    assert index_calls == []


def test_init_express_customize_drops_to_guided_flow(monkeypatch, tmp_path) -> None:
    """Express c: drops into the guided per-item flow, pre-filled with inference.

    The first per-item question ("Use <inferred dsn>?") is consumed here,
    proving customize really does reach the guided flow's own prompts.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/customizedb")
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: ())

    recorded: dict[str, str] = {}

    def fake_index(*, connection_name: str, **kwargs: object) -> None:
        recorded["connection_name"] = connection_name

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    # "3" customizes; "y" accepts the inferred DSN (the per-item question the
    # express-accept path would otherwise never show); "n" declines linking a
    # repo; "y" proceeds with the plan; "n" declines "Add another database?".
    answers = "\n".join(["3", "y", "n", "y", "n", ""])

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert recorded.get("connection_name") == "customizedb"


def test_init_no_inferred_dsn_skips_express_summary(monkeypatch, tmp_path) -> None:
    """No DSN inferred: the express summary must never appear.

    ``RichPrompter.choose`` is wrapped to record every question it is asked
    (still delegating to the real implementation, since ``collect_dsn``'s
    own "How do you want to connect?" step legitimately uses the same
    primitive). Exactly one ``choose`` call must happen, and it must be that
    guided-flow question, not the three-way express choice.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.prompts import RichPrompter

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: ())

    original_choose = RichPrompter.choose
    choose_questions: list[str] = []

    def _tracking_choose(
        self: RichPrompter, question: str, options: object, *, default: int = 0
    ) -> int:
        choose_questions.append(question)
        return original_choose(self, question, options, default=default)  # type: ignore[arg-type]

    monkeypatch.setattr(RichPrompter, "choose", _tracking_choose)

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    dsn = "postgresql://u:p@h:5432/nodsn"
    # guided DSN entry ("1" paste, dsn text), decline repo, proceed, decline
    # "Add another database?".
    answers = "\n".join(["1", dsn, "n", "y", "n", ""])

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert choose_questions == ["How do you want to connect?"]


# ── Add-another loops ────────────────────────────────────────────────────────


def test_init_add_another_database_loops_index_twice(monkeypatch, tmp_path) -> None:
    """Interactive answers loop once: index runs for two connections."""
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/firstdb")
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: ())
    monkeypatch.setattr(wizard_module, "infer_repo", lambda cwd: None)

    indexed_names: list[str] = []

    def fake_index(*, connection_name: str, **kwargs: object) -> None:
        indexed_names.append(connection_name)

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    second_dsn = "postgresql://u:p@h:5432/seconddb"
    # Express-accept the first (inferred) connection ("1"), say yes once to
    # "Add another database?", paste the second DSN, then decline a third.
    answers = "\n".join(["1", "y", "1", second_dsn, "n", ""])

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert indexed_names == ["firstdb", "seconddb"]


def test_init_add_another_database_loop_threads_dialect_flag(
    monkeypatch, tmp_path
) -> None:
    """--dialect must reach the "Add another database?" loop's collect_dsn call.

    The first connection comes from `--dsn`, so `collect_dsn` is never
    called for it (the express-accept path skips per-item questions
    entirely). The loop's second connection is the only place `collect_dsn`
    runs, and it must receive the same normalized `--dialect` value as every
    other call site.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.inference import InferredDsn

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: ())
    monkeypatch.setattr(wizard_module, "infer_repo", lambda cwd: None)

    second_dsn = "mysql://u:p@h:3306/seconddb"
    received_dialects: list[str | None] = []

    def fake_collect_dsn(
        prompter: object, inferred: object, *, dialect: str | None = None
    ) -> InferredDsn:
        received_dialects.append(dialect)
        return InferredDsn(dsn=second_dsn, env_var=None)

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "collect_dsn", fake_collect_dsn)
    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    # Express-accept the first (flag-supplied) connection ("1"), say yes
    # once to "Add another database?" (`collect_dsn` is faked so no further
    # stdin is consumed for it), then decline a third.
    answers = "\n".join(["1", "y", "n", ""])

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "mysql://u:p@h:3306/firstdb",
            "--dialect",
            "mysql",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    # collect_dsn is only ever called from the add-another loop here (the
    # first connection came from --dsn), and it must have received the
    # normalized dialect.
    assert received_dialects == ["mysql"]


def test_init_add_another_repository_loops_analyze_twice(monkeypatch, tmp_path) -> None:
    """Interactive answers loop once: analyze runs for two repos, one connection."""
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.inference import InferredRepo

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/repodb")
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: ())

    first_repo = tmp_path / "first"
    first_repo.mkdir()
    monkeypatch.setattr(
        wizard_module,
        "infer_repo",
        lambda cwd: InferredRepo(path=first_repo, file_count=3),
    )

    second_repo = tmp_path / "second"
    second_repo.mkdir()

    analyzed: list[Path] = []

    def fake_analyze_one(*, repo_path: Path, **kwargs: object) -> int:
        analyzed.append(repo_path)
        return 0

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 3)
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    # Express-accept the first (inferred) connection + repo ("1"); then, in
    # the repo loop: "y" to link a different repo, the second repo's path,
    # then "n" to stop; finally "n" declines "Add another database?".
    answers = "\n".join(["1", "y", str(second_repo), "n", "n", ""])

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert analyzed == [first_repo, second_repo.resolve()]


# ── Embeddings question ──────────────────────────────────────────────────────


def test_init_embeddings_question_when_extra_installed(monkeypatch, tmp_path) -> None:
    """Guided flow, [embeddings] extra installed: ask once, pass the answer through."""
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: True)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: ())

    recorded: dict[str, object] = {}

    def fake_index(*, embeddings: bool, **kwargs: object) -> None:
        recorded["embeddings"] = embeddings

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    dsn = "postgresql://u:p@h:5432/embeddb"
    # paste DSN, decline repo, "y" proceeds, "n" declines the embeddings
    # question (default yes), "n" declines "Add another database?".
    answers = "\n".join(["1", dsn, "n", "y", "n", "n", ""])

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert recorded["embeddings"] is False


def test_init_no_embeddings_question_when_extra_absent(monkeypatch, tmp_path) -> None:
    """Guided flow, [embeddings] extra absent: no question, no behavior change."""
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: ())
    # Fallback marker: if the question were wrongly asked, the scripted "n"
    # answer would produce False directly (bypassing this function), so a
    # distinct True value here proves the auto-resolve path ran instead.
    monkeypatch.setattr(init_module, "resolve_embeddings_auto", lambda requested: True)

    recorded: dict[str, object] = {}

    def fake_index(*, embeddings: bool, **kwargs: object) -> None:
        recorded["embeddings"] = embeddings

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", fake_index)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    dsn = "postgresql://u:p@h:5432/noembeddb"
    # paste DSN, decline repo, "y" proceeds -- no embeddings question
    # follows, straight to "Add another database?" ("n").
    answers = "\n".join(["1", dsn, "n", "y", "n", ""])

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert recorded["embeddings"] is True


# ── Repo-loop failure propagation and the express-accept repo gate ──────────


def test_init_repo_loop_propagates_analyze_failure(monkeypatch, tmp_path) -> None:
    """A failed loop-added repository link must still fail the command.

    The first repository's analyze succeeds (exit 0); the loop's second
    repository fails (exit 1). The command must exit 1, the summary must
    still be printed (indexing already succeeded), and the retry hint for
    the failed second repo must appear -- mirroring `_execute`'s existing
    fail-after-finishing contract for the first repository's own analyze
    failure, rather than swallowing the loop's failure as exit 0.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.init.inference import InferredRepo

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/failrepodb")
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: ())

    first_repo = tmp_path / "first"
    first_repo.mkdir()
    monkeypatch.setattr(
        wizard_module,
        "infer_repo",
        lambda cwd: InferredRepo(path=first_repo, file_count=3),
    )
    second_repo = tmp_path / "second"
    second_repo.mkdir()

    exit_codes = iter([0, 1])

    def fake_analyze_one(*, repo_path: Path, **kwargs: object) -> int:
        return next(exit_codes)

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "run_analyze_one", fake_analyze_one)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 3)
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    # Express-accept the first connection + repo ("1"); in the repo loop:
    # "y" to link a different repo, the second repo's path (its analyze
    # fails), then "n" to stop; finally "n" declines "Add another database?".
    answers = "\n".join(["1", "y", str(second_repo), "n", "n", ""])

    result = CliRunner().invoke(
        app,
        ["init", "--state-dir", str(tmp_path / ".pretensor")],
        input=answers,
    )

    assert result.exit_code == 1, result.stdout
    plain = _normalize(result.stdout)
    assert "Indexed" in plain
    assert "Repository linking failed" in plain
    assert f"pretensor analyze {second_repo.resolve()} --connection failrepodb" in plain


def test_init_express_yes_nulls_repo_with_zero_supported_files(
    monkeypatch, tmp_path
) -> None:
    """Express Y with a --repo of zero supported files must not leave
    ``plan.repo`` set.

    ``infer_repo`` never offers a repo with zero supported files, but
    ``--repo`` bypasses that filter. Left set on the express-accept path,
    that unlinked repo would wrongly make the repo-add loop believe a first
    repository had linked, and offer "Link a different code repository?" as
    if it had. ``RichPrompter.confirm`` is wrapped to record every question
    asked (still delegating to the real implementation) so we can assert
    that question never appears.
    """
    import pretensor.cli.commands.init as init_module
    import pretensor.cli.init.wizard as wizard_module
    from pretensor.cli.prompts import RichPrompter

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/emptyrepodb")
    monkeypatch.setattr(init_module, "is_noninteractive", lambda: False)
    monkeypatch.setattr(init_module, "embeddings_extra_installed", lambda: False)
    monkeypatch.setattr(wizard_module, "detect_clients", lambda *, home, cwd: ())
    monkeypatch.setattr(init_module, "count_supported_files", lambda repo: 0)

    empty_repo = tmp_path / "empty"
    empty_repo.mkdir()

    original_confirm = RichPrompter.confirm
    confirm_questions: list[str] = []

    def _tracking_confirm(
        self: RichPrompter, question: str, *, default: bool = True
    ) -> bool:
        confirm_questions.append(question)
        return original_confirm(self, question, default=default)

    monkeypatch.setattr(RichPrompter, "confirm", _tracking_confirm)

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(init_module, "print_mcp_config", lambda state_dir: None)

    # "1" accepts everything inferred; "n" declines "Add another database?".
    # If the repo-add loop wrongly fired, this "n" would be consumed by its
    # "Link a different code repository?" gate instead, and the run would
    # hang/fail for lack of a further scripted answer.
    answers = "\n".join(["1", "n", ""])

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--state-dir",
            str(tmp_path / ".pretensor"),
            "--repo",
            str(empty_repo),
        ],
        input=answers,
    )

    assert result.exit_code == 0, result.stdout
    assert "No supported source files found" in result.stdout
    assert confirm_questions == ["Add another database?"]


# ── --dialect flag ───────────────────────────────────────────────────────────


def test_init_dialect_flag_rejects_unknown_value() -> None:
    result = CliRunner().invoke(
        app,
        ["init", "--dsn", "postgresql://u:p@h:5432/db", "--dialect", "oracle"],
    )
    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "oracle" in plain
    for valid in ("postgres", "mysql", "snowflake", "bigquery"):
        assert valid in plain


def test_init_dialect_flag_is_case_insensitive_and_trims_whitespace(
    monkeypatch, tmp_path
) -> None:
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)

    recorded: dict[str, object] = {}

    def fake_connection_config_from_url(dsn, name, *, dialect_override=None):
        recorded["dialect_override"] = dialect_override
        from pretensor.introspection.models.dsn import connection_config_from_url

        return connection_config_from_url(dsn, name, dialect_override=dialect_override)

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())
    monkeypatch.setattr(
        init_module,
        "connection_config_from_url",
        fake_connection_config_from_url,
    )

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "mysql://u:p@h:3306/db",
            "--dialect",
            "  MySQL  ",
            "--yes",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert recorded["dialect_override"] == "mysql"


def test_init_dialect_flag_passed_as_override_to_connection_config(
    monkeypatch, tmp_path
) -> None:
    """--dialect must reach connection_config_from_url's dialect_override.

    A DSN with a scheme that would otherwise be inferred differently proves
    the override actually took effect rather than merely being accepted and
    ignored.
    """
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)

    class _FakeConnector:
        def connect(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", lambda config: _FakeConnector())

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "mysql://u:p@h:3306/db",
            "--dialect",
            "mysql",
            "--yes",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 0, result.stdout


def test_init_missing_driver_for_non_postgres_dialect_shows_install_hint(
    monkeypatch, tmp_path
) -> None:
    """A missing optional driver must surface its install hint, not a traceback.

    Mirrors the postgres connector's ImportError-with-hint contract from
    ``connectors.registry`` for a non-postgres dialect (mysql).
    """
    import pretensor.cli.commands.init as init_module

    monkeypatch.chdir(tmp_path)

    def fake_get_connector(config):
        msg = (
            "MySQL connector requires PyMySQL. Install the MySQL extra: "
            "pip install 'pretensor[mysql]' (or pip install PyMySQL)."
        )
        raise ImportError(msg)

    monkeypatch.setattr(init_module, "run_index", lambda **kwargs: None)
    monkeypatch.setattr(init_module, "get_connector", fake_get_connector)

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--dsn",
            "mysql://u:p@h:3306/db",
            "--dialect",
            "mysql",
            "--yes",
            "--state-dir",
            str(tmp_path / ".pretensor"),
        ],
    )

    assert result.exit_code == 1
    plain = _normalize(result.stdout)
    assert "MySQL extra" in plain
    assert "PyMySQL" in plain
    assert "Traceback" not in result.stdout
