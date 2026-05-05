"""Unit tests for the MCP stdio-client wrapper.

The subprocess path is exercised in the e2e test; here we focus on
the pure helpers (``tool_result_to_text``) and the failure modes
that don't require spawning a server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from pretensor.benchmark.l3.mcp_client import (
    McpClientError,
    McpToolResult,
    StdioMcpClient,
    tool_result_to_text,
)

# ---------------------------------------------------------------------------
# tool_result_to_text
# ---------------------------------------------------------------------------


def test_prefers_structured_content_when_present() -> None:
    """``structuredContent`` round-trips as JSON, sorted for byte-stability."""
    result = CallToolResult(
        content=[TextContent(type="text", text="ignored when structured set")],
        structuredContent={"b": 2, "a": 1},
    )
    rendered = tool_result_to_text(result)
    assert json.loads(rendered) == {"a": 1, "b": 2}
    # sorted keys → deterministic ordering
    assert rendered == '{"a": 1, "b": 2}'


def test_falls_back_to_concatenated_text_blocks() -> None:
    """Without structured content, text blocks are joined verbatim."""
    result = CallToolResult(
        content=[
            TextContent(type="text", text="hello "),
            TextContent(type="text", text="world"),
        ],
    )
    assert tool_result_to_text(result) == "hello world"


def test_empty_result_renders_as_empty_object() -> None:
    """No text and no structured content → ``{}`` (well-formed JSON for the LLM)."""
    result = CallToolResult(content=[])
    assert tool_result_to_text(result) == "{}"


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_call_tool_outside_with_block_raises_clear_error() -> None:
    """Operating on the client without entering its context manager fails fast."""
    client = StdioMcpClient(graph_dir=Path("/tmp/nope"))
    with pytest.raises(McpClientError, match="outside the with-block"):
        client.call_tool("x", {})
    with pytest.raises(McpClientError, match="outside the with-block"):
        client.list_tools()


def test_exit_emits_resource_warning_when_async_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure during ``_async_close`` surfaces as a ``ResourceWarning``.

    The previous behaviour swallowed cleanup exceptions silently; the
    runner now warns so a subprocess leak is observable on stderr
    rather than hidden. Verifies the warning category, message, and
    that the loop is still torn down (``_loop`` is None afterwards).
    """
    import warnings

    from pretensor.benchmark.l3 import mcp_client as mcp_mod

    monkeypatch.setattr(mcp_mod, "_DEFAULT_STARTUP_TIMEOUT_S", 0.5)
    monkeypatch.setattr(mcp_mod, "_DEFAULT_TEARDOWN_TIMEOUT_S", 1.0)
    monkeypatch.setattr(mcp_mod, "_DEFAULT_OUTER_LOOP_TIMEOUT_S", 5.0)

    # _async_open succeeds; _async_close raises a clear, identifiable error.
    async def quick_open(self):  # type: ignore[no-untyped-def]
        return None

    async def broken_close(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated cleanup failure")

    monkeypatch.setattr(mcp_mod.StdioMcpClient, "_async_open", quick_open)
    monkeypatch.setattr(mcp_mod.StdioMcpClient, "_async_close", broken_close)

    client = StdioMcpClient(graph_dir=Path("/tmp/nope"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with client:
            pass

    rw = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert rw, f"expected a ResourceWarning, got {[w.category for w in caught]}"
    assert "teardown error" in str(rw[0].message)
    assert "simulated cleanup failure" in str(rw[0].message)
    # Loop is fully torn down regardless of the warning.
    assert client._loop is None
    assert client._loop_thread is None


def test_startup_timeout_cancels_pending_open_coroutine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled ``_async_open`` is cancelled so its cleanup branch runs.

    The fix being pinned: previously ``__enter__`` stopped the event
    loop on timeout without cancelling the in-flight open coroutine,
    so the ``except BaseException: await stack.aclose()`` branch in
    ``_async_open`` never ran and a partly-spawned subprocess could
    leak. This test makes ``_async_open`` await forever, drops the
    startup timeout to ~50ms so ``__enter__`` times out fast, and
    asserts the coroutine reached its cleanup branch (we observe a
    ``CancelledError`` propagating into a sentinel).
    """
    import asyncio as _asyncio

    from pretensor.benchmark.l3 import mcp_client as mcp_mod

    cleanup_ran = {"ok": False}

    async def stalled_open(self):  # type: ignore[no-untyped-def]
        try:
            # Block forever — emulates stdio_client hanging mid-handshake.
            await _asyncio.Event().wait()
        except BaseException:
            cleanup_ran["ok"] = True
            raise

    monkeypatch.setattr(mcp_mod.StdioMcpClient, "_async_open", stalled_open)
    # Drop the startup timeout so the test doesn't have to wait 30s.
    monkeypatch.setattr(mcp_mod, "_DEFAULT_STARTUP_TIMEOUT_S", 0.05)
    monkeypatch.setattr(mcp_mod, "_DEFAULT_TEARDOWN_TIMEOUT_S", 5.0)

    client = StdioMcpClient(graph_dir=Path("/tmp/nope"))
    with pytest.raises(McpClientError, match="Failed to start MCP subprocess"):
        client.__enter__()

    assert cleanup_ran["ok"], (
        "stalled _async_open did not receive cancellation — its cleanup "
        "branch never ran, which is the orphan-subprocess regression we "
        "are guarding against."
    )


def test_failed_subprocess_startup_raises_McpClientError() -> None:
    """A non-existent ``command`` surfaces as ``McpClientError`` from ``__enter__``.

    We pass ``command="pretensor-does-not-exist"`` so the stdio_client
    open path immediately fails. The expectation is a clean
    :class:`McpClientError` with a helpful message — not an asyncio
    traceback escaping into the runner.
    """
    client = StdioMcpClient(
        graph_dir=Path("/tmp/nope"),
        command="pretensor-binary-that-does-not-exist-xyz",
    )
    with pytest.raises(McpClientError, match="Failed to start MCP subprocess"):
        client.__enter__()


def _drive_call_tool_with_session(session: object) -> McpToolResult:
    """Boot just enough of :class:`StdioMcpClient` to invoke ``call_tool`` once.

    Spawns a private event-loop thread, hands ``call_tool`` a fake
    session, then tears the loop down. Used by the transport-failure
    tests below so they don't depend on a real subprocess.
    """
    import asyncio
    import threading as _threading

    client = StdioMcpClient(graph_dir=Path("/tmp/nope"))
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = _threading.Thread(target=_run, name="t-call-tool", daemon=True)
    thread.start()
    try:
        client._loop = loop
        client._session = session  # type: ignore[assignment]
        return client.call_tool("query", {"q": "x"})
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()


def test_call_tool_transport_failure_returns_is_error_outcome() -> None:
    """Transport-level failures inside ``call_tool`` surface as ``is_error=True``.

    The runner relies on this: a single broken MCP call must not abort
    the whole question — it lands in the trace as an error and the
    LLM's next step gets the failure message instead of an exception
    propagating up.
    """

    class _BrokenSession:
        async def call_tool(self, name, arguments=None, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated transport reset")

    outcome = _drive_call_tool_with_session(_BrokenSession())

    assert outcome.is_error is True
    # The content is JSON the LLM can parse — embeds the original
    # exception so a grader can debug from the envelope alone.
    payload = json.loads(outcome.content)
    assert "MCP transport failure" in payload["error"]
    assert "simulated transport reset" in payload["error"]


def test_call_tool_transport_failure_emits_valid_json_when_exc_has_quotes() -> None:
    """Exception text with quotes / newlines must not produce malformed JSON.

    Previously the error payload was built via raw f-string
    interpolation; an exception message carrying ``"`` or ``\\n`` would
    inject unescaped characters into the JSON literal and the LLM's
    tool-result decoder would then fail to parse it. ``json.dumps`` is
    what fixes this — this test pins it.
    """

    class _NoisySession:
        async def call_tool(self, name, arguments=None, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError('boom: "unterminated\nmessage with backslash\\here')

    outcome = _drive_call_tool_with_session(_NoisySession())

    assert outcome.is_error is True
    # The whole content must be parseable JSON — no manual escaping
    # required by the caller.
    payload = json.loads(outcome.content)
    assert "unterminated" in payload["error"]
    assert "backslash" in payload["error"]


def test_list_tools_caches_after_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second ``list_tools`` returns the cached list without a server round-trip.

    The OSS pretensor server does not register tools after startup, so
    re-asking the server on every question is pure overhead. We cover
    the cache directly here rather than depending on a live subprocess.
    """
    import asyncio
    import threading as _threading

    from pretensor.benchmark.l3 import mcp_client as mcp_mod
    from pretensor.benchmark.l3.agent import AgentTool

    call_counter = {"n": 0}

    class _FakeListResult:
        def __init__(self, tools: list[AgentTool]) -> None:
            self.tools = tools

    class _FakeSession:
        async def list_tools(self) -> _FakeListResult:
            call_counter["n"] += 1
            return _FakeListResult(
                [AgentTool(name="schema", description="", input_schema={})]
            )

    # The conversion helper on the real path turns ``mcp.types.Tool``
    # into ``AgentTool``. Our fake already returns AgentTool, so swap
    # in an identity helper.
    monkeypatch.setattr(mcp_mod, "_to_agent_tool", lambda t: t)

    client = StdioMcpClient(graph_dir=Path("/tmp/nope"))
    # Bypass __enter__: stand up only the parts list_tools touches.
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = _threading.Thread(target=_run, name="t-list-tools-cache", daemon=True)
    thread.start()
    try:
        client._loop = loop
        client._loop_thread = thread
        client._session = _FakeSession()  # type: ignore[assignment]

        first = client.list_tools()
        second = client.list_tools()
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        loop.close()

    assert call_counter["n"] == 1, "second list_tools must hit the cache"
    assert [t.name for t in first] == ["schema"]
    assert [t.name for t in second] == ["schema"]
