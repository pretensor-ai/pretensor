"""Sync wrapper around the MCP stdio client for the L3 pretensor runner.

The MCP Python SDK (``mcp``) is async-only — :class:`mcp.ClientSession`
exposes coroutines for ``initialize``, ``list_tools``, and ``call_tool``,
and the stdio transport is an async context manager. The L3 pretensor
runner is otherwise sync (mirrors the baseline runner shape and the
``BenchmarkResult`` writer), so we host one event loop in a daemon
worker thread and forward every operation through it.

Subprocess ownership lives in the SDK's :func:`stdio_client` context
manager: entering the context spawns ``pretensor serve …`` and pipes
its stdio; exiting sends EOF, waits for shutdown, and reaps the
subprocess. All paths through :class:`StdioMcpClient` go through
``__exit__`` even on test failure, so AC #6 (no orphan processes) is
held by the SDK rather than by hand-rolled signal handling.
"""

from __future__ import annotations

import asyncio
import json
import threading
import warnings
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult
from mcp.types import Tool as McpTool

from mcp import ClientSession
from pretensor.benchmark.l3.agent import AgentTool
from pretensor.errors import PretensorError

__all__ = [
    "McpClient",
    "McpClientError",
    "McpToolResult",
    "StdioMcpClient",
    "tool_result_to_text",
]


_DEFAULT_STARTUP_TIMEOUT_S = 30.0
"""How long to wait for ``pretensor serve`` to respond to ``initialize``.

The serve handshake is fast on cold-cache machines; 30s is a generous
upper bound. A longer-than-30s startup almost always means a broken
graph dir or a version mismatch — surface those as clear errors rather
than waiting forever.
"""

_DEFAULT_TEARDOWN_TIMEOUT_S = 30.0
"""How long to wait for the async session close + worker-thread join.

Distinct from :data:`_DEFAULT_STARTUP_TIMEOUT_S` so the two intents
read clearly even though the numeric values match today. ``stdio_client``
sends EOF and waits for the subprocess to exit; if the child hangs,
this cap keeps shutdown bounded.

Worst-case shutdown wall time is roughly **2x** this value:
:meth:`StdioMcpClient.__exit__` waits up to ``_DEFAULT_TEARDOWN_TIMEOUT_S``
for ``_async_close`` to finish, then ``_tear_down_loop`` joins the
worker thread for another ``_DEFAULT_TEARDOWN_TIMEOUT_S``. Both budgets
only get fully consumed when the subprocess refuses to exit AND the
loop thread itself is wedged — a real fault, not a normal slow run.
"""

_DEFAULT_OUTER_LOOP_TIMEOUT_S = 90.0
"""Outer guard on the ``__enter__`` wait for the worker loop's response.

``asyncio.wait_for`` inside the loop already enforces
:data:`_DEFAULT_STARTUP_TIMEOUT_S`, so a healthy run never approaches
this bound. The guard exists to bound the main-thread wait when the
worker loop thread itself crashes (e.g. an unhandled exception in
``_run_loop``) — without it, ``concurrent.futures.Future.result()``
would block the caller indefinitely. 90s is generous: it's long
enough to ride out the inner timeout plus its cleanup, and short
enough that an operator notices when it fires.
"""

_DEFAULT_CALL_TIMEOUT_S = 60.0
"""Per-tool-call timeout submitted to ``ClientSession.call_tool``.

Some tools (``cypher`` over a 100k-node graph, ``traverse`` with high
``k``) take real time; the L3 runner already caps the LLM budget per
question elsewhere. 60s keeps any one stuck call from blocking the
whole question without short-circuiting healthy slow tools.
"""


class McpClientError(PretensorError, RuntimeError):
    """Raised on subprocess startup, shutdown, or call-level failures."""


@dataclass(frozen=True, slots=True)
class McpToolResult:
    """One MCP tool invocation's output, normalised for the agent loop.

    ``content`` is what we hand the LLM (a JSON-encoded snapshot of the
    structured result, or the concatenated text blocks if the tool
    returned plain text). ``is_error`` flags MCP-level failures so the
    runner can record a tool-call trace entry even when the call faulted.
    """

    content: str
    is_error: bool


class McpClient(Protocol):
    """Sync MCP surface used by the L3 pretensor runner.

    Production callers use :class:`StdioMcpClient` (spawns
    ``pretensor serve`` and speaks stdio). Tests inject a fake.
    Both ``list_tools`` and ``call_tool`` are sync because the runner
    drives one tool at a time per question; concurrency would not
    produce a faster benchmark.
    """

    def list_tools(self) -> list[AgentTool]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult: ...


class StdioMcpClient:
    """Spawn-and-drive a ``pretensor serve`` MCP subprocess over stdio.

    Use as a context manager (``with StdioMcpClient(...) as client``):
    enter starts the subprocess, runs the MCP handshake, and stages the
    background event loop; exit tears down the session, signals EOF on
    stdio, and joins the worker thread. Cleanup is in a try/finally
    inside the SDK so a tool-call exception does not leak the subprocess.
    """

    def __init__(
        self,
        *,
        graph_dir: Path,
        command: str = "pretensor",
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._params = StdioServerParameters(
            command=command,
            args=["serve", "--graph-dir", str(graph_dir), "--no-print-config"]
            + list(extra_args or []),
            env=env,
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._cached_tools: list[AgentTool] | None = None

    def __enter__(self) -> StdioMcpClient:
        self._loop = asyncio.new_event_loop()

        def _run_loop() -> None:
            assert self._loop is not None
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(
            target=_run_loop, name="l3-mcp-stdio-loop", daemon=True
        )
        self._loop_thread.start()
        # Push the startup timeout INSIDE the loop via ``asyncio.wait_for``.
        # If the inner coroutine is suspended (e.g. ``stdio_client``
        # spawned the child but ``initialize`` hasn't returned yet),
        # ``wait_for`` cancels it cleanly: the task receives
        # ``CancelledError`` at its next ``await``, ``_async_open``'s
        # ``except BaseException: await stack.aclose()`` branch runs to
        # completion (which calls ``stdio_client.__aexit__`` and reaps
        # the subprocess), and only THEN is ``TimeoutError`` re-raised.
        # An outer ``concurrent.futures`` timeout would mark the
        # destination future cancelled and return immediately, leaving
        # the source task to run its cleanup after the loop had already
        # been torn down — orphaning the subprocess.
        try:
            asyncio.run_coroutine_threadsafe(
                _wait_for_open(self, _DEFAULT_STARTUP_TIMEOUT_S), self._loop
            ).result(timeout=_DEFAULT_OUTER_LOOP_TIMEOUT_S)
        except Exception as exc:
            self._tear_down_loop()
            raise McpClientError(
                f"Failed to start MCP subprocess `{self._params.command} "
                f"{' '.join(self._params.args)}`: {exc}"
            ) from exc
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._async_close(), self._loop).result(
                timeout=_DEFAULT_TEARDOWN_TIMEOUT_S
            )
        except Exception as cleanup_exc:
            # ``AsyncExitStack.aclose()`` propagates the FIRST inner
            # exception but still attempts to close every registered
            # context — so ``stdio_client.__aexit__`` will normally
            # have run and the subprocess will be reaped even when
            # this branch fires. The exception we trap here is that
            # propagated inner exception, OR a TimeoutError if the
            # whole close took longer than the teardown budget.
            #
            # We don't re-raise: an exception in the original ``with``
            # body would otherwise be silently shadowed by a cleanup
            # error. We DO surface it as a warning so the run is not
            # silent — the runner is non-interactive and a subprocess
            # leak (e.g. a child that ignored EOF) would otherwise hide
            # here. Operators see the warning on stderr and can
            # investigate before the next run.
            warnings.warn(
                f"StdioMcpClient teardown error (subprocess may be "
                f"orphaned if it did not exit on EOF): {cleanup_exc!r}",
                ResourceWarning,
                stacklevel=2,
            )
        finally:
            self._tear_down_loop()

    async def _async_open(self) -> None:
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(self._params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._session = session
        self._exit_stack = stack

    async def _async_close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._session = None

    def _tear_down_loop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=_DEFAULT_TEARDOWN_TIMEOUT_S)
        if self._loop is not None:
            self._loop.close()
        self._loop = None
        self._loop_thread = None

    def list_tools(self) -> list[AgentTool]:
        """Return the full tool catalogue advertised by ``pretensor serve``.

        The first call hits the server; subsequent calls return a
        cached list — the OSS server does not register tools after
        startup, so caching is safe and avoids repeating the round-trip
        once per question.
        """
        if self._cached_tools is not None:
            return list(self._cached_tools)
        if self._session is None or self._loop is None:
            raise McpClientError(
                "StdioMcpClient.list_tools called outside the with-block "
                "(or after an open() failure)."
            )
        result = asyncio.run_coroutine_threadsafe(
            self._session.list_tools(), self._loop
        ).result(timeout=_DEFAULT_CALL_TIMEOUT_S)
        tools = [_to_agent_tool(t) for t in result.tools]
        self._cached_tools = tools
        return list(tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        """Invoke one MCP tool synchronously; surface server errors as ``is_error``."""
        if self._session is None or self._loop is None:
            raise McpClientError(
                "StdioMcpClient.call_tool called outside the with-block "
                "(or after an open() failure)."
            )
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments=arguments), self._loop
        )
        try:
            result = future.result(timeout=_DEFAULT_CALL_TIMEOUT_S)
        except Exception as exc:
            # Build the error payload via ``json.dumps`` so an exception
            # message containing quotes, backslashes, or newlines escapes
            # cleanly — a raw f-string would emit invalid JSON the LLM
            # tool-result decoder then misparses.
            return McpToolResult(
                content=json.dumps({"error": f"MCP transport failure: {exc}"}),
                is_error=True,
            )
        return McpToolResult(
            content=tool_result_to_text(result),
            is_error=bool(result.isError),
        )


async def _wait_for_open(client: StdioMcpClient, timeout_s: float) -> None:
    """Run ``client._async_open`` under an in-loop ``asyncio.wait_for``.

    Module-level (not a method) so the patching pattern used by tests
    — ``monkeypatch.setattr(StdioMcpClient, "_async_open", fake)`` — is
    picked up via the bound method lookup at call time. Lives next to
    the class because it is intimate with its lifecycle.
    """
    await asyncio.wait_for(client._async_open(), timeout=timeout_s)


def _to_agent_tool(tool: McpTool) -> AgentTool:
    """Down-cast :class:`mcp.types.Tool` into the L3-local representation.

    MCP's ``Tool.description`` is optional; substituting an empty string
    keeps the LLM payload well-formed without hiding a missing description
    behind a placeholder that might get echoed back.
    """
    return AgentTool(
        name=tool.name,
        description=tool.description or "",
        input_schema=dict(tool.inputSchema or {}),
    )


def tool_result_to_text(result: CallToolResult) -> str:
    """Flatten an MCP :class:`CallToolResult` into a single string for the LLM.

    Preference order:
    1. ``structuredContent`` — JSON-encode it. The pretensor server
       returns dict payloads through this channel; JSON is the cheapest
       lossless rendering for the LLM.
    2. Concatenated ``text`` content blocks. Other content types
       (image, audio, resource link) are skipped — the OSS server does
       not produce them today.

    A result with neither structured content nor text blocks is rendered
    as an empty JSON object so the LLM still sees a well-formed tool
    response.
    """
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent, sort_keys=True, default=str)
    parts: list[str] = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    if parts:
        return "".join(parts)
    return "{}"
