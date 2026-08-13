"""End-to-end stdio smoke test for ``pretensor serve``.

Launches the real CLI in a subprocess and performs an MCP initialize
handshake plus a ``tools/list`` call over stdio — the exact path a fresh
``pip install pretensor`` user exercises when wiring the server into
Claude/Cursor. The unit suite constructs servers in-process, so a
dependency drift that breaks server *startup* (e.g. an incompatible
``mcp`` release changing its public API) is invisible to it; this test
fails loudly instead.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from pretensor.connectors.models import Column, SchemaSnapshot, Table
from pretensor.core.builder import GraphBuilder
from pretensor.core.registry import GraphRegistry
from pretensor.core.store import KuzuStore

_HANDSHAKE_TIMEOUT_S = 30.0


def _build_state_dir(tmp_path: Path) -> Path:
    users = Table(
        name="users",
        schema_name="public",
        columns=[Column(name="id", data_type="int", is_primary_key=True)],
    )
    snap = SchemaSnapshot(
        connection_name="demo",
        database="demo",
        schemas=["public"],
        tables=[users],
        introspected_at=datetime.now(timezone.utc),
    )
    graph = tmp_path / "graphs" / "demo.kuzu"
    graph.parent.mkdir(parents=True, exist_ok=True)
    store = KuzuStore(graph)
    try:
        GraphBuilder().build(snap, store, run_relationship_discovery=False)
    finally:
        store.close()
    reg = GraphRegistry(tmp_path / "registry.json").load()
    reg.upsert(
        connection_name="demo",
        database="demo",
        dsn="postgresql://localhost/demo",
        graph_path=graph,
        indexed_at=datetime.now(timezone.utc),
    )
    reg.save()
    return tmp_path


def test_serve_stdio_initialize_and_list_tools(tmp_path: Path) -> None:
    state = _build_state_dir(tmp_path)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from pretensor.cli.main import app; app()",
            "serve",
            "--state-dir",
            str(state),
            "--no-print-config",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=_pump, daemon=True).start()

    def send(msg: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def recv(want_id: int) -> dict:
        while True:
            line = lines.get(timeout=_HANDSHAKE_TIMEOUT_S)
            if line is None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise AssertionError(
                    f"server exited before responding (rc={proc.poll()}):\n{stderr}"
                )
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-protocol noise; must not crash the client
            if msg.get("id") == want_id:
                return msg

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke-test", "version": "0"},
                },
            }
        )
        init = recv(1)
        assert "result" in init, f"initialize failed: {init}"
        assert "serverInfo" in init["result"]

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = {t["name"] for t in recv(2)["result"]["tools"]}
        assert "consumers" in tools
        assert "impact" in tools
    finally:
        proc.terminate()
        proc.wait(timeout=10)
