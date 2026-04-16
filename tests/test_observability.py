"""Tests for process-level logging and timing helpers."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from pretensor.observability import (
    configure_logging,
    log_timed_operation,
    run_timed_async,
)


def _reset_root_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_configure_logging_json_to_file(tmp_path: Path) -> None:
    """JSON formatter writes machine-readable records to log file."""
    _reset_root_handlers()
    log_path = tmp_path / "logs" / "pretensor.log"
    configure_logging(level="info", log_format="json", log_file=log_path)

    logger = logging.getLogger("pretensor.test")
    logger.info(
        "hello world",
        extra={"event": "test.event", "status": "ok", "request_id": "abc"},
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload["message"] == "hello world"
    assert payload["event"] == "test.event"
    assert payload["status"] == "ok"
    assert payload["request_id"] == "abc"
    assert payload["level"] == "info"
    _reset_root_handlers()


def test_log_timed_operation_emits_duration(caplog: object) -> None:
    """Timed helper emits event/status/duration fields."""
    configure_logging(level="info", log_format="text")
    logger = logging.getLogger("pretensor.timing")

    from _pytest.logging import LogCaptureFixture

    cap = caplog if isinstance(caplog, LogCaptureFixture) else None
    assert cap is not None
    with cap.at_level(logging.INFO):
        with log_timed_operation(logger, event="unit.operation", answer=42):
            pass

    matched = [r for r in cap.records if getattr(r, "event", "") == "unit.operation"]
    assert matched
    rec = matched[-1]
    assert getattr(rec, "status", None) == "ok"
    duration = getattr(rec, "duration_ms", None)
    assert isinstance(duration, (int, float))
    assert duration >= 0
    assert getattr(rec, "answer", None) == 42


def test_run_timed_async_returns_value(caplog: object) -> None:
    """Async timed helper returns callback value and logs timing fields."""
    configure_logging(level="info", log_format="text")
    logger = logging.getLogger("pretensor.async")

    async def _callback() -> int:
        return 7

    from _pytest.logging import LogCaptureFixture

    cap = caplog if isinstance(caplog, LogCaptureFixture) else None
    assert cap is not None
    with cap.at_level(logging.INFO):
        value = asyncio.run(
            run_timed_async(
                logger,
                event="unit.async_operation",
                callback=_callback,
                worker="test",
            )
        )
    assert value == 7
    matched = [r for r in cap.records if getattr(r, "event", "") == "unit.async_operation"]
    assert matched
    rec = matched[-1]
    assert getattr(rec, "status", None) == "ok"
    assert getattr(rec, "worker", None) == "test"
