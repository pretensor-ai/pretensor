"""Process-wide logging configuration and timing helpers."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Literal, TypeVar

__all__ = [
    "LogFormat",
    "LogLevel",
    "configure_logging",
    "log_timed_operation",
    "run_timed_async",
]

LogLevel = Literal["debug", "info", "warning", "error"]
LogFormat = Literal["text", "json"]

_LOG_LEVELS: dict[LogLevel, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
_PRETENSOR_HANDLER_ATTR = "_pretensor_logging_handler"
_T = TypeVar("_T")


class _JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": getattr(record, "event", None),
            "status": getattr(record, "status", None),
            "message": record.getMessage(),
        }
        if hasattr(record, "duration_ms"):
            payload["duration_ms"] = getattr(record, "duration_ms")
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in payload:
                continue
            if key in ("args", "msg", "exc_info", "exc_text", "stack_info"):
                continue
            if key in (
                "name",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "taskName",
                "asctime",
            ):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return _json_dumps(payload)


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    *,
    level: LogLevel = "info",
    log_format: LogFormat = "text",
    log_file: Path | None = None,
) -> None:
    """Configure root logging handlers for CLI and MCP code paths."""
    root = logging.getLogger()
    target_level = _LOG_LEVELS[level]
    root.setLevel(target_level)
    logging.getLogger("pretensor").setLevel(logging.NOTSET)
    _remove_pretensor_handlers(root)

    formatter: logging.Formatter
    if log_format == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_build_handler_filter(target_level))
    setattr(stream_handler, _PRETENSOR_HANDLER_ATTR, True)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_build_handler_filter(target_level))
        setattr(file_handler, _PRETENSOR_HANDLER_ATTR, True)
        root.addHandler(file_handler)


def _remove_pretensor_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        if not getattr(handler, _PRETENSOR_HANDLER_ATTR, False):
            continue
        root.removeHandler(handler)
        handler.close()


def _build_handler_filter(min_level: int) -> Callable[[logging.LogRecord], bool]:
    def _handler_filter(record: logging.LogRecord) -> bool:
        if record.name.startswith("pretensor"):
            if record.levelno < min_level:
                return False
            if (
                min_level == logging.INFO
                and record.levelno == logging.INFO
                and not hasattr(record, "event")
            ):
                return False
            return True
        return record.levelno >= logging.WARNING

    return _handler_filter


@contextmanager
def log_timed_operation(
    logger: logging.Logger,
    *,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> Iterator[None]:
    """Log operation latency and status with structured fields."""
    started = time.perf_counter()
    try:
        yield
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.log(
            level,
            "%s failed in %.2fms",
            event,
            duration_ms,
            extra={
                "event": event,
                "status": "error",
                "duration_ms": duration_ms,
                **fields,
            },
            exc_info=True,
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    logger.log(
        level,
        "%s completed in %.2fms",
        event,
        duration_ms,
        extra={
            "event": event,
            "status": "ok",
            "duration_ms": duration_ms,
            **fields,
        },
    )


async def run_timed_async(
    logger: logging.Logger,
    *,
    event: str,
    callback: Callable[[], Awaitable[_T]],
    level: int = logging.INFO,
    **fields: Any,
) -> _T:
    """Await ``callback`` and emit a structured timing log."""
    with log_timed_operation(logger, event=event, level=level, **fields):
        return await callback()
