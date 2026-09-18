"""Structured logging using structlog.

Logs are emitted as JSON when `LOG_JSON=true`. Sensitive fields are redacted
when `REDACT_LOGS=true`. The logger is safe to import from any module —
configuring twice is a no-op.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import Settings, get_settings

_REDACT_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "token",
    "secret",
    "password",
    "llm_api_key",
    "embedding_api_key",
    "cookie",
}


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root + structlog logging exactly once."""
    settings = settings or get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Reset handlers so configure is idempotent across test runs.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    root.addHandler(handler)
    root.setLevel(level)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redactor,
    ]
    if settings.log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def _redactor(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    if not settings.redact_logs:
        return event_dict
    return {k: ("***REDACTED***" if k.lower() in _REDACT_KEYS and v else v) for k, v in event_dict.items()}


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to `name`."""
    try:
        log: structlog.stdlib.BoundLogger = structlog.get_logger(name or "app")
        return log
    except Exception:
        # Fallback during import-time edge cases.
        fallback: structlog.stdlib.BoundLogger = structlog.get_logger("app")
        return fallback


__all__ = ["configure_logging", "get_logger"]
