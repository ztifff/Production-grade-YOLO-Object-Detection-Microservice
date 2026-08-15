"""
app/core/logging.py
───────────────────
Configures a structured, JSON-formatted logger for the entire application.

All modules should obtain their logger via:

    from app.core.logging import get_logger
    logger = get_logger(__name__)

Output format is NDJSON, compatible with:
  - ELK Stack (Logstash / OpenSearch)
  - Datadog log ingestion
  - Google Cloud Logging
  - AWS CloudWatch Logs Insights
"""

from __future__ import annotations

import logging
import logging.config
import sys
from typing import Any, Dict

from app.core.config import settings


# ── JSON log formatter ────────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """
    Renders each log record as a single-line JSON object.  We avoid a heavy
    third-party dependency (structlog) to keep the image lean while still
    producing machine-readable output.
    """

    import json as _json  # noqa: PLC0415 — intentional class-level import

    # Common fields attached to every record.
    STATIC_FIELDS: Dict[str, str] = {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "env": settings.ENV,
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        import json  # re-import inside method for Py < 3.12 compat

        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **self.STATIC_FIELDS,
        }

        # Attach extra fields injected by callers via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in _STDLIB_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


# Attributes on every stdlib LogRecord — we skip these to avoid noise.
_STDLIB_LOG_ATTRS = frozenset(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__.keys()
    | {"message", "asctime"}
)


# ── Public setup function ─────────────────────────────────────────────────────

def configure_logging() -> None:
    """
    Apply JSON-structured logging to the root logger.

    Call this once from ``app/main.py`` before any other module emits logs.
    Subsequent calls are idempotent.
    """
    numeric_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())

    root = logging.getLogger()
    if root.handlers:
        # Already configured — skip (handles hot-reload in development)
        return

    root.setLevel(numeric_level)
    root.addHandler(handler)

    # Silence noisy third-party loggers at a higher threshold
    for noisy_lib in ("uvicorn.access", "ultralytics", "torch"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-specific logger.

    Usage::

        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Inference complete", extra={"latency_ms": 42.1})
    """
    return logging.getLogger(name)
