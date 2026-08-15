"""
app/middleware/usage_logger.py
──────────────────────────────
Structured usage metering for billing and observability.

On every successful inference call the route handler calls
:func:`log_inference_event` which emits a single structured JSON line
containing all billing-relevant fields.

The output is intentionally log-drain-friendly (NDJSON) so that it can be
shipped directly into:
  - **ELK Stack**: via Filebeat / Logstash
  - **Datadog**: via the log agent
  - **AWS CloudWatch Logs Insights**: query by ``event = "inference_complete"``
  - **BigQuery**: streamed via a sidecar log forwarder

No database writes happen in this module — that concern is left to the
downstream log aggregator or a separate billing microservice.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from app.core.logging import get_logger
from app.middleware.auth import TenantContext

logger = get_logger(__name__)

# ── Application start time (used for uptime calculations) ────────────────────
_START_TS: float = time.time()


def _hash_key(raw_key_prefix: str) -> str:
    """
    Return a short SHA-256 digest of the API key prefix.

    We deliberately hash an *already-truncated* prefix rather than the full
    key to provide a consistent log correlation handle without exposing
    sensitive credential material.
    """
    return "sha256:" + hashlib.sha256(raw_key_prefix.encode()).hexdigest()[:16]


def log_inference_event(
    tenant: TenantContext,
    processing_ms: float,
    object_count: int,
    image_size_bytes: int,
    image_width: int,
    image_height: int,
    model_name: str,
    request_id: Optional[str] = None,
) -> None:
    """
    Emit a structured billing/usage event for a completed inference request.

    Parameters
    ----------
    tenant:
        The authenticated :class:`~app.middleware.auth.TenantContext`.
    processing_ms:
        Total server-side inference duration in milliseconds.
    object_count:
        Number of objects detected in the image.
    image_size_bytes:
        Raw size of the uploaded image payload in bytes.
    image_width, image_height:
        Pixel dimensions of the submitted image.
    model_name:
        Filename of the YOLO model used (e.g. ``"yolov8n.pt"``).
    request_id:
        Optional correlation ID injected by an upstream proxy or gateway.

    Log payload fields
    ------------------
    All fields are at the top level of the JSON log line so that log
    aggregators can index them without further parsing.

    =================== ============================================
    Field               Description
    =================== ============================================
    event               Always ``"inference_complete"``
    client_id           Tenant identifier from API key map
    api_key_hash        Short SHA-256 of key prefix (safe to log)
    tier                Billing tier (free / premium / enterprise)
    processing_ms       Inference wall-clock time
    objects_detected    Count of returned detections
    image_size_bytes    Payload size (useful for bandwidth billing)
    image_width/height  Image resolution
    model               Active model filename
    request_id          Correlation / trace ID (if available)
    timestamp_epoch     Unix epoch float for time-series aggregation
    =================== ============================================
    """
    logger.info(
        "inference_complete",
        extra={
            "event": "inference_complete",
            "client_id": tenant.client_id,
            "api_key_hash": _hash_key(tenant.api_key_prefix),
            "tier": tenant.tier,
            "processing_ms": round(processing_ms, 3),
            "objects_detected": object_count,
            "image_size_bytes": image_size_bytes,
            "image_width": image_width,
            "image_height": image_height,
            "model": model_name,
            "request_id": request_id,
            "timestamp_epoch": round(time.time(), 3),
        },
    )


def log_error_event(
    tenant: Optional[TenantContext],
    error_code: str,
    message: str,
    request_id: Optional[str] = None,
) -> None:
    """
    Emit a structured error event for failed inference requests.

    Used by the global exception handler in ``app/main.py``.
    """
    logger.error(
        "inference_error",
        extra={
            "event": "inference_error",
            "client_id": tenant.client_id if tenant else "unauthenticated",
            "tier": tenant.tier if tenant else "unknown",
            "error_code": error_code,
            "message": message,
            "request_id": request_id,
            "timestamp_epoch": round(time.time(), 3),
        },
    )
