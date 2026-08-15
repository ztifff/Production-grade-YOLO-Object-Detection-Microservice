"""
app/api/metrics.py
──────────────────
GET /metrics — operational metrics snapshot.

Counters are stored as module-level atomics protected by a Lock so they can
be safely incremented from multiple Uvicorn worker threads.

In production, replace or supplement this with Prometheus /metrics via the
``prometheus-fastapi-instrumentator`` library for richer time-series data.
"""

from __future__ import annotations

import threading
from typing import Dict

from fastapi import APIRouter

from app.schemas.health import MetricsResponse

router = APIRouter()

# ── In-process counters ───────────────────────────────────────────────────────
_lock = threading.Lock()
_counters: Dict[str, float] = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "rate_limited_requests": 0,
    "total_latency_ms": 0.0,
}
_tier_breakdown: Dict[str, int] = {}
_active_keys: set = set()


def record_request(
    *,
    success: bool,
    rate_limited: bool = False,
    latency_ms: float = 0.0,
    tier: str = "unknown",
    key_prefix: str = "",
) -> None:
    """
    Increment counters for a completed request.

    Call this from the vision router after every inference attempt.
    """
    with _lock:
        _counters["total_requests"] += 1
        if rate_limited:
            _counters["rate_limited_requests"] += 1
        elif success:
            _counters["successful_requests"] += 1
            _counters["total_latency_ms"] += latency_ms
        else:
            _counters["failed_requests"] += 1

        _tier_breakdown[tier] = _tier_breakdown.get(tier, 0) + 1
        if key_prefix:
            _active_keys.add(key_prefix)


def reset_counters() -> None:
    """Reset all counters. Used in tests only."""
    with _lock:
        for k in _counters:
            _counters[k] = 0.0
        _tier_breakdown.clear()
        _active_keys.clear()


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Operations"],
    summary="Operational metrics",
    description=(
        "Returns a snapshot of operational counters since the last service restart. "
        "For time-series metrics, integrate Prometheus via "
        "``prometheus-fastapi-instrumentator``."
    ),
)
async def metrics() -> MetricsResponse:
    """Return a snapshot of operational metrics."""
    with _lock:
        total = int(_counters["total_requests"])
        success = int(_counters["successful_requests"])
        failed = int(_counters["failed_requests"])
        rate_limited = int(_counters["rate_limited_requests"])
        total_lat = _counters["total_latency_ms"]
        tier_copy = dict(_tier_breakdown)
        key_count = len(_active_keys)

    avg_lat = (total_lat / success) if success > 0 else 0.0
    error_rate = (failed / total) if total > 0 else 0.0

    return MetricsResponse(
        total_requests=total,
        successful_requests=success,
        failed_requests=failed,
        rate_limited_requests=rate_limited,
        avg_latency_ms=round(avg_lat, 2),
        error_rate=round(error_rate, 4),
        active_api_keys=key_count,
        tier_breakdown=tier_copy,
    )
