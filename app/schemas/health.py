"""
app/schemas/health.py
─────────────────────
Pydantic models for /healthz and /metrics response payloads.
"""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Kubernetes liveness / readiness probe response."""

    status: str = Field(..., description="'healthy' or 'degraded'.")
    model_loaded: bool = Field(..., description="True once the YOLO Singleton is ready.")
    model_path: Optional[str] = Field(default=None, description="Active model file name.")
    version: str = Field(..., description="Application version string.")
    uptime_seconds: float = Field(..., description="Seconds since application startup.")

    model_config = {"json_schema_extra": {"example": {
        "status": "healthy",
        "model_loaded": True,
        "model_path": "yolov8n.pt",
        "version": "1.0.0",
        "uptime_seconds": 142.7,
    }}}


class MetricsResponse(BaseModel):
    """Basic operational metrics snapshot."""

    total_requests: int = Field(default=0, description="Total inference requests received.")
    successful_requests: int = Field(default=0, description="Requests that returned 2xx.")
    failed_requests: int = Field(default=0, description="Requests that returned 4xx/5xx.")
    rate_limited_requests: int = Field(default=0, description="Requests rejected by rate limiter.")
    avg_latency_ms: float = Field(default=0.0, description="Rolling average inference latency.")
    error_rate: float = Field(
        default=0.0,
        description="failed_requests / total_requests (0.0–1.0).",
    )
    active_api_keys: int = Field(default=0, description="Number of distinct API keys seen.")
    tier_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Request counts split by tier (free / premium / enterprise).",
    )
