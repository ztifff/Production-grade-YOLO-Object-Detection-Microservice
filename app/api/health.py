"""
app/api/health.py
─────────────────
GET /healthz — Kubernetes liveness & readiness probe.

Returns HTTP 200 when the service is healthy and ready to accept traffic,
and HTTP 503 when the model has not yet been loaded (e.g. during startup
or after a failed model load).
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.inference.engine import YOLOEngine
from app.schemas.health import HealthResponse

router = APIRouter()

# Captured once at module import — reflects container start time
_SERVICE_START: float = time.time()


@router.get(
    "/healthz",
    response_model=HealthResponse,
    tags=["Operations"],
    summary="Liveness & readiness probe",
    description=(
        "Kubernetes-compatible health check endpoint. "
        "Returns **200** when the YOLO model is loaded and ready. "
        "Returns **503** during startup or if model initialisation failed."
    ),
)
async def healthz() -> JSONResponse:
    """Return service health status including model readiness."""
    engine_instance = YOLOEngine._instance  # noqa: SLF001 — intentional singleton peek

    model_loaded = engine_instance is not None and engine_instance.is_ready
    model_path = engine_instance.model_name if model_loaded else None
    uptime = round(time.time() - _SERVICE_START, 2)

    response_body = HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=model_path,
        version=settings.APP_VERSION,
        uptime_seconds=uptime,
    )

    status_code = 200 if model_loaded else 503
    return JSONResponse(content=response_body.model_dump(), status_code=status_code)
