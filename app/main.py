"""
app/main.py
───────────
FastAPI application factory with lifespan management.

Startup sequence
────────────────
1. ``configure_logging()`` — JSON logger attached to root logger.
2. ``YOLOEngine.get_instance()`` — model loaded into memory/GPU.
3. Routers mounted and middleware registered.

Shutdown sequence
─────────────────
1. Thread pool flushed (in-flight inference requests complete gracefully).
2. Log final shutdown message.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, metrics
from app.api.v1 import router as v1_router
from app.api.v1.vision import _thread_pool
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.inference.engine import ModelLoadError, YOLOEngine

# Configure logging before anything else emits a log line
configure_logging()
logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifecycle.

    Everything before ``yield`` runs at startup; everything after runs at
    shutdown.
    """
    start = time.time()
    logger.info("Service starting", extra={"version": settings.APP_VERSION, "env": settings.ENV})

    # ── Startup ───────────────────────────────────────────────────────────────
    try:
        YOLOEngine.get_instance()
        elapsed = round((time.time() - start) * 1000, 1)
        logger.info(
            "Service ready",
            extra={
                "model": settings.MODEL_PATH,
                "device": settings.YOLO_DEVICE,
                "startup_ms": elapsed,
            },
        )
    except ModelLoadError as exc:
        # Log the error but don't abort startup — /healthz will return 503
        # until the model is available, allowing ops teams to diagnose.
        logger.error(
            "Model failed to load during startup",
            extra={"error": str(exc)},
        )

    yield  # ← Application is running

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Service shutting down — draining thread pool …")
    _thread_pool.shutdown(wait=True, cancel_futures=False)
    logger.info("Thread pool drained. Goodbye.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-grade YOLO Object Detection Microservice.\n\n"
            "Supports multipart image uploads and Base64 JSON payloads. "
            "Secured with API key authentication and sliding-window rate limiting."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID middleware ─────────────────────────────────────────────────
    @app.middleware("http")
    async def _attach_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(v1_router.router)
    app.include_router(health.router)
    app.include_router(metrics.router)

    # ── Global exception handlers ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled exception",
            extra={"path": str(request.url), "method": request.method},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": request.headers.get("X-Request-ID"),
            },
        )

    return app


# ── WSGI/ASGI entry point ─────────────────────────────────────────────────────
app = create_app()
