"""
app/api/v1/vision.py
─────────────────────
POST /api/v1/vision/detect — main inference endpoint.

Supports two input modes:
  1. **Multipart file upload** (``multipart/form-data``) — recommended for
     direct image uploads from clients.
  2. **JSON body with Base64-encoded image** — convenient for API integrations
     that prefer a single JSON envelope.

Concurrency model
──────────────────
YOLO inference is CPU/GPU-bound and would block the asyncio event loop if
called directly from an ``async def`` handler.  We offload it to a shared
:class:`~concurrent.futures.ThreadPoolExecutor` via
``asyncio.get_event_loop().run_in_executor`` so that other coroutines
(health checks, metrics, etc.) continue to run concurrently.
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Security, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.inference.engine import InferenceError, InvalidImageError, ModelLoadError, YOLOEngine
from app.middleware.auth import TenantContext, require_api_key
from app.middleware.rate_limiter import enforce_rate_limit
from app.middleware.usage_logger import log_error_event, log_inference_event
from app.schemas.detection import Base64DetectRequest, DetectionResponse, ErrorDetail

logger = get_logger(__name__)

router = APIRouter()

# ── Shared thread pool for CPU/GPU-bound inference ────────────────────────────
# Max workers = number of CPU cores; keeps GPU serialised without starving CPU.
_thread_pool = ThreadPoolExecutor(max_workers=settings.WORKERS, thread_name_prefix="yolo-infer")


# ── Helper: run blocking inference in the thread pool ─────────────────────────

async def _run_inference(
    payload: bytes,
    conf: Optional[float],
    iou: Optional[float],
) -> dict:
    """Off-load ``YOLOEngine.detect`` to the thread pool."""
    engine = YOLOEngine.get_instance()
    loop = asyncio.get_event_loop()
    fn = partial(engine.detect, payload, conf, iou)
    return await loop.run_in_executor(_thread_pool, fn)


# ── Endpoint 1: Multipart file upload ────────────────────────────────────────

@router.post(
    "/detect",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect objects in an image (file upload)",
    description=(
        "Submit an image as **multipart/form-data** to run YOLO object detection. "
        "Returns a list of detections with labels, confidence scores, and bounding boxes."
    ),
    responses={
        401: {"model": ErrorDetail, "description": "Missing API key."},
        403: {"model": ErrorDetail, "description": "Invalid API key."},
        413: {"model": ErrorDetail, "description": "Image too large."},
        422: {"model": ErrorDetail, "description": "Unsupported image format."},
        429: {"model": ErrorDetail, "description": "Rate limit exceeded."},
        503: {"model": ErrorDetail, "description": "Model not ready."},
    },
    tags=["Vision"],
)
async def detect_upload(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, WEBP — max 10 MB)."),
    conf: Optional[float] = Form(default=None, ge=0.01, le=1.0, description="Confidence threshold override."),
    iou: Optional[float] = Form(default=None, ge=0.01, le=1.0, description="IoU threshold override."),
    tenant: TenantContext = Security(require_api_key),
) -> DetectionResponse:
    await enforce_rate_limit(tenant)

    request_id = str(uuid.uuid4())

    # ── Size guard ────────────────────────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "payload_too_large",
                "message": (
                    f"Image exceeds the {settings.MAX_IMAGE_BYTES // (1024 * 1024)} MB "
                    "size limit."
                ),
            },
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        result = await _run_inference(image_bytes, conf, iou)
    except (InvalidImageError, ValueError) as exc:
        log_error_event(tenant, "invalid_image", str(exc), request_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "invalid_image", "message": str(exc)},
        ) from exc
    except ModelLoadError as exc:
        log_error_event(tenant, "model_not_ready", str(exc), request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_not_ready", "message": str(exc)},
        ) from exc
    except InferenceError as exc:
        log_error_event(tenant, "inference_error", str(exc), request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "inference_error", "message": str(exc)},
        ) from exc

    # ── Usage metering ────────────────────────────────────────────────────────
    log_inference_event(
        tenant=tenant,
        processing_ms=result["execution_time_ms"],
        object_count=result["object_count"],
        image_size_bytes=len(image_bytes),
        image_width=result["image_width"],
        image_height=result["image_height"],
        model_name=result["model"],
        request_id=request_id,
    )

    return DetectionResponse(
        model=result["model"],
        image_width=result["image_width"],
        image_height=result["image_height"],
        execution_time_ms=result["execution_time_ms"],
        object_count=result["object_count"],
        detections=result["detections"],
    )


# ── Endpoint 2: Base64 JSON body ──────────────────────────────────────────────

@router.post(
    "/detect/base64",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect objects in an image (Base64 JSON body)",
    description=(
        "Submit a **Base64-encoded** image in a JSON body. "
        "Optionally prefix with a data URI, e.g. ``data:image/jpeg;base64,...``."
    ),
    responses={
        401: {"model": ErrorDetail, "description": "Missing API key."},
        403: {"model": ErrorDetail, "description": "Invalid API key."},
        413: {"model": ErrorDetail, "description": "Image too large."},
        422: {"model": ErrorDetail, "description": "Invalid Base64 or image format."},
        429: {"model": ErrorDetail, "description": "Rate limit exceeded."},
        503: {"model": ErrorDetail, "description": "Model not ready."},
    },
    tags=["Vision"],
)
async def detect_base64(
    body: Base64DetectRequest,
    request: Request,
    tenant: TenantContext = Security(require_api_key),
) -> DetectionResponse:
    await enforce_rate_limit(tenant)

    request_id = str(uuid.uuid4())

    # ── Decode Base64 ─────────────────────────────────────────────────────────
    try:
        image_bytes = base64.b64decode(body.image_base64, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "invalid_base64", "message": "Could not decode Base64 string."},
        ) from exc

    if len(image_bytes) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "payload_too_large",
                "message": (
                    f"Decoded image exceeds the "
                    f"{settings.MAX_IMAGE_BYTES // (1024 * 1024)} MB size limit."
                ),
            },
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        result = await _run_inference(image_bytes, body.conf, body.iou)
    except (InvalidImageError, ValueError) as exc:
        log_error_event(tenant, "invalid_image", str(exc), request_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "invalid_image", "message": str(exc)},
        ) from exc
    except ModelLoadError as exc:
        log_error_event(tenant, "model_not_ready", str(exc), request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_not_ready", "message": str(exc)},
        ) from exc
    except InferenceError as exc:
        log_error_event(tenant, "inference_error", str(exc), request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "inference_error", "message": str(exc)},
        ) from exc

    log_inference_event(
        tenant=tenant,
        processing_ms=result["execution_time_ms"],
        object_count=result["object_count"],
        image_size_bytes=len(image_bytes),
        image_width=result["image_width"],
        image_height=result["image_height"],
        model_name=result["model"],
        request_id=request_id,
    )

    return DetectionResponse(
        model=result["model"],
        image_width=result["image_width"],
        image_height=result["image_height"],
        execution_time_ms=result["execution_time_ms"],
        object_count=result["object_count"],
        detections=result["detections"],
    )
