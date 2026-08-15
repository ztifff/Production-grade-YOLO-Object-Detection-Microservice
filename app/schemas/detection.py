"""
app/schemas/detection.py
────────────────────────
Pydantic models for detection request/response contracts.

These schemas serve as the single source of truth for:
  - FastAPI request validation
  - OpenAPI schema generation
  - Serialised JSON responses
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Bounding Box ──────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Absolute pixel coordinates of the detected object bounding box."""

    x_min: float = Field(..., description="Left edge of the box (pixels).")
    y_min: float = Field(..., description="Top edge of the box (pixels).")
    x_max: float = Field(..., description="Right edge of the box (pixels).")
    y_max: float = Field(..., description="Bottom edge of the box (pixels).")


# ── Single Detection ──────────────────────────────────────────────────────────

class DetectionResult(BaseModel):
    """A single detected object returned by the inference engine."""

    label: str = Field(..., description="Human-readable class label (e.g. 'person').")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score rounded to 4 decimal places.",
    )
    box: BoundingBox = Field(..., description="Absolute bounding-box coordinates.")
    class_id: int = Field(..., description="Numeric class index from the model vocabulary.")

    model_config = {"json_schema_extra": {"example": {
        "label": "person",
        "confidence": 0.9342,
        "box": {"x_min": 120.5, "y_min": 80.2, "x_max": 340.1, "y_max": 510.7},
        "class_id": 0,
    }}}


# ── Inference Response ────────────────────────────────────────────────────────

class DetectionResponse(BaseModel):
    """Full response envelope returned by POST /api/v1/vision/detect."""

    success: bool = Field(default=True)
    model: str = Field(..., description="Model file name used for inference.")
    image_width: int = Field(..., description="Width of the submitted image in pixels.")
    image_height: int = Field(..., description="Height of the submitted image in pixels.")
    execution_time_ms: float = Field(
        ...,
        description="Total server-side inference duration in milliseconds.",
    )
    object_count: int = Field(..., description="Total number of detected objects.")
    detections: List[DetectionResult] = Field(
        default_factory=list,
        description="List of detected objects; empty if none found.",
    )

    model_config = {"json_schema_extra": {"example": {
        "success": True,
        "model": "yolov8n.pt",
        "image_width": 640,
        "image_height": 480,
        "execution_time_ms": 38.42,
        "object_count": 2,
        "detections": [
            {
                "label": "person",
                "confidence": 0.9342,
                "box": {"x_min": 120.5, "y_min": 80.2, "x_max": 340.1, "y_max": 510.7},
                "class_id": 0,
            }
        ],
    }}}


# ── Request — Base64 body variant ────────────────────────────────────────────

class Base64DetectRequest(BaseModel):
    """Request body for the Base64-encoded image variant of the detect endpoint."""

    image_base64: str = Field(
        ...,
        min_length=4,
        description=(
            "Standard Base64-encoded image data. "
            "Optionally prefixed with a data URI, e.g. 'data:image/jpeg;base64,...'."
        ),
    )
    conf: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=1.0,
        description="Override confidence threshold for this request.",
    )
    iou: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=1.0,
        description="Override IoU threshold for this request.",
    )

    @field_validator("image_base64", mode="before")
    @classmethod
    def _strip_data_uri_prefix(cls, v: str) -> str:
        """Strip the 'data:<mime>;base64,' prefix if present."""
        if isinstance(v, str) and v.startswith("data:"):
            _, _, encoded = v.partition(",")
            return encoded
        return v


# ── Error response ────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """RFC 7807 Problem JSON compatible error envelope."""

    success: bool = False
    error: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")
    request_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for distributed tracing.",
    )
