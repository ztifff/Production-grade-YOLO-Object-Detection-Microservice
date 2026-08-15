"""
app/core/config.py
──────────────────
Centralised application settings loaded from environment variables or a
.env file.  All modules import `settings` from here so that configuration
is validated exactly once at startup.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic-Settings model.  Values are resolved in this priority order:
      1. Actual environment variables
      2. Variables in the .env file (if present)
      3. Field defaults below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "YOLO Vision API"
    APP_VERSION: str = "1.0.0"
    ENV: Literal["development", "staging", "production"] = "production"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Inference Engine ─────────────────────────────────────────────────────
    MODEL_PATH: str = Field(
        default="models/yolov8n.pt",
        description="Relative or absolute path to the YOLO .pt or .onnx weight file.",
    )
    YOLO_CONF: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Minimum confidence threshold for detections.",
    )
    YOLO_IOU: float = Field(
        default=0.45,
        ge=0.01,
        le=1.0,
        description="IoU threshold used during Non-Maximum Suppression.",
    )
    YOLO_DEVICE: str = Field(
        default="cpu",
        description="Inference device: 'cpu', 'cuda', 'cuda:0', 'mps', etc.",
    )
    YOLO_MAX_DET: int = Field(
        default=300,
        ge=1,
        description="Maximum number of detections returned per image.",
    )

    # ── API Gateway ───────────────────────────────────────────────────────────
    MAX_IMAGE_BYTES: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum allowed raw image payload size in bytes.",
    )
    CORS_ORIGINS: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins.",
    )
    WORKERS: int = Field(
        default=4,
        ge=1,
        description="Number of Gunicorn/Uvicorn worker processes.",
    )
    PORT: int = Field(default=8000, ge=1, le=65535)

    # ── Security & Monetisation ───────────────────────────────────────────────
    #
    # API_KEYS is a JSON-encoded dict mapping raw API key strings to tenant
    # metadata dicts, e.g.:
    #   {"sk-free-abc123": {"client_id": "demo", "tier": "free"},
    #    "sk-prem-xyz789": {"client_id": "acme", "tier": "premium"}}
    #
    # In production inject this as a single env-var string (see .env.example).
    API_KEYS_JSON: str = Field(
        default='{"sk-dev-00000000": {"client_id": "dev", "tier": "free"}}',
        alias="API_KEYS",
        description="JSON string mapping API keys to tenant metadata.",
    )

    @field_validator("API_KEYS_JSON", mode="before")
    @classmethod
    def _validate_api_keys(cls, v: Any) -> str:
        """Ensure the value can be parsed as JSON and has the expected shape."""
        if isinstance(v, dict):
            # Allow callers to pass a real dict (e.g. in tests)
            return json.dumps(v)
        try:
            parsed: Dict[str, Any] = json.loads(v)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"API_KEYS must be a valid JSON string: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("API_KEYS JSON must be a top-level object.")
        return v

    @property
    def api_keys(self) -> Dict[str, Dict[str, str]]:
        """Parsed API key map — use this everywhere in the application."""
        return json.loads(self.API_KEYS_JSON)

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for rate limiting state.",
    )
    REDIS_PASSWORD: str = Field(
        default="",
        description="Redis AUTH password (leave empty to disable).",
    )

    # ── Rate Limiting Tiers ───────────────────────────────────────────────────
    RATE_LIMIT_FREE: int = Field(
        default=60,
        description="Max requests per minute for free-tier API keys.",
    )
    RATE_LIMIT_PREMIUM: int = Field(
        default=5000,
        description="Max requests per minute for premium-tier API keys.",
    )
    RATE_LIMIT_ENTERPRISE: int = Field(
        default=0,  # 0 = unlimited
        description="Max requests per minute for enterprise-tier (0 = unlimited).",
    )
    RATE_LIMIT_WINDOW_SECONDS: int = Field(
        default=60,
        description="Sliding-window duration in seconds.",
    )


# ── Module-level singleton ────────────────────────────────────────────────────
settings = Settings()
