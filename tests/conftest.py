"""
tests/conftest.py
─────────────────
Shared pytest fixtures for the YOLO microservice test suite.

Provides:
  - A FastAPI TestClient with a mocked YOLOEngine (no real model required)
  - Sample image bytes (1x1 red JPEG generated in-memory)
  - API key fixtures matching the dev tier map
"""

from __future__ import annotations

import io
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Fixtures ──────────────────────────────────────────────────────────────────

FREE_API_KEY = "sk-dev-00000000"
PREMIUM_API_KEY = "sk-prem-test99"

# API_KEYS map used by all tests — injected via env override in app settings
TEST_API_KEYS = {
    FREE_API_KEY: {"client_id": "test-free", "tier": "free"},
    PREMIUM_API_KEY: {"client_id": "test-premium", "tier": "premium"},
}

# A minimal 1×1 red JPEG in bytes — used as a valid image fixture
@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(255, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


# Mock inference result returned by the mocked YOLOEngine
MOCK_INFERENCE_RESULT = {
    "detections": [
        {
            "label": "person",
            "confidence": 0.9342,
            "box": {"x_min": 120.5, "y_min": 80.2, "x_max": 340.1, "y_max": 510.7},
            "class_id": 0,
        }
    ],
    "object_count": 1,
    "execution_time_ms": 38.42,
    "image_width": 640,
    "image_height": 480,
    "model": "yolov8n.pt",
}


@pytest.fixture()
def mock_engine() -> Generator[MagicMock, None, None]:
    """Patch YOLOEngine.get_instance to return a mock that never loads weights."""
    mock = MagicMock()
    mock.is_ready = True
    mock.model_name = "yolov8n.pt"
    mock.detect.return_value = MOCK_INFERENCE_RESULT

    with patch("app.inference.engine.YOLOEngine.get_instance", return_value=mock), \
         patch("app.inference.engine.YOLOEngine._instance", mock):
        yield mock


@pytest.fixture()
def client(mock_engine) -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient with:
      - YOLOEngine mocked (no GPU/model required)
      - API_KEYS env var set to TEST_API_KEYS
      - Redis rate limiter replaced with in-memory fallback
    """
    import json
    from unittest.mock import patch as _patch

    # Override settings before the app is imported
    with _patch.dict(
        "os.environ",
        {"API_KEYS": json.dumps(TEST_API_KEYS)},
        clear=False,
    ):
        # Re-import app after patching env so Settings picks up TEST_API_KEYS
        from app.main import create_app  # noqa: PLC0415

        test_app = create_app()
        with TestClient(test_app, raise_server_exceptions=False) as c:
            yield c
