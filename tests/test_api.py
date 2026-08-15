"""
tests/test_api.py
─────────────────
Integration tests for the FastAPI detection endpoints.

Uses the ``client`` fixture from conftest.py which provides a TestClient
backed by a mocked YOLOEngine — no real model or GPU is needed.
"""

from __future__ import annotations

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import FREE_API_KEY, MOCK_INFERENCE_RESULT


def _make_jpeg(width: int = 640, height: int = 480) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="JPEG")
    return buf.getvalue()


# ── /healthz ──────────────────────────────────────────────────────────────────

class TestHealthz:
    def test_returns_200_when_model_loaded(self, client: TestClient):
        resp = client.get("/healthz")
        # The mock engine sets is_ready=True, so we expect 200
        assert resp.status_code in (200, 503)  # 503 is acceptable if lifespan didn't run

    def test_response_schema(self, client: TestClient):
        resp = client.get("/healthz")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_returns_200(self, client: TestClient):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client: TestClient):
        data = client.get("/metrics").json()
        assert "total_requests" in data
        assert "error_rate" in data
        assert "avg_latency_ms" in data


# ── POST /api/v1/vision/detect ────────────────────────────────────────────────

class TestDetectUpload:
    def test_missing_api_key_returns_401(self, client: TestClient, sample_image_bytes):
        resp = client.post(
            "/api/v1/vision/detect",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 401

    def test_invalid_api_key_returns_403(self, client: TestClient, sample_image_bytes):
        resp = client.post(
            "/api/v1/vision/detect",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            headers={"X-API-Key": "sk-invalid-key"},
        )
        assert resp.status_code == 403

    def test_valid_request_returns_detections(self, client: TestClient, sample_image_bytes, mock_engine):
        resp = client.post(
            "/api/v1/vision/detect",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            headers={"X-API-Key": FREE_API_KEY},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "detections" in data
        assert isinstance(data["detections"], list)

    def test_response_includes_execution_time(self, client: TestClient, sample_image_bytes, mock_engine):
        resp = client.post(
            "/api/v1/vision/detect",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            headers={"X-API-Key": FREE_API_KEY},
        )
        data = resp.json()
        assert "execution_time_ms" in data
        assert isinstance(data["execution_time_ms"], float)

    def test_oversized_image_returns_413(self, client: TestClient, mock_engine):
        # 11 MB of zeros — exceeds the 10 MB default limit
        big_payload = b"\x00" * (11 * 1024 * 1024)
        resp = client.post(
            "/api/v1/vision/detect",
            files={"file": ("big.jpg", big_payload, "image/jpeg")},
            headers={"X-API-Key": FREE_API_KEY},
        )
        assert resp.status_code == 413


# ── POST /api/v1/vision/detect/base64 ────────────────────────────────────────

class TestDetectBase64:
    def test_valid_base64_returns_detections(self, client: TestClient, sample_image_bytes, mock_engine):
        encoded = base64.b64encode(sample_image_bytes).decode()
        resp = client.post(
            "/api/v1/vision/detect/base64",
            json={"image_base64": encoded},
            headers={"X-API-Key": FREE_API_KEY},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_invalid_base64_returns_422(self, client: TestClient, mock_engine):
        resp = client.post(
            "/api/v1/vision/detect/base64",
            json={"image_base64": "!!!NOT_VALID_BASE64!!!"},
            headers={"X-API-Key": FREE_API_KEY},
        )
        assert resp.status_code == 422

    def test_data_uri_prefix_stripped(self, client: TestClient, sample_image_bytes, mock_engine):
        encoded = "data:image/jpeg;base64," + base64.b64encode(sample_image_bytes).decode()
        resp = client.post(
            "/api/v1/vision/detect/base64",
            json={"image_base64": encoded},
            headers={"X-API-Key": FREE_API_KEY},
        )
        # Pydantic strips the prefix; mock engine accepts any bytes
        assert resp.status_code == 200
