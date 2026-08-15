"""
tests/test_engine.py
────────────────────
Unit tests for the YOLOEngine Singleton.

These tests exercise the engine's error handling and Singleton behaviour
without requiring real model weights by mocking the Ultralytics import.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.inference.engine import (
    InvalidImageError,
    ModelLoadError,
    YOLOEngine,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure a clean Singleton state before each test."""
    YOLOEngine.reset()
    yield
    YOLOEngine.reset()


def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(0, 128, 255)).save(buf, format="JPEG")
    return buf.getvalue()


class TestSingleton:
    def test_same_instance_returned(self, tmp_path):
        """Two calls to get_instance() must return the identical object."""
        fake_pt = tmp_path / "fake.pt"
        fake_pt.write_bytes(b"FAKE")

        mock_model = MagicMock()
        mock_model.predict.return_value = []

        with patch("app.inference.engine.YOLO", return_value=mock_model, create=True), \
             patch("ultralytics.YOLO", return_value=mock_model, create=True):
            # Patch the import inside _load_model
            with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=MagicMock(return_value=mock_model))}):
                a = YOLOEngine.get_instance(model_path=str(fake_pt), device="cpu", conf=0.25, iou=0.45, max_det=100)
                b = YOLOEngine.get_instance()
                assert a is b

    def test_missing_model_raises(self):
        """A non-existent model path must raise ModelLoadError."""
        with pytest.raises(ModelLoadError, match="not found"):
            YOLOEngine.get_instance(
                model_path="/non/existent/model.pt",
                device="cpu",
                conf=0.25,
                iou=0.45,
                max_det=100,
            )


class TestImageDecoding:
    def test_valid_jpeg_bytes(self):
        """Valid JPEG bytes should decode to a numpy array."""
        arr, w, h = YOLOEngine._decode_image(_make_jpeg(100, 80))
        assert isinstance(arr, np.ndarray)
        assert w == 100
        assert h == 80

    def test_corrupted_bytes_raises(self):
        """Garbage bytes should raise InvalidImageError."""
        with pytest.raises(InvalidImageError):
            YOLOEngine._decode_image(b"\x00\xFF\xD8corrupted!!")

    def test_empty_bytes_raises(self):
        """Empty payload should raise InvalidImageError."""
        with pytest.raises(InvalidImageError):
            YOLOEngine._decode_image(b"")
