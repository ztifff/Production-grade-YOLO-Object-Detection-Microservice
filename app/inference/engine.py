"""
app/inference/engine.py
───────────────────────
Thread-safe Singleton YOLO inference engine.

Design decisions
────────────────
* **Singleton via double-checked locking**: The class-level ``_instance``
  attribute is guarded by a ``threading.Lock`` so that concurrent startup
  threads cannot race to load the model more than once.

* **Lazy image decoding**: Accepts either raw ``bytes`` (HTTP upload) or a
  filesystem path ``str``/``Path`` so callers never need to pre-process.

* **Graceful error taxonomy**: Distinguishes model-load errors from
  inference errors, logs them with full tracebacks, and re-raises typed
  exceptions that the API layer maps to appropriate HTTP status codes.

* **Zero global state outside the class**: All mutable state lives on the
  Singleton instance, making the engine fully testable via dependency
  injection.
"""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.detection import BoundingBox, DetectionResult

logger = get_logger(__name__)

# ── Typed aliases ─────────────────────────────────────────────────────────────
ImagePayload = Union[bytes, str, Path]


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class ModelLoadError(RuntimeError):
    """Raised when the YOLO model cannot be initialised."""


class InferenceError(RuntimeError):
    """Raised when inference fails for a specific image."""


class InvalidImageError(ValueError):
    """Raised when the provided image bytes cannot be decoded."""


# ── Singleton Engine ──────────────────────────────────────────────────────────

class YOLOEngine:
    """
    Thread-safe Singleton wrapper around the Ultralytics YOLO model.

    Usage::

        engine = YOLOEngine.get_instance()
        results = engine.detect(image_bytes)
    """

    _instance: Optional["YOLOEngine"] = None
    _lock: threading.Lock = threading.Lock()

    # ── Constructor (internal) ────────────────────────────────────────────────

    def __init__(
        self,
        model_path: str,
        device: str,
        conf: float,
        iou: float,
        max_det: int,
    ) -> None:
        """
        Load the YOLO model from *model_path*.  Never call this directly —
        use :meth:`get_instance` instead.
        """
        self._model_path = Path(model_path)
        self._device = device
        self._default_conf = conf
        self._default_iou = iou
        self._max_det = max_det
        self._model: Any = None  # ultralytics.YOLO instance
        self._model_name: str = self._model_path.name
        self._ready: bool = False

        self._load_model()

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Import Ultralytics and load weights.

        Raises
        ------
        ModelLoadError
            If the model file is missing or corrupted, or if the GPU runs
            out of memory during model initialisation.
        """
        try:
            # Lazy import — keeps startup fast even if ultralytics is not yet
            # installed when running unit tests with mocked engines.
            from ultralytics import YOLO  # noqa: PLC0415

            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"Model weights not found at '{self._model_path}'. "
                    "Download the file or set MODEL_PATH correctly."
                )

            logger.info(
                "Loading YOLO model",
                extra={
                    "model_path": str(self._model_path),
                    "device": self._device,
                },
            )
            self._model = YOLO(str(self._model_path))
            # Warm-up: move model to requested device
            self._model.to(self._device)
            self._ready = True

            logger.info(
                "YOLO model loaded successfully",
                extra={
                    "model_name": self._model_name,
                    "device": self._device,
                    "conf": self._default_conf,
                    "iou": self._default_iou,
                },
            )

        except FileNotFoundError as exc:
            logger.error("Model file missing", extra={"path": str(self._model_path)})
            raise ModelLoadError(str(exc)) from exc

        except MemoryError as exc:
            logger.error("Out-of-memory loading model", extra={"device": self._device})
            raise ModelLoadError("Insufficient memory to load model weights.") from exc

        except Exception as exc:  # noqa: BLE001
            # Catch torch.cuda.OutOfMemoryError and any other Ultralytics
            # initialisation errors without importing torch at module level.
            cls_name = type(exc).__name__
            if "OutOfMemory" in cls_name or "CUDA" in str(exc):
                logger.error(
                    "GPU OOM during model load — retrying on CPU",
                    extra={"original_device": self._device},
                )
                self._device = "cpu"
                try:
                    from ultralytics import YOLO  # noqa: PLC0415
                    self._model = YOLO(str(self._model_path))
                    self._model.to("cpu")
                    self._ready = True
                    logger.warning("Fell back to CPU inference due to GPU OOM.")
                    return
                except Exception as cpu_exc:  # noqa: BLE001
                    raise ModelLoadError(
                        f"CPU fallback also failed: {cpu_exc}"
                    ) from cpu_exc
            logger.exception("Unexpected error loading YOLO model")
            raise ModelLoadError(f"Failed to load model: {exc}") from exc

    # ── Singleton factory ─────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: Optional[int] = None,
    ) -> "YOLOEngine":
        """
        Return the shared :class:`YOLOEngine` instance, creating it on the
        first call (double-checked locking).

        All parameters fall back to :mod:`app.core.config` values when omitted.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_path=model_path or settings.MODEL_PATH,
                        device=device or settings.YOLO_DEVICE,
                        conf=conf if conf is not None else settings.YOLO_CONF,
                        iou=iou if iou is not None else settings.YOLO_IOU,
                        max_det=max_det or settings.YOLO_MAX_DET,
                    )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Destroy the Singleton instance.

        Intended **only** for use in tests or hot-reload scenarios —
        not safe to call during live traffic.
        """
        with cls._lock:
            cls._instance = None
        logger.warning("YOLOEngine Singleton has been reset.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once the model has been loaded successfully."""
        return self._ready

    @property
    def model_name(self) -> str:
        """Filename of the active model weights."""
        return self._model_name

    # ── Image Decoding ────────────────────────────────────────────────────────

    @staticmethod
    def _decode_image(payload: ImagePayload) -> Tuple[np.ndarray, int, int]:
        """
        Convert *payload* to a decoded NumPy image array.

        Parameters
        ----------
        payload:
            Raw ``bytes``, a filesystem ``str`` path, or a ``pathlib.Path``.

        Returns
        -------
        Tuple of (numpy_array, width, height)

        Raises
        ------
        InvalidImageError
            If the bytes cannot be decoded as a valid image.
        """
        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

        try:
            if isinstance(payload, (str, Path)):
                img = Image.open(str(payload)).convert("RGB")
            else:
                img = Image.open(io.BytesIO(payload)).convert("RGB")

            width, height = img.size
            return np.array(img), width, height

        except UnidentifiedImageError as exc:
            raise InvalidImageError(
                "Cannot decode image — the data may be corrupted or an unsupported format."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise InvalidImageError(f"Image decode error: {exc}") from exc

    # ── Core Inference ────────────────────────────────────────────────────────

    def detect(
        self,
        payload: ImagePayload,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run object detection on *payload* and return a fully serialisable dict.

        Parameters
        ----------
        payload:
            Image as raw bytes, a path string, or a :class:`pathlib.Path`.
        conf:
            Per-request confidence threshold override.  Falls back to the
            engine's default if ``None``.
        iou:
            Per-request IoU threshold override.  Falls back to the
            engine's default if ``None``.

        Returns
        -------
        dict with keys:
          - ``detections``       — list of :class:`DetectionResult`-shaped dicts
          - ``object_count``     — int
          - ``execution_time_ms``— float
          - ``image_width``      — int
          - ``image_height``     — int
          - ``model``            — str

        Raises
        ------
        InvalidImageError
            If the image bytes cannot be decoded.
        InferenceError
            If the model raises an unexpected error during inference.
        ModelLoadError
            If the engine is not yet ready (should not normally occur after
            startup).
        """
        if not self._ready:
            raise ModelLoadError("YOLO engine is not ready — model failed to load.")

        effective_conf = conf if conf is not None else self._default_conf
        effective_iou = iou if iou is not None else self._default_iou

        # ── Decode ───────────────────────────────────────────────────────────
        image_array, img_width, img_height = self._decode_image(payload)

        # ── Inference ────────────────────────────────────────────────────────
        t_start = time.perf_counter()
        try:
            raw_results = self._model.predict(
                source=image_array,
                conf=effective_conf,
                iou=effective_iou,
                max_det=self._max_det,
                device=self._device,
                verbose=False,
            )
        except MemoryError as exc:
            raise InferenceError("Out-of-memory during inference.") from exc
        except Exception as exc:  # noqa: BLE001
            cls_name = type(exc).__name__
            if "OutOfMemory" in cls_name or "CUDA" in str(exc):
                raise InferenceError("GPU out-of-memory during inference.") from exc
            logger.exception("Unexpected inference error", extra={"payload_type": type(payload).__name__})
            raise InferenceError(f"Inference failed: {exc}") from exc

        t_elapsed_ms = (time.perf_counter() - t_start) * 1_000

        # ── Parse results ────────────────────────────────────────────────────
        detections: List[Dict[str, Any]] = []
        for result in raw_results:
            boxes = result.boxes
            if boxes is None:
                continue

            names: Dict[int, str] = result.names  # {class_id: label}

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()  # [x_min, y_min, x_max, y_max]
                conf_val = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                detections.append(
                    DetectionResult(
                        label=names.get(cls_id, str(cls_id)),
                        confidence=round(conf_val, 4),
                        box=BoundingBox(
                            x_min=float(xyxy[0]),
                            y_min=float(xyxy[1]),
                            x_max=float(xyxy[2]),
                            y_max=float(xyxy[3]),
                        ),
                        class_id=cls_id,
                    ).model_dump()
                )

        logger.info(
            "Inference complete",
            extra={
                "object_count": len(detections),
                "execution_time_ms": round(t_elapsed_ms, 2),
                "model": self._model_name,
                "conf": effective_conf,
                "iou": effective_iou,
            },
        )

        return {
            "detections": detections,
            "object_count": len(detections),
            "execution_time_ms": round(t_elapsed_ms, 2),
            "image_width": img_width,
            "image_height": img_height,
            "model": self._model_name,
        }
