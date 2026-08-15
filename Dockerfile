# syntax=docker/dockerfile:1.6
# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for the YOLO Vision API
#
# Stages:
#   builder  — installs Python dependencies into a virtual environment
#   runner   — minimal runtime image copying only the venv + app code
#
# GPU variant (uncomment GPU sections below and comment out CPU base images):
#   Replace python:3.11-slim with nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
#   and add cuda toolkit libs to the runner stage.
# ─────────────────────────────────────────────────────────────────────────────

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 1 — builder                                                       ║
# ║  Install all Python dependencies and compile C extensions.               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
FROM python:3.11-slim AS builder

# --- GPU variant (uncomment to use CUDA) ---
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder
# RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip \
#     && ln -s /usr/bin/python3.11 /usr/bin/python3 \
#     && ln -s /usr/bin/python3 /usr/bin/python

# Build-time system dependencies (compilers, headers for native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        curl \
        # OpenCV build dependencies
        libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment — we copy only this into the runner
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install wheel first to speed up subsequent installs
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt /tmp/requirements.txt

# Install all dependencies into the venv
# --no-cache-dir keeps the image lean; wheels are already cached above
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STAGE 2 — runner                                                        ║
# ║  Minimal runtime image; no compilers, no build tools.                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
FROM python:3.11-slim AS runner

# --- GPU variant (uncomment to use CUDA) ---
# FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runner

# ── Runtime system dependencies only ─────────────────────────────────────────
# libgl1       : required by OpenCV (libGL.so.1)
# libglib2.0-0 : required by OpenCV (libgthread-2.0.so.0)
# libgomp1     : required by PyTorch / YOLO for OpenMP parallelism
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        # curl for optional container-level healthcheck
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment tuning ─────────────────────────────────────────────────
# Do not write .pyc bytecode files (keeps image clean)
ENV PYTHONDONTWRITEBYTECODE=1
# Disable output buffering so logs appear immediately in Docker / Kubernetes
ENV PYTHONUNBUFFERED=1
# Point PATH to our venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ── Security hardening — non-root user ───────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --no-create-home --shell /sbin/nologin appuser

# ── Application directory ─────────────────────────────────────────────────────
WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv

# Copy application source code
COPY --chown=appuser:appgroup app/ ./app/

# Create the models directory; the actual .pt file is mounted at runtime
RUN mkdir -p /app/models && chown appuser:appgroup /app/models

# Switch to non-root user
USER appuser

# ── Runtime configuration ─────────────────────────────────────────────────────
# These defaults are overridden by docker-compose / Kubernetes env vars
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=4
ENV LOG_LEVEL=info

EXPOSE $PORT

# ── Health check (Docker engine level) ───────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Gunicorn manages worker processes; UvicornWorker provides ASGI support.
# Workers = (2 * CPU_cores) + 1 is the standard recommendation for I/O-bound.
# For CPU/GPU-bound ML workloads, set WORKERS to the number of GPU devices
# or to a value that prevents GPU OOM under concurrent load.
ENTRYPOINT ["sh", "-c", \
    "exec gunicorn app.main:app \
        --worker-class uvicorn.workers.UvicornWorker \
        --workers ${WORKERS} \
        --bind ${HOST}:${PORT} \
        --timeout 120 \
        --graceful-timeout 30 \
        --keep-alive 5 \
        --access-logfile - \
        --error-logfile - \
        --log-level ${LOG_LEVEL}" \
]
