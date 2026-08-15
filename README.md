# YOLO Vision API

**Production-grade YOLO Object Detection Microservice** — built with YOLOv8/v11, FastAPI, Redis, and Docker.

Secure, monetisable, and deployable in under 5 minutes.

---

## Features

| Feature | Details |
|---------|---------|
| **Inference Engine** | Thread-safe Singleton, GPU OOM fallback, ONNX + `.pt` support |
| **Async API** | FastAPI + ThreadPoolExecutor, multipart upload + Base64 JSON |
| **Authentication** | `X-API-Key` header, per-tenant `TenantContext` |
| **Rate Limiting** | Redis sliding-window per API key tier |
| **Usage Metering** | Structured NDJSON billing logs (ELK/Datadog/CloudWatch ready) |
| **Containerisation** | Multi-stage Dockerfile, non-root user, Gunicorn + UvicornWorker |
| **Orchestration** | docker-compose with Redis, healthchecks, network segmentation |
| **Observability** | `/healthz` K8s probe, `/metrics` counter snapshot |

---

## Quick Start

### 1. Clone and configure

```bash
git clone <your-repo-url> && cd YOLO
cp .env.example .env
# Edit .env — set your API_KEYS and MODEL_PATH
```

### 2. Download model weights

```bash
mkdir -p models
# YOLOv8 nano (fastest, ~6 MB)
curl -L https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt \
     -o models/yolov8n.pt
```

### 3. Start the stack

```bash
docker compose up --build
```

The API is available at **http://localhost:8000**

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health probe**: http://localhost:8000/healthz

---

## API Reference

### Authentication

All inference endpoints require an `X-API-Key` header:

```http
X-API-Key: sk-your-api-key-here
```

| Status | Condition |
|--------|-----------|
| `401 Unauthorized` | Header missing |
| `403 Forbidden` | Key not in the `API_KEYS` map |

---

### POST `/api/v1/vision/detect`

Detect objects via **multipart file upload**.

**Request**

```bash
curl -X POST http://localhost:8000/api/v1/vision/detect \
  -H "X-API-Key: sk-your-key" \
  -F "file=@/path/to/image.jpg" \
  -F "conf=0.4"
```

**Optional form fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `conf` | float | settings default | Confidence threshold override (0.01–1.0) |
| `iou` | float | settings default | IoU threshold override (0.01–1.0) |

**Response `200 OK`**

```json
{
  "success": true,
  "model": "yolov8n.pt",
  "image_width": 1280,
  "image_height": 720,
  "execution_time_ms": 42.3,
  "object_count": 3,
  "detections": [
    {
      "label": "person",
      "confidence": 0.9342,
      "box": {
        "x_min": 120.5,
        "y_min": 80.2,
        "x_max": 340.1,
        "y_max": 510.7
      },
      "class_id": 0
    }
  ]
}
```

---

### POST `/api/v1/vision/detect/base64`

Detect objects via **Base64-encoded JSON body**.

```bash
curl -X POST http://localhost:8000/api/v1/vision/detect/base64 \
  -H "X-API-Key: sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "'"$(base64 -w 0 /path/to/image.jpg)"'",
    "conf": 0.3
  }'
```

Accepts optional `data:image/jpeg;base64,...` data URI prefix.

---

### GET `/healthz`

Kubernetes liveness & readiness probe.

```bash
curl http://localhost:8000/healthz
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "yolov8n.pt",
  "version": "1.0.0",
  "uptime_seconds": 142.7
}
```

Returns **`200`** when model is loaded, **`503`** otherwise.

---

### GET `/metrics`

Operational counter snapshot.

```json
{
  "total_requests": 1024,
  "successful_requests": 1019,
  "failed_requests": 5,
  "rate_limited_requests": 12,
  "avg_latency_ms": 38.4,
  "error_rate": 0.0049,
  "active_api_keys": 3,
  "tier_breakdown": {
    "free": 200,
    "premium": 812,
    "enterprise": 12
  }
}
```

---

## Configuration Reference

All settings are loaded from environment variables or `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/yolov8n.pt` | Path to YOLO weights |
| `YOLO_CONF` | `0.25` | Confidence threshold |
| `YOLO_IOU` | `0.45` | IoU threshold (NMS) |
| `YOLO_DEVICE` | `cpu` | `cpu`, `cuda`, `cuda:0`, `mps` |
| `YOLO_MAX_DET` | `300` | Max detections per image |
| `MAX_IMAGE_BYTES` | `10485760` | 10 MB upload limit |
| `WORKERS` | `4` | Gunicorn worker count |
| `API_KEYS` | (dev key) | JSON map of keys → tenant metadata |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `REDIS_PASSWORD` | `` | Redis AUTH password |
| `RATE_LIMIT_FREE` | `60` | Free tier: req/min |
| `RATE_LIMIT_PREMIUM` | `5000` | Premium tier: req/min |
| `RATE_LIMIT_ENTERPRISE` | `0` | Enterprise: 0 = unlimited |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window duration |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `ENV` | `production` | Environment label |

---

## API Key Tiers

| Tier | Requests/min | Use case |
|------|-------------|---------|
| `free` | 60 | Trials, demos |
| `premium` | 5 000 | Commercial clients |
| `enterprise` | Unlimited | High-volume / SLA customers |

**Generating a secure key:**

```bash
python -c "import secrets; print('sk-' + secrets.token_hex(24))"
```

**Adding a key to `.env`:**

```bash
API_KEYS={"sk-free-abc123": {"client_id": "acme", "tier": "free"}, "sk-prem-xyz789": {"client_id": "bigco", "tier": "premium"}}
```

---

## Deployment

### Docker Compose (recommended)

```bash
# Build and start
docker compose up --build -d

# Tail logs
docker compose logs -f yolo-api

# Scale workers (without Redis scaling)
docker compose up --build -d --scale yolo-api=3

# Stop everything
docker compose down
```

### GPU Support

1. Edit `Dockerfile` — uncomment the `nvidia/cuda` base image lines and comment out `python:3.11-slim`.
2. Set `YOLO_DEVICE=cuda` in `.env`.
3. Add `runtime: nvidia` to the `yolo-api` service in `docker-compose.yml`.
4. Ensure `nvidia-container-toolkit` is installed on the host.

### Kubernetes

```bash
# Build and push image
docker build -t your-registry/yolo-vision-api:1.0.0 .
docker push your-registry/yolo-vision-api:1.0.0

# Deploy (example — adapt to your cluster)
kubectl create secret generic yolo-secrets \
  --from-literal=API_KEYS='{"sk-key": {"client_id": "prod", "tier": "enterprise"}}' \
  --from-literal=REDIS_PASSWORD='your-redis-password'
```

Configure `/healthz` as both `livenessProbe` and `readinessProbe` with `initialDelaySeconds: 90`.

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v --tb=short --cov=app --cov-report=term-missing
```

---

## Billing Log Format

Every successful inference emits a structured JSON line to stdout:

```json
{
  "timestamp": "2026-08-15T15:12:43",
  "level": "INFO",
  "event": "inference_complete",
  "client_id": "acme-corp",
  "api_key_hash": "sha256:a3f8c2d1...",
  "tier": "premium",
  "processing_ms": 42.3,
  "objects_detected": 5,
  "image_size_bytes": 204800,
  "image_width": 1280,
  "image_height": 720,
  "model": "yolov8n.pt",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Ship these logs to your aggregator with Filebeat, the Datadog agent, or `fluentd`.

---

## Project Structure

```
app/
├── main.py               # FastAPI factory + lifespan
├── core/
│   ├── config.py         # Pydantic settings (env vars)
│   └── logging.py        # JSON logger
├── inference/
│   └── engine.py         # Thread-safe Singleton YOLO engine
├── api/
│   ├── health.py         # GET /healthz
│   ├── metrics.py        # GET /metrics
│   └── v1/
│       ├── router.py     # v1 aggregator
│       └── vision.py     # POST /api/v1/vision/detect
├── middleware/
│   ├── auth.py           # API Key dependency
│   ├── rate_limiter.py   # Redis sliding-window
│   └── usage_logger.py   # Billing event logger
└── schemas/
    ├── detection.py      # I/O Pydantic models
    └── health.py         # Health/metrics schemas
```

---

## License

MIT — see `LICENSE` for details.
