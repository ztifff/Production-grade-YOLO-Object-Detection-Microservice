# Prompt Engineering Plan: Production-Ready YOLO Microservice

This document outlines a structured, step-by-step Prompt Engineering Plan to build, optimize, containerize, and monetize a commercial **YOLO Object Detection Microservice**. 

Follow this sequential prompting strategy to guide an AI assistant in generating the complete commercial-grade product repository.

---

## Phase 1: High-Performance Core Inference Engine

### Prompt 1.1: Optimized Object Detection Worker
```text
Act as a Principal Python Engineer specializing in Computer Vision and High-Performance Computing. 

Write an production-grade inference module using the 'ultralytics' library for YOLOv8/YOLOv11. The architecture must conform to these strict enterprise requirements:
1. Singleton Model Loader: Implement a thread-safe Singleton pattern to ensure model weights (.pt or .onnx) are loaded into memory or GPU exactly once at initialization, avoiding reload overhead per request.
2. Robust Error Handling: Wrap the inference pipeline in detailed try-except blocks, handling corrupted images, missing model files, and out-of-memory (OOM) errors gracefully.
3. Configurable Settings: Allow confidence thresholds, IOU thresholds, and device settings (cpu vs cuda) to be passed dynamically or fallback to environment variables.
4. Structured JSON Output: The function must accept an image payload (byte array or file path) and return a fully serializable Python dictionary containing:
   - 'label' (string)
   - 'confidence' (float, rounded to 4 decimals)
   - 'box' (dictionary with x_min, y_min, x_max, y_max absolute coordinates)
   - 'execution_time_ms' (float)

Provide clean, PEP-8 compliant code with descriptive type hinting, logger integrations (avoiding print statements), and inline documentation.
```

---

## Phase 2: Asynchronous Microservice Layer & Concurrency

### Prompt 2.1: FastAPI Asynchronous Router
```text
Act as an Expert Backend Engineer specializing in building enterprise-grade APIs. 

Using FastAPI, create an asynchronous API gateway layout that wraps our previous YOLO core inference engine. The service must handle high-concurrency commercial workloads. Include the following features:
1. Input Validation with Pydantic: Ensure incoming data supports both multipart/form-data image uploads and Base64 encoded string images with explicit size limitations (e.g., max 10MB) to prevent Denial of Service (DoS).
2. Lifespan Event Handling: Use FastAPI's 'lifespan' context manager to initialize the global YOLO model instance during startup and safely clean up resources on shutdown.
3. Concurrency via ThreadPoolExecutor: Because ML inference is CPU/GPU bound and blocks the event loop, safely offload the inference call using FastAPI's 'run_in_threadpool' or an explicit 'concurrent.futures.ThreadPoolExecutor' pool.
4. Production-Ready Endpoints:
   - POST /api/v1/vision/detect: Main inference route.
   - GET /healthz: Kubernetes-compatible liveness and readiness probe checking system health and model initialization state.
   - GET /metrics: Basic stub to track total requests and error rates.

Ensure code is structured cleanly with separate routers and modules.
```

---

## Phase 3: Monetization, Security & Rate Limiting

### Prompt 3.1: API Key Middleware & Usage Metering
```text
Act as a SaaS Security Architect. Enhance the existing FastAPI application with a robust monetization and security middleware layer to prepare it for commercial B2B sales. 

Implement the following:
1. API Key Authentication: Secure the '/api/v1/vision/detect' route using a custom security middleware or FastAPI security dependencies ('HTTPBearer' or 'APIKeyHeader').
2. Sliding-Window Rate Limiting: Implement a memory-efficient rate limiter (using an in-memory dictionary or integrated Redis structure) to restrict tenants/clients based on their API key tier (e.g., Free Tier = 60 requests/min, Premium Tier = 5000 requests/min).
3. Payload Usage Logging: On every successful inference request, log metadata tracking the client ID, API Key used, processing time, and the number of objects detected. Format this into a standardized structured log payload that can be easily pushed to an enterprise logging aggregator (like ELK Stack or Datadog) or saved to a database for billing generation.

Write modular code that can easily be attached to the router built in Phase 2.
```

---

## Phase 4: Commercial DevOps & Production Deployment

### Prompt 4.1: Optimized Multi-Stage Dockerfile
```text
Act as a Senior DevOps and MLOps Engineer. Write a commercial-grade, multi-stage Dockerfile optimized to build, compress, and run our Python YOLO FastAPI service. 

Your configuration must follow modern industry best practices:
1. Multi-Stage Build: Use a build stage to install compilers and dependency wheels, and a separate minimal runner stage to minimize image footprint.
2. Dependency Separation: Use a lightweight base image (e.g., python:3.11-slim or nvidia/cuda base if GPU is specified). Handle heavy system dependencies cleanly (e.g., libGL.so, libgomp) required by OpenCV without bloating the image.
3. Security Hardening: Never run the container application as root. Create a dedicated system user and group (e.g., 'appuser') with restricted filesystem permissions.
4. Build Optimizations: Configure environment variables to stop Python from writing bytecode (.pyc) and disable stdout/stderr buffering (PYTHONUNBUFFERED=1) for instant container streaming logs.
5. Production Server Configuration: Configure the ENTRYPOINT to spin up the application via Uvicorn or Gunicorn with multiple worker processes, specifying bound hosts and ports dynamically.
```

### Prompt 4.2: Production Docker-Compose with Redis Cache
```text
Write a production-ready docker-compose.yml file that orchestrates the entire monetization platform. 

The compose stack must spin up and tie together:
1. The FastAPI YOLO Microservice: Built from the custom Dockerfile, passing configuration variables (Model Paths, Confidence thresholds, System Keys) through environment configurations.
2. A Redis Instance: Configured as an internal caching/rate-limiting backend layer to keep state across API workers.
3. Automated Healthchecks: Define health checks for all services with appropriate intervals, timeouts, and retries.
4. Volume/Network Segmentation: Keep storage volumes isolated and isolate backend communication on a dedicated private network while exposing only the API Gateway to the public host.
```