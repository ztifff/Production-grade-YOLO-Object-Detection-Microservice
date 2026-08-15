"""
app/middleware/rate_limiter.py
──────────────────────────────
Sliding-window rate limiter backed by Redis.

Algorithm
─────────────────────────────────────────────────────────────────────────────
A Redis **sorted set** is maintained per API key where:
  - Each member is a unique request UUID.
  - Each score is the Unix timestamp (float) of the request.

On every request:
  1. Remove stale members (score < now - window_seconds).
  2. Count remaining members (requests in current window).
  3. If count >= limit → reject with HTTP 429 + Retry-After header.
  4. Otherwise → add current request and set TTL on the key.

This is O(log N) per request with no background cleanup jobs needed.

Fallback
──────────────────────────────────────────────────────────────────────────────
If Redis is unavailable at import time (e.g., local dev without Docker),
the module automatically degrades to an **in-memory** deque-based
implementation.  In-memory state is per-process and NOT shared across
workers — adequate for development only.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.logging import get_logger
from app.middleware.auth import TenantContext

logger = get_logger(__name__)

# ── Tier → request limit map ──────────────────────────────────────────────────
TIER_LIMITS: Dict[str, int] = {
    "free": settings.RATE_LIMIT_FREE,
    "premium": settings.RATE_LIMIT_PREMIUM,
    "enterprise": settings.RATE_LIMIT_ENTERPRISE,  # 0 = unlimited
}


def _limit_for_tier(tier: str) -> int:
    """Return the per-window request cap for *tier*. 0 means unlimited."""
    return TIER_LIMITS.get(tier, settings.RATE_LIMIT_FREE)


# ── Abstract base ─────────────────────────────────────────────────────────────

class _BaseRateLimiter(ABC):
    """Common interface for both Redis and in-memory backends."""

    @abstractmethod
    def check_and_record(self, key: str, limit: int) -> Tuple[bool, int, float]:
        """
        Atomically check and record a request.

        Parameters
        ----------
        key:
            Unique string key per client (typically API key prefix + tier).
        limit:
            Maximum allowed requests in the current window.

        Returns
        -------
        Tuple of (allowed: bool, current_count: int, retry_after_seconds: float)
        """


# ── Redis backend ─────────────────────────────────────────────────────────────

class _RedisRateLimiter(_BaseRateLimiter):
    """Sliding-window rate limiter using Redis sorted sets."""

    def __init__(self, redis_url: str, password: Optional[str] = None) -> None:
        import redis as redis_lib  # noqa: PLC0415

        kwargs: Dict = {"decode_responses": True}
        if password:
            kwargs["password"] = password
        self._client = redis_lib.from_url(redis_url, **kwargs)
        # Verify connection eagerly
        self._client.ping()
        logger.info("Rate limiter connected to Redis", extra={"url": redis_url})

    def check_and_record(self, key: str, limit: int) -> Tuple[bool, int, float]:
        window = settings.RATE_LIMIT_WINDOW_SECONDS
        now = time.time()
        window_start = now - window

        pipe = self._client.pipeline()
        # 1. Remove expired members
        pipe.zremrangebyscore(key, "-inf", window_start)
        # 2. Count active members
        pipe.zcard(key)
        # 3. Add this request
        pipe.zadd(key, {str(uuid.uuid4()): now})
        # 4. Set TTL so Redis auto-cleans idle keys
        pipe.expire(key, window * 2)

        results = pipe.execute()
        current_count: int = results[1]  # before this request

        if current_count >= limit:
            # Calculate when the oldest request will expire
            oldest = self._client.zrange(key, 0, 0, withscores=True)
            retry_after = window - (now - oldest[0][1]) if oldest else float(window)
            return False, current_count, retry_after

        return True, current_count + 1, 0.0


# ── In-memory fallback ────────────────────────────────────────────────────────

class _InMemoryRateLimiter(_BaseRateLimiter):
    """
    Per-process deque-based fallback when Redis is unavailable.

    NOT safe for multi-worker deployments.
    """

    def __init__(self) -> None:
        self._buckets: Dict[str, Deque[float]] = {}
        logger.warning(
            "Rate limiter using in-memory fallback — not suitable for production "
            "multi-worker deployments."
        )

    def check_and_record(self, key: str, limit: int) -> Tuple[bool, int, float]:
        window = float(settings.RATE_LIMIT_WINDOW_SECONDS)
        now = time.time()
        window_start = now - window

        if key not in self._buckets:
            self._buckets[key] = deque()

        bucket = self._buckets[key]

        # Evict expired timestamps
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= limit:
            retry_after = window - (now - bucket[0]) if bucket else window
            return False, len(bucket), retry_after

        bucket.append(now)
        return True, len(bucket), 0.0


# ── Factory ───────────────────────────────────────────────────────────────────

def _build_limiter() -> _BaseRateLimiter:
    try:
        return _RedisRateLimiter(
            redis_url=settings.REDIS_URL,
            password=settings.REDIS_PASSWORD or None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Redis unavailable — falling back to in-memory rate limiter",
            extra={"error": str(exc)},
        )
        return _InMemoryRateLimiter()


# Module-level singleton initialised lazily on first use
_limiter: Optional[_BaseRateLimiter] = None


def get_limiter() -> _BaseRateLimiter:
    """Return (or create) the module-level rate limiter instance."""
    global _limiter  # noqa: PLW0603
    if _limiter is None:
        _limiter = _build_limiter()
    return _limiter


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def enforce_rate_limit(tenant: TenantContext) -> None:
    """
    FastAPI dependency that enforces the sliding-window rate limit for *tenant*.

    Designed to be chained after :func:`app.middleware.auth.require_api_key`::

        @router.post("/detect")
        async def detect(
            tenant: TenantContext = Security(require_api_key),
            _: None = Depends(enforce_rate_limit_dep),
        ):
            ...

    Raises
    ------
    HTTP 429
        When the tenant exceeds their tier's request limit.
    """
    limit = _limit_for_tier(tenant.tier)

    # Enterprise tier with limit 0 → skip all checks
    if limit == 0:
        return

    limiter = get_limiter()
    bucket_key = f"rl:{tenant.client_id}:{tenant.tier}"
    allowed, count, retry_after = limiter.check_and_record(bucket_key, limit)

    if not allowed:
        logger.warning(
            "Rate limit exceeded",
            extra={
                "client_id": tenant.client_id,
                "tier": tenant.tier,
                "current_count": count,
                "limit": limit,
                "retry_after": retry_after,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": (
                    f"You have exceeded the {limit} requests/min limit for the "
                    f"'{tenant.tier}' tier. Upgrade your plan for higher limits."
                ),
                "retry_after_seconds": round(retry_after, 1),
            },
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
