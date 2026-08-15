"""
tests/test_middleware.py
────────────────────────
Unit tests for authentication and rate limiting middleware.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.middleware.auth import TenantContext, require_api_key
from app.middleware.rate_limiter import (
    TIER_LIMITS,
    _InMemoryRateLimiter,
    _limit_for_tier,
)


# ── Auth middleware ───────────────────────────────────────────────────────────

class TestAuth:
    @pytest.mark.asyncio
    async def test_valid_key_returns_tenant(self):
        from tests.conftest import FREE_API_KEY, TEST_API_KEYS
        import json

        with patch.dict("os.environ", {"API_KEYS": json.dumps(TEST_API_KEYS)}):
            # Re-instantiate settings with patched env
            from importlib import reload
            import app.core.config as cfg
            reload(cfg)

            from app.middleware import auth as auth_mod
            reload(auth_mod)

            tenant = await auth_mod.require_api_key(raw_key=FREE_API_KEY)
            assert tenant.client_id == "test-free"
            assert tenant.tier == "free"

    @pytest.mark.asyncio
    async def test_missing_key_raises_401(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(raw_key=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_key_raises_403(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await require_api_key(raw_key="sk-totally-wrong")
        assert exc_info.value.status_code == 403


# ── Rate limiter ──────────────────────────────────────────────────────────────

class TestInMemoryRateLimiter:
    def test_allows_requests_within_limit(self):
        limiter = _InMemoryRateLimiter()
        for _ in range(5):
            allowed, count, _ = limiter.check_and_record("test-key", limit=10)
            assert allowed

    def test_blocks_requests_over_limit(self):
        limiter = _InMemoryRateLimiter()
        limit = 3
        for _ in range(limit):
            limiter.check_and_record("test-key", limit=limit)

        # Next request should be rejected
        allowed, count, retry_after = limiter.check_and_record("test-key", limit=limit)
        assert not allowed
        assert count == limit
        assert retry_after > 0

    def test_sliding_window_clears_old_requests(self):
        """Entries older than the window should not count against the limit."""
        limiter = _InMemoryRateLimiter()
        limit = 2

        # Add entries that are "in the past" (older than the window)
        import app.middleware.rate_limiter as rl_mod
        old_time = time.time() - rl_mod.settings.RATE_LIMIT_WINDOW_SECONDS - 1
        limiter._buckets["old-key"] = __import__("collections").deque([old_time, old_time])

        allowed, count, _ = limiter.check_and_record("old-key", limit=limit)
        assert allowed  # Old entries evicted; count reset to 1

    def test_different_keys_are_isolated(self):
        limiter = _InMemoryRateLimiter()
        limit = 2
        # Exhaust key-A
        for _ in range(limit):
            limiter.check_and_record("key-A", limit=limit)
        blocked, _, _ = limiter.check_and_record("key-A", limit=limit)
        assert not blocked

        # key-B should still be free
        allowed, _, _ = limiter.check_and_record("key-B", limit=limit)
        assert allowed


class TestTierLimits:
    def test_free_tier_limit(self):
        assert _limit_for_tier("free") == TIER_LIMITS["free"]

    def test_enterprise_is_unlimited(self):
        assert _limit_for_tier("enterprise") == 0

    def test_unknown_tier_defaults_to_free(self):
        assert _limit_for_tier("unknown_tier") == TIER_LIMITS["free"]
