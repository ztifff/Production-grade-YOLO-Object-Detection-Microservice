"""
app/middleware/auth.py
─────────────────────
FastAPI security dependency for API Key authentication.

Every protected route declares::

    from app.middleware.auth import require_api_key, TenantContext
    
    @router.post("/detect")
    async def detect(tenant: TenantContext = Security(require_api_key)):
        ...

The dependency resolves the raw ``X-API-Key`` header into a
:class:`TenantContext` dataclass, which the route handler and usage logger
both consume without re-parsing the key map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Security scheme (shows in OpenAPI /docs) ──────────────────────────────────
_api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="APIKey",
    description="Pass your API key in the `X-API-Key` request header.",
    auto_error=False,  # We raise custom HTTP errors below
)

# ── Tenant data transfer object ───────────────────────────────────────────────
TierType = Literal["free", "premium", "enterprise"]


@dataclass(frozen=True)
class TenantContext:
    """Resolved tenant metadata attached to every authenticated request."""

    client_id: str
    tier: TierType
    api_key_prefix: str  # First 8 chars — safe to log, never the full key


# ── Dependency ────────────────────────────────────────────────────────────────

async def require_api_key(
    raw_key: str | None = Security(_api_key_header),
) -> TenantContext:
    """
    FastAPI security dependency.

    Returns a :class:`TenantContext` if the key is valid.

    Raises
    ------
    HTTP 401
        When no ``X-API-Key`` header is present.
    HTTP 403
        When the header is present but the key is not recognised.
    """
    if not raw_key:
        logger.warning("Request rejected — missing X-API-Key header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_api_key",
                "message": "An API key is required. Pass it in the X-API-Key header.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_map = settings.api_keys
    tenant_meta = key_map.get(raw_key)

    if tenant_meta is None:
        # Log only a safe prefix — never the full key
        logger.warning(
            "Request rejected — invalid API key",
            extra={"key_prefix": raw_key[:8] + "..."},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "invalid_api_key",
                "message": "The provided API key is not valid or has been revoked.",
            },
        )

    return TenantContext(
        client_id=tenant_meta.get("client_id", "unknown"),
        tier=tenant_meta.get("tier", "free"),
        api_key_prefix=raw_key[:8] + "...",
    )
