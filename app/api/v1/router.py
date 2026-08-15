"""
app/api/v1/router.py
────────────────────
Aggregates all v1 sub-routers under the /api/v1 prefix.
"""

from fastapi import APIRouter

from app.api.v1 import vision

router = APIRouter(prefix="/api/v1")
router.include_router(vision.router, prefix="/vision")
