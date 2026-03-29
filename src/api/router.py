"""
Root API router — mounts the versioned API under /api and exposes /health.
"""

from __future__ import annotations

from fastapi import APIRouter

import services.ml_engine as ml_engine
from api.api_v1.router import router as api_v1_router

router = APIRouter()

router.include_router(api_v1_router, prefix="/api")


@router.get("/health", include_in_schema=False)
async def health():
    profiles = ml_engine.get_profiles()
    return {
        "status": "ok",
        "model_loaded": ml_engine.is_ready(),
        "localities_loaded": profiles is not None,
        "n_localities": len(profiles) if profiles is not None else 0,
    }
