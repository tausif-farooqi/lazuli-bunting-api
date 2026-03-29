"""
API v1 router — aggregates all endpoint routers under the /api prefix.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.api_v1.endpoints import live_sightings, predict, stats, top_parks

router = APIRouter()

router.include_router(predict.router)
router.include_router(stats.router)
router.include_router(top_parks.router)
router.include_router(live_sightings.router)
