"""
GET /api/live-sightings — recent Lazuli Bunting observations from the eBird API.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from services.ebird import fetch_live_sightings

router = APIRouter()


@router.get("/live-sightings")
async def get_live_sightings(days_back: int = Query(7, ge=1, le=30)):
    """Live sightings feed with async I/O and TTL caching (10-minute window)."""
    return await fetch_live_sightings(days_back)
