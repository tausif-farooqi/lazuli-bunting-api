"""
GET /api/stats/topparks — top 20 parks by Lazuli Bunting sightings.
Results are cached in-process for 24 hours since this data rarely changes.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException

from db.supabase_client import get_supabase_client

router = APIRouter()

_TOP_PARKS_TTL_SECONDS = 24 * 60 * 60
_cache: dict[str, Any] = {"data": None, "fetched_at": 0.0}


@router.get("/stats/topparks")
def get_top_parks() -> list[dict[str, Any]]:
    """Top 20 parks for Lazuli Bunting sightings. Cached for 24 hours."""
    now = time.time()
    if _cache["data"] is not None and (now - _cache["fetched_at"]) < _TOP_PARKS_TTL_SECONDS:
        return _cache["data"]

    supabase = get_supabase_client()
    try:
        response = supabase.rpc("get_top_20_parks").execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")

    rows = response.data or []
    _cache["data"] = rows
    _cache["fetched_at"] = now
    return rows
