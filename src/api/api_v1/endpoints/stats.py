"""
Stats endpoints — annual summary, state-level, and county-level sighting counts.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from db.supabase_client import get_supabase_client

router = APIRouter(prefix="/stats")


@router.get("/annualsummary")
def get_annual_summary() -> list[dict[str, Any]]:
    """Yearly sighting totals across all years in the database."""
    supabase = get_supabase_client()
    try:
        response = supabase.rpc("get_annual_sightings_summary").execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")
    rows = response.data or []
    return [{"obs_year": row.get("obs_year"), "total_sightings": row.get("total_sightings")} for row in rows]


@router.get("/state")
def get_state_stats(
    year: int = Query(..., description="Year (e.g. 2024)"),
) -> list[dict[str, Any]]:
    """State-level sighting counts for a given year."""
    supabase = get_supabase_client()
    try:
        response = supabase.rpc("get_state_stats_by_year", {"target_year": year}).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")
    rows = response.data or []
    return [{"state": row.get("state"), "total_sightings": row.get("total_sightings")} for row in rows]


@router.get("/counties")
def get_county_stats(
    state: str = Query(..., description="State code or name (e.g. CA)"),
    year: int = Query(..., description="Year (e.g. 2024)"),
) -> list[dict[str, Any]]:
    """County-level sighting counts for a given state and year."""
    supabase = get_supabase_client()
    try:
        response = supabase.rpc(
            "get_county_stats_by_state_year",
            {"target_state": state, "target_year": year},
        ).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")
    rows = response.data or []
    return [{"county": row.get("county"), "total_sightings": row.get("total_sightings")} for row in rows]
