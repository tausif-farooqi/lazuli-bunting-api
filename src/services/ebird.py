"""
eBird API service — async network calls with TTL caching.

The ``live_sightings_cache`` caches up to 20 distinct ``days_back`` values
for 10 minutes to avoid hammering the eBird API on every page refresh.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

import httpx
from cachetools import TTLCache
from fastapi import HTTPException

from core.config import settings
from db.supabase_client import get_supabase_client

# Cache up to 20 different days_back values for 10 minutes (600 seconds)
live_sightings_cache: TTLCache = TTLCache(maxsize=20, ttl=600)

_EBIRD_LAZBUN_URL = "https://api.ebird.org/v2/data/obs/US/recent/lazbun"


async def fetch_live_sightings(days_back: int) -> Dict[str, Any]:
    """
    Fetch, aggregate, and return live Lazuli Bunting sightings from the eBird API.
    Results are cached per ``days_back`` value for 10 minutes.
    """
    if days_back in live_sightings_cache:
        print(f"serving days_back={days_back} from cache")
        return live_sightings_cache[days_back]

    params = {"back": days_back, "key": settings.ebird_api_key}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(_EBIRD_LAZBUN_URL, params=params, timeout=10.0)
            response.raise_for_status()
            ebird_data = response.json()
        except Exception as exc:
            print(f"eBird Error: {exc}")
            raise HTTPException(status_code=503, detail="eBird service unavailable")

    if not ebird_data:
        empty_res: Dict[str, Any] = {"message": "No sightings found.", "data": {}}
        live_sightings_cache[days_back] = empty_res
        return empty_res

    # Batch locality metadata lookup
    unique_localities = list({obs["locName"] for obs in ebird_data})
    supabase = get_supabase_client()
    try:
        meta_response = (
            supabase.table("mv_park_metadata")
            .select("locality, state, county")
            .in_("locality", unique_localities)
            .execute()
        )
        meta_lookup = {row["locality"]: row for row in meta_response.data}
    except Exception as exc:
        print(f"Supabase Lookup Error: {exc}")
        meta_lookup = {}

    # Aggregate observations by (state, location, date)
    aggregator: Dict[tuple, Dict[str, Any]] = {}
    for obs in ebird_data:
        loc = obs.get("locName")
        date_raw = obs.get("obsDt", "")
        date_only = date_raw.split(" ")[0] if date_raw else "Unknown"

        meta = meta_lookup.get(loc, {})
        state = meta.get("state") or obs.get("subnational1Code", "").split("-")[-1]
        county = meta.get("county") or "Unknown County"

        key = (state, loc, date_only)
        count = obs.get("howMany", 1)
        sub_id = obs.get("subId")

        if key in aggregator:
            aggregator[key]["count"] += count
            if sub_id not in aggregator[key]["subIds"]:
                aggregator[key]["subIds"].append(sub_id)
        else:
            aggregator[key] = {
                "location": loc,
                "county": county,
                "state": state,
                "count": count,
                "date": date_only,
                "subIds": [sub_id] if sub_id else [],
            }

    # Group by state, sort each state's sightings by date descending
    final_data: Dict[str, list] = defaultdict(list)
    for entry in aggregator.values():
        final_data[entry["state"]].append(entry)
    for state_key in final_data:
        final_data[state_key].sort(key=lambda x: x["date"], reverse=True)

    result: Dict[str, Any] = {
        "metadata": {
            "days_back": days_back,
            "cached": False,
            "generated_at": datetime.now().isoformat(),
        },
        "data": dict(final_data),
    }

    live_sightings_cache[days_back] = result
    return result
