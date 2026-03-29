"""
GET /api/predictions — Lazuli Bunting sighting probability near a coordinate.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import services.ml_engine as ml_engine

router = APIRouter()


class LocationDetail(BaseModel):
    id: str
    name: str
    region: str
    latitude: float
    longitude: float
    baseSightings: int
    habitat: str = "eBird hotspot"


class PredictionResult(BaseModel):
    location: LocationDetail
    distance: float = Field(..., description="Miles from the query point")
    totalSightings: int
    reliabilityScore: float = Field(
        ..., ge=25, le=99,
        description="Probability of sighting expressed as a 25–99 percentage",
    )


@router.get(
    "/predictions",
    response_model=list[PredictionResult],
    summary="Predict Lazuli Bunting sighting probability near a point",
)
async def get_predictions(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    month: int = Query(..., ge=1, le=12),
    radius_miles: float = Query(10.0, gt=0, le=30),
):
    """Find all known eBird localities within *radius_miles* of the query point,
    run the ML model for the requested *month*, and return results ranked by
    predicted sighting probability (highest first).
    """
    if not ml_engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model or locality data has not been loaded yet.",
        )

    raw = ml_engine.run_inference(latitude, longitude, month, radius_miles)

    return [
        PredictionResult(
            location=LocationDetail(
                id=r["location_id"],
                name=r["name"],
                region=r["region"],
                latitude=r["latitude"],
                longitude=r["longitude"],
                baseSightings=r["base_sightings"],
            ),
            distance=r["distance"],
            totalSightings=r["total_sightings"],
            reliabilityScore=r["reliability_score"],
        )
        for r in raw
    ]
