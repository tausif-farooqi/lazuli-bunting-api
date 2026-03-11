"""
Lazuli Bunting Sighting Prediction API
======================================
Production FastAPI service that returns ranked probabilities of observing
Lazuli Buntings at locations near a given coordinate for a given month.

Startup loads three artifacts produced by ``train.py``:
  - models/lazuli_bunting_xgboost.json   (XGBoost model)
  - models/locality_profiles.parquet     (pre‑computed per‑locality features)
  - models/metadata.json                 (feature list + optimal threshold)

All spatial filtering is done in‑memory via vectorised Haversine — no
Supabase round‑trip on the hot path.
"""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client

# Avoid PermissionError on Windows: Python's ssl module (used by httpx/Supabase)
# sets keylog_filename when SSLKEYLOGFILE is set, which can point at an unwritable path.
os.environ.pop("SSLKEYLOGFILE", None)

# Optional: use a writable cwd on Windows for libs that write to the current dir.
if os.name == "nt":
    _safe_cwd = os.environ.get("TEMP") or os.path.expanduser("~")
    try:
        os.chdir(_safe_cwd)
    except OSError:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from features import MODEL_FEATURES, encode_month

load_dotenv(ROOT_DIR / ".env")

MODELS_DIR = ROOT_DIR / "models"
EARTH_RADIUS_MILES = 3_958.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    return create_client(url, key)

def haversine_miles(
    lat1: float, lon1: float,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised great‑circle distance in miles (scalar → array)."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def slugify(name: str) -> str:
    """Turn a locality name into a URL‑friendly identifier."""
    s = re.sub(r"[^\w\s-]", "", name.lower().strip())
    return re.sub(r"[-\s]+", "-", s).strip("-")


# ---------------------------------------------------------------------------
# Application state — populated at startup, cleared at shutdown.
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading ML artifacts from {MODELS_DIR} …")
    print(f"  Directory exists: {MODELS_DIR.exists()}")
    if MODELS_DIR.exists():
        print(f"  Contents: {[f.name for f in MODELS_DIR.iterdir()]}")

    # Model
    model_path = MODELS_DIR / "lazuli_bunting_xgboost.json"
    model = xgb.XGBClassifier()
    try:
        model.load_model(str(model_path))
        _state["model"] = model
        print("  XGBoost model loaded.")
    except Exception as exc:
        print(f"  ERROR loading model ({model_path}): {exc}")

    # Locality profiles
    profiles_path = MODELS_DIR / "locality_profiles.parquet"
    try:
        _state["profiles"] = pd.read_parquet(str(profiles_path))
        print(f"  {len(_state['profiles']):,} locality profiles loaded.")
    except Exception as exc:
        print(f"  ERROR loading profiles ({profiles_path}): {exc}")

    # Metadata
    meta_path = MODELS_DIR / "metadata.json"
    try:
        _state["meta"] = json.loads(meta_path.read_text())
        print(f"  Metadata loaded (threshold={_state['meta']['optimal_threshold']:.4f}).")
    except Exception as exc:
        _state["meta"] = {"optimal_threshold": 0.5, "features": MODEL_FEATURES}
        print(f"  WARNING: metadata missing — using defaults ({exc})")

    yield

    _state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    lifespan=lifespan,
    title="Lazuli Bunting Prediction API",
    description=(
        "Returns ranked probabilities of observing Lazuli Buntings "
        "at eBird hotspots near a given coordinate for a given month."
    ),
    version="2.0.0",
)

allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/predictions",
    response_model=list[PredictionResult],
    summary="Predict Lazuli Bunting sighting probability near a point",
)
async def get_predictions(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    month: int = Query(..., ge=1, le=12),
    radius_miles: float = Query(10.0, gt=0, le=30),
):
    """Find all known eBird localities within *radius_miles* of the query
    point, run the ML model for the requested *month*, and return results
    ranked by predicted sighting probability (highest first).
    """
    model: xgb.XGBClassifier | None = _state.get("model")
    profiles: pd.DataFrame | None = _state.get("profiles")

    if model is None or profiles is None:
        raise HTTPException(
            status_code=503,
            detail="Model or locality data has not been loaded yet.",
        )

    if profiles.empty:
        return []

    # -- Spatial filter (vectorised Haversine) --
    distances = haversine_miles(
        latitude, longitude,
        profiles["latitude"].values,
        profiles["longitude"].values,
    )
    mask = distances <= radius_miles
    nearby = profiles.loc[mask].copy()
    nearby["_dist"] = distances[mask]

    if nearby.empty:
        return []

    # -- Build feature matrix --
    m_sin, m_cos = encode_month(month)
    nearby["month_sin"] = float(m_sin)
    nearby["month_cos"] = float(m_cos)
    nearby["lat_x_month_sin"] = nearby["latitude"] * float(m_sin)
    nearby["lat_x_month_cos"] = nearby["latitude"] * float(m_cos)
    nearby["lon_x_month_sin"] = nearby["longitude"] * float(m_sin)
    nearby["lon_x_month_cos"] = nearby["longitude"] * float(m_cos)
    diff = np.abs(month - nearby["peak_month"].values)
    nearby["months_from_peak"] = np.minimum(diff, 12 - diff)
    nearby["season_margin"] = nearby["n_months_with_sightings"] / 2 - nearby["months_from_peak"]

    X = nearby[MODEL_FEATURES].fillna(0)

    # -- Inference --
    probabilities = model.predict_proba(X)[:, 1]

    # Discount reliability for under-observed locations.
    # Reaches full confidence (~1.0) at 20+ total sightings;
    # a single sighting reduces the score to ~23% of the model output.
    CONFIDENCE_REF = 20
    sightings = nearby["locality_total_sightings"].values.astype(float)
    confidence = np.minimum(1.0, np.log1p(sightings) / np.log1p(CONFIDENCE_REF))

    scores = np.clip(probabilities * confidence * 100, 0, 99).round(2)
    nearby["_score"] = scores

    # Drop locations below the minimum reliability threshold
    nearby = nearby[nearby["_score"] >= 25].copy()
    if nearby.empty:
        return []

    nearby.sort_values(
        ["_score", "locality_total_sightings"],
        ascending=[False, False],
        inplace=True,
    )

    # -- Format response --
    results: list[PredictionResult] = []
    for _, row in nearby.iterrows():
        results.append(
            PredictionResult(
                location=LocationDetail(
                    id=slugify(row["locality"]),
                    name=row["locality"],
                    region=f"{row['county']}, {row['state']}",
                    latitude=round(float(row["latitude"]), 6),
                    longitude=round(float(row["longitude"]), 6),
                    baseSightings=int(row["locality_total_sightings"]),
                ),
                distance=round(float(row["_dist"]), 1),
                totalSightings=int(row["locality_total_sightings"]),
                reliabilityScore=round(float(row["_score"]), 2),
            )
        )

    return results

@app.get("/api/stats/annualsummary")
def get_annual_summary() -> list[dict[str, Any]]:
    """Returns yearly sighting totals. No parameters. Calls get_annual_sightings_summary()."""
    supabase = _get_supabase()
    try:
        response = supabase.rpc("get_annual_sightings_summary").execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    rows = response.data or []
    return [{"obs_year": row.get("obs_year"), "total_sightings": row.get("total_sightings")} for row in rows]


@app.get("/api/stats/state")
def get_state_stats(year: int = Query(..., description="Year (e.g. 2024)")) -> list[dict[str, Any]]:
    """Returns state-level sighting counts for the given year. Calls get_state_stats_by_year()."""
    supabase = _get_supabase()
    try:
        response = supabase.rpc("get_state_stats_by_year", {"target_year": year}).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    rows = response.data or []
    return [{"state": row.get("state"), "total_sightings": row.get("total_sightings")} for row in rows]


@app.get("/api/stats/counties")
def get_county_stats(
    state: str = Query(..., description="State code or name (e.g. CA)"),
    year: int = Query(..., description="Year (e.g. 2024)"),
) -> list[dict[str, Any]]:
    """Returns county-level sighting counts for the given state and year. Calls get_county_stats_by_state_year()."""
    supabase = _get_supabase()
    try:
        response = supabase.rpc(
            "get_county_stats_by_state_year",
            {"target_state": state, "target_year": year},
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    rows = response.data or []
    return [{"county": row.get("county"), "total_sightings": row.get("total_sightings")} for row in rows]


@app.get("/health", include_in_schema=False)
async def health():
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
        "localities_loaded": "profiles" in _state,
        "n_localities": len(_state.get("profiles", [])),
    }


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
