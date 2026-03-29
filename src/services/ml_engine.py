"""
ML Engine — XGBoost model loading, spatial filtering, and inference.

The module owns a private ``_state`` dict that is populated at startup
via ``load_artifacts()`` and cleared at shutdown via ``clear_artifacts()``.
The predict endpoint calls ``run_inference()`` which returns plain dicts so
the endpoint layer is responsible for Pydantic schema construction.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from core.config import MODELS_DIR
from features import MODEL_FEATURES, encode_month

EARTH_RADIUS_MILES = 3_958.8

_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifecycle helpers — called by the FastAPI lifespan context manager
# ---------------------------------------------------------------------------

def load_artifacts() -> None:
    """Load XGBoost model, locality profiles, and metadata from disk."""
    print(f"Loading ML artifacts from {MODELS_DIR} …")
    print(f"  Directory exists: {MODELS_DIR.exists()}")
    if MODELS_DIR.exists():
        print(f"  Contents: {[f.name for f in MODELS_DIR.iterdir()]}")

    model_path = MODELS_DIR / "lazuli_bunting_xgboost.json"
    model = xgb.XGBClassifier()
    try:
        model.load_model(str(model_path))
        _state["model"] = model
        print("  XGBoost model loaded.")
    except Exception as exc:
        print(f"  ERROR loading model ({model_path}): {exc}")

    profiles_path = MODELS_DIR / "locality_profiles.parquet"
    try:
        _state["profiles"] = pd.read_parquet(str(profiles_path))
        print(f"  {len(_state['profiles']):,} locality profiles loaded.")
    except Exception as exc:
        print(f"  ERROR loading profiles ({profiles_path}): {exc}")

    meta_path = MODELS_DIR / "metadata.json"
    try:
        _state["meta"] = json.loads(meta_path.read_text())
        print(f"  Metadata loaded (threshold={_state['meta']['optimal_threshold']:.4f}).")
    except Exception as exc:
        _state["meta"] = {"optimal_threshold": 0.5, "features": MODEL_FEATURES}
        print(f"  WARNING: metadata missing — using defaults ({exc})")


def clear_artifacts() -> None:
    _state.clear()


def is_ready() -> bool:
    return "model" in _state and "profiles" in _state


def get_profiles() -> pd.DataFrame | None:
    return _state.get("profiles")


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def haversine_miles(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised great-circle distance in miles (scalar → array)."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def slugify(name: str) -> str:
    """Turn a locality name into a URL-friendly identifier."""
    s = re.sub(r"[^\w\s-]", "", name.lower().strip())
    return re.sub(r"[-\s]+", "-", s).strip("-")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    latitude: float,
    longitude: float,
    month: int,
    radius_miles: float,
) -> list[dict[str, Any]]:
    """
    Spatial filter → feature matrix → XGBoost predict_proba → confidence weighting.

    Returns a list of result dicts ready for the endpoint to map to Pydantic models.
    Each dict contains: location_id, name, region, latitude, longitude,
    base_sightings, distance, total_sightings, reliability_score.
    """
    model: xgb.XGBClassifier | None = _state.get("model")
    profiles: pd.DataFrame | None = _state.get("profiles")

    if model is None or profiles is None:
        raise RuntimeError("Model or locality data has not been loaded yet.")

    if profiles.empty:
        return []

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
    probabilities = model.predict_proba(X)[:, 1]

    # Discount reliability for under-observed locations.
    # Reaches full confidence (~1.0) at 20+ total sightings;
    # a single sighting reduces the score to ~23% of the model output.
    CONFIDENCE_REF = 20
    sightings = nearby["locality_total_sightings"].values.astype(float)
    confidence = np.minimum(1.0, np.log1p(sightings) / np.log1p(CONFIDENCE_REF))

    scores = np.clip(probabilities * confidence * 100, 0, 99).round(2)
    nearby["_score"] = scores

    nearby = nearby[nearby["_score"] >= 25].copy()
    if nearby.empty:
        return []

    nearby.sort_values(
        ["_score", "locality_total_sightings"],
        ascending=[False, False],
        inplace=True,
    )

    results: list[dict[str, Any]] = []
    for _, row in nearby.iterrows():
        results.append({
            "location_id": slugify(row["locality"]),
            "name": row["locality"],
            "region": f"{row['county']}, {row['state']}",
            "latitude": round(float(row["latitude"]), 6),
            "longitude": round(float(row["longitude"]), 6),
            "base_sightings": int(row["locality_total_sightings"]),
            "distance": round(float(row["_dist"]), 1),
            "total_sightings": int(row["locality_total_sightings"]),
            "reliability_score": round(float(row["_score"]), 2),
        })
    return results
