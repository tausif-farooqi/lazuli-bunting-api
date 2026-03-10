"""
Feature engineering for the Lazuli Bunting sighting prediction model.

Shared between the training pipeline (train.py) and the inference API (main.py).
The feature set is designed around three signal axes:
  - Spatial:  raw lat/lon encodes the species' western‑NA range.
  - Temporal: sin/cos month encoding captures seasonal migration without
              an artificial December→January discontinuity.
  - Context:  locality / county / state aggregate statistics act as priors
              for observation density and birder effort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Authoritative ordered list of features the model expects.
MODEL_FEATURES: list[str] = [
    "latitude",
    "longitude",
    "month_sin",
    "month_cos",
    "lat_x_month_sin",
    "lat_x_month_cos",
    "lon_x_month_sin",
    "lon_x_month_cos",
    "months_from_peak",
    "season_margin",
    "locality_total_sightings",
    "locality_mean_count",
    "years_observed",
    "n_months_with_sightings",
    "sighting_frequency",
    "county_sighting_density",
    "state_sighting_density",
]


def encode_month(month: int | float | np.ndarray | pd.Series) -> tuple:
    """Return (sin, cos) cyclical encoding for a month value (1–12)."""
    angle = 2 * np.pi * np.asarray(month, dtype=float) / 12.0
    return np.sin(angle), np.cos(angle)


# ---------------------------------------------------------------------------
# Locality profile construction (used by training and optionally by the API
# to refresh cached profiles).
# ---------------------------------------------------------------------------

def build_locality_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw observations into one row per locality.

    Returns a DataFrame with columns suitable for direct use as model features
    at inference time (plus identifiers like locality, county, state).
    """
    df = df.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["year"] = df["observation_date"].dt.year
    df["month"] = df["observation_date"].dt.month

    locality_agg = (
        df.groupby("locality")
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            county=("county", "first"),
            state=("state", "first"),
            locality_total_sightings=("observed_count", "sum"),
            locality_mean_count=("observed_count", "mean"),
            years_observed=("year", "nunique"),
            n_months_with_sightings=("month", "nunique"),
            _obs_count=("observed_count", "count"),
        )
        .reset_index()
    )
    locality_agg["sighting_frequency"] = (
        locality_agg["_obs_count"] / locality_agg["years_observed"].clip(lower=1)
    )
    locality_agg.drop(columns=["_obs_count"], inplace=True)

    county_density = (
        df.groupby("county")
        .agg(
            _total=("observed_count", "sum"),
            _n_loc=("locality", "nunique"),
        )
        .reset_index()
    )
    county_density["county_sighting_density"] = county_density["_total"] / county_density["_n_loc"]

    state_density = (
        df.groupby("state")
        .agg(
            _total=("observed_count", "sum"),
            _n_loc=("locality", "nunique"),
        )
        .reset_index()
    )
    state_density["state_sighting_density"] = state_density["_total"] / state_density["_n_loc"]

    # Peak sighting month per locality (month with highest total observed_count)
    monthly_totals = df.groupby(["locality", "month"])["observed_count"].sum().reset_index()
    peak_months = (
        monthly_totals
        .sort_values("observed_count", ascending=False)
        .drop_duplicates("locality")[["locality", "month"]]
        .rename(columns={"month": "peak_month"})
    )

    locality_agg = locality_agg.merge(peak_months, on="locality", how="left")
    locality_agg["peak_month"] = locality_agg["peak_month"].fillna(6).astype(int)

    locality_agg = locality_agg.merge(
        county_density[["county", "county_sighting_density"]], on="county", how="left",
    )
    locality_agg = locality_agg.merge(
        state_density[["state", "state_sighting_density"]], on="state", how="left",
    )

    return locality_agg


def build_locality_month_presence(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of (locality, month) pairs that have ≥1 sighting."""
    df = df.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["month"] = df["observation_date"].dt.month
    return df.groupby(["locality", "month"]).size().reset_index(name="_count")[["locality", "month"]]


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------

def _circular_month_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest distance between two months on a 12‑month circle (0–6)."""
    diff = np.abs(a - b)
    return np.minimum(diff, 12 - diff)


def build_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create a labelled dataset from raw observations.

    For every known locality, create 12 rows (one per month).
    - ``label = 1``  if at least one sighting was recorded in that month.
    - ``label = 0``  otherwise (pseudo‑absence).

    Key derived features:
    - ``months_from_peak``: circular distance to the locality's peak sighting
      month — the single strongest seasonal discriminator.
    - ``lon_x_month_*``: longitude × month interactions to capture how
      coastal vs. interior timing differs.
    """
    profiles = build_locality_profiles(df)
    presence = build_locality_month_presence(df)

    # Observation strength per (locality, month) — used for sample weighting
    tmp = df.copy()
    tmp["observation_date"] = pd.to_datetime(tmp["observation_date"])
    tmp["month"] = tmp["observation_date"].dt.month
    obs_strength = (
        tmp.groupby(["locality", "month"])["observed_count"]
        .sum()
        .reset_index(name="_obs_total")
    )

    localities = profiles["locality"].unique()
    grid = pd.DataFrame(
        [(loc, m) for loc in localities for m in range(1, 13)],
        columns=["locality", "month"],
    )

    presence["_present"] = 1
    grid = grid.merge(presence, on=["locality", "month"], how="left")
    grid["label"] = grid["_present"].fillna(0).astype(int)
    grid.drop(columns=["_present"], inplace=True)

    grid = grid.merge(obs_strength, on=["locality", "month"], how="left")
    grid["_obs_total"] = grid["_obs_total"].fillna(0)
    grid["sample_weight"] = np.where(
        grid["label"] == 1,
        np.log1p(grid["_obs_total"]),
        1.0,
    )
    grid.drop(columns=["_obs_total"], inplace=True)

    grid = grid.merge(profiles, on="locality", how="left")

    # Temporal encodings
    grid["month_sin"], grid["month_cos"] = encode_month(grid["month"])
    grid["lat_x_month_sin"] = grid["latitude"] * grid["month_sin"]
    grid["lat_x_month_cos"] = grid["latitude"] * grid["month_cos"]
    grid["lon_x_month_sin"] = grid["longitude"] * grid["month_sin"]
    grid["lon_x_month_cos"] = grid["longitude"] * grid["month_cos"]

    # Seasonal proximity to this locality's peak month
    grid["months_from_peak"] = _circular_month_distance(
        grid["month"].values, grid["peak_month"].values,
    )
    # Positive when inside the expected season, negative when outside.
    # A wide-season locality (n_months=8) tolerates months_from_peak=3;
    # a narrow one (n_months=2) does not.
    grid["season_margin"] = grid["n_months_with_sightings"] / 2 - grid["months_from_peak"]

    return grid


def extract_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the MODEL_FEATURES columns from *df*."""
    return df[MODEL_FEATURES].copy()
