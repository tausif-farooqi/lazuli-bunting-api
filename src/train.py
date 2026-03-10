"""
Lazuli Bunting Sighting Model — Training Pipeline
==================================================
Loads eBird observation data from Supabase, engineers features, trains an
XGBoost binary classifier, evaluates via stratified k‑fold CV, and writes
artifacts to ``models/``.

Usage
-----
    cd <project‑root>
    python src/train.py

Environment variables (from .env):
    SUPABASE_URL
    SUPABASE_SERVICE_KEY
    SUPABASE_TABLE_NAME  (optional, default: "lazuli_bunting_sightings")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

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

from features import (
    MODEL_FEATURES,
    build_locality_profiles,
    build_training_data,
    extract_feature_matrix,
)

load_dotenv(ROOT_DIR / ".env")

MODELS_DIR = ROOT_DIR / "models"
TABLE_NAME = os.environ.get("SUPABASE_TABLE_NAME", "lazuli_bunting_sightings")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_from_supabase() -> pd.DataFrame:
    """Fetch the full observation table from Supabase via PostgREST.

    Uses **keyset pagination** (``order=id.asc`` + ``id=gt.<last_id>``)
    instead of OFFSET pagination.  Large offsets cause PostgREST to
    sequentially skip rows, which triggers 500 errors on big tables.
    Keyset pagination is O(log n) at every page.
    """
    import httpx

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }

    all_rows: list[dict] = []
    page_size = 1000          # Supabase default server-side max is 1 000
    last_id: int | None = None
    max_retries = 3

    with httpx.Client(timeout=120) as client:
        while True:
            params: dict = {
                "select": "*",
                "order": "id.asc",
                "limit": page_size,
            }
            if last_id is not None:
                params["id"] = f"gt.{last_id}"

            for attempt in range(1, max_retries + 1):
                resp = client.get(
                    f"{url}/rest/v1/{TABLE_NAME}",
                    headers=headers,
                    params=params,
                )
                if resp.status_code < 500:
                    break
                wait = 2 ** attempt
                print(f"  Server error {resp.status_code} — retrying in {wait}s (attempt {attempt}/{max_retries})")
                import time; time.sleep(wait)
            resp.raise_for_status()

            rows = resp.json()
            if not rows:
                break

            all_rows.extend(rows)
            last_id = rows[-1]["id"]

            if len(all_rows) % 50_000 < page_size:
                print(f"  … {len(all_rows):,} rows fetched")

            if len(rows) < page_size:
                break

    print(f"Loaded {len(all_rows):,} observations from Supabase ({TABLE_NAME})")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _fold_months_from_peak(
    localities: np.ndarray,
    months: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
) -> np.ndarray:
    """Recompute months_from_peak using only training‑fold positives.

    For each locality, peak_month = median of the months that are positive
    in the training fold.  Localities with zero training positives (their
    single positive month fell into this validation fold) get the median
    peak of all training‑fold positives — a location‑agnostic prior that
    avoids leaking any validation‑side information.
    """
    pos_mask = labels[train_idx] == 1
    train_pos_months = months[train_idx][pos_mask]

    # Location-agnostic fallback: the global median positive month from
    # the training fold.  Much better than a hard-coded 6 and leak-free.
    global_median_peak = int(round(float(np.median(train_pos_months)))) if len(train_pos_months) else 6

    peak_df = pd.DataFrame({
        "locality": localities[train_idx][pos_mask],
        "month": train_pos_months,
    })

    if peak_df.empty:
        fold_peak_map = pd.Series(dtype=int)
    else:
        fold_peak_map = peak_df.groupby("locality")["month"].median().round().astype(int)

    mapped = pd.Series(localities).map(fold_peak_map).fillna(global_median_peak).astype(int).values
    diff = np.abs(months - mapped)
    return np.minimum(diff, 12 - diff)


def train_model(df: pd.DataFrame) -> tuple[xgb.XGBClassifier, pd.DataFrame, float]:
    """Train XGBoost binary classifier and return (model, training_df, threshold).

    ``months_from_peak`` is recomputed per CV fold from training labels only,
    so the evaluation metrics are honest (no target leakage).  Localities
    whose only positive month falls in the validation fold use the full‑data
    peak as a fallback — acceptable because with a single observation the
    information content is minimal.

    Predictions from all five folds are pooled for a stable classification
    report (5,844 evaluated samples instead of ~1,168).
    """

    training = build_training_data(df)
    X_base = extract_feature_matrix(training)
    y = training["label"]

    all_localities = training["locality"].values
    all_months = training["month"].values
    weights = training["sample_weight"].values
    y_arr = y.values

    pos = int(y.sum())
    total = len(y)
    print(f"Training set: {total:,} rows  |  {pos:,} positive ({pos / total:.1%})")
    print(f"Features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")

    scale_pos_weight = float((y == 0).sum()) / max((y == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.05,
        min_child_weight=10,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.7,
        max_delta_step=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
    )

    # --- Stratified 5‑fold CV (months_from_peak recomputed per fold) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv: dict[str, list[float]] = {"log_loss": [], "roc_auc": [], "avg_precision": []}

    pooled_y: list[int] = []
    pooled_proba: list[float] = []

    print("\nCross‑validation (months_from_peak recomputed per fold to avoid leakage)")
    print("-" * 70)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_base, y), 1):
        fold_mfp = _fold_months_from_peak(
            all_localities, all_months, y_arr, train_idx,
        )

        X_tr = X_base.iloc[train_idx].copy()
        X_va = X_base.iloc[val_idx].copy()
        X_tr["months_from_peak"] = fold_mfp[train_idx]
        X_va["months_from_peak"] = fold_mfp[val_idx]
        X_tr["season_margin"] = X_tr["n_months_with_sightings"] / 2 - X_tr["months_from_peak"]
        X_va["season_margin"] = X_va["n_months_with_sightings"] / 2 - X_va["months_from_peak"]

        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        w_tr = weights[train_idx]

        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)

        y_proba = model.predict_proba(X_va)[:, 1]
        ll = log_loss(y_va, y_proba)
        auc = roc_auc_score(y_va, y_proba)
        ap = average_precision_score(y_va, y_proba)

        cv["log_loss"].append(ll)
        cv["roc_auc"].append(auc)
        cv["avg_precision"].append(ap)

        pooled_y.extend(y_va.values)
        pooled_proba.extend(y_proba)

        print(f"  Fold {fold}:  log_loss={ll:.4f}  ROC‑AUC={auc:.4f}  AP={ap:.4f}")

    print()
    for metric, scores in cv.items():
        print(f"  {metric:16s}  {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # --- Pooled evaluation across ALL folds (much more stable than single fold) ---
    pooled_y_arr = np.array(pooled_y)
    pooled_proba_arr = np.array(pooled_proba)

    precision, recall, thresholds = precision_recall_curve(pooled_y_arr, pooled_proba_arr)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    optimal_threshold = float(thresholds[np.argmax(f1)])
    print(f"\nOptimal threshold (max F1, pooled): {optimal_threshold:.4f}")

    pooled_auc = roc_auc_score(pooled_y_arr, pooled_proba_arr)
    print(f"Pooled ROC‑AUC: {pooled_auc:.4f}")

    y_pred = (pooled_proba_arr >= optimal_threshold).astype(int)
    print(f"\nClassification report (all 5 folds pooled, {total:,} samples):")
    print(classification_report(pooled_y_arr, y_pred, target_names=["absent", "present"]))

    # --- Retrain on full data (full‑data peak_month is valid for production) ---
    print("Re‑training on full dataset …")
    model.set_params(early_stopping_rounds=None)
    model.fit(X_base, y, sample_weight=weights, verbose=False)

    # Feature importances
    print("\nFeature importances:")
    for feat, imp in sorted(
        zip(MODEL_FEATURES, model.feature_importances_), key=lambda x: -x[1],
    ):
        print(f"  {feat:30s}  {imp:.4f}")

    return model, training, optimal_threshold


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    model: xgb.XGBClassifier,
    raw_df: pd.DataFrame,
    optimal_threshold: float,
) -> None:
    """Write model, locality profiles, and metadata to ``models/``."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "lazuli_bunting_xgboost.json"
    model.save_model(str(model_path))
    print(f"  Model          → {model_path}")

    profiles = build_locality_profiles(raw_df)
    profiles_path = MODELS_DIR / "locality_profiles.parquet"
    profiles.to_parquet(str(profiles_path), index=False)
    print(f"  Profiles ({len(profiles):,}) → {profiles_path}")

    meta = {
        "features": MODEL_FEATURES,
        "optimal_threshold": optimal_threshold,
        "n_localities": len(profiles),
    }
    meta_path = MODELS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata       → {meta_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Lazuli Bunting Sighting Model — Training Pipeline")
    print("=" * 60)

    raw = load_data_from_supabase()
    model, _, threshold = train_model(raw)

    print("\nSaving artifacts …")
    save_artifacts(model, raw, threshold)

    print("\nDone.")
