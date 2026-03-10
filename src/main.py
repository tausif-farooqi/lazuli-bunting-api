import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import xgboost as xgb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS").split(",")

# Global dictionary to hold our loaded model
ml_models = {}

# The optimal threshold discovered during your Precision-Recall evaluation
DEFAULT_THRESHOLD = 0.2295 
#DEFAULT_THRESHOLD = 0.5259

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing ML models...")
    model = xgb.XGBClassifier()
    try:
        model.load_model("models/lazuli_bunting_xgboost.json")
        ml_models["xgb_model"] = model
        print("XGBoost model loaded successfully.")
    except Exception as e:
        print(f"Critical Error: Could not load model. {e}")
    
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="Lazuli Bunting Hotspot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class LocationInput(BaseModel):
    latitude: float = Field(..., description="Latitude")
    longitude: float = Field(..., description="Longitude")
    month: int = Field(..., ge=1, le=12, description="Month of the year (1-12)")

class PredictionResponse(BaseModel):
    latitude: float
    longitude: float
    hotspot_probability: float
    is_hotspot: bool  # Added to explicitly flag if it crosses the threshold

# ---------------------------------------------------------
# Inference Endpoint
# ---------------------------------------------------------
@app.post("/predict/hotspots", response_model=List[PredictionResponse])
async def predict_hotspots(
    locations: List[LocationInput],
    threshold: float = Query(DEFAULT_THRESHOLD, description="Probability cutoff for a positive sighting")
):
    model = ml_models.get("xgb_model")
    if not model:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    
    if not locations:
        return []

    df_input = pd.DataFrame([loc.model_dump() for loc in locations])
    
    try:
        # Extract the probability of class 1 (Presence)
        probabilities = model.predict_proba(df_input)[:, 1]
        
        results = [
            PredictionResponse(
                latitude=loc.latitude,
                longitude=loc.longitude,
                hotspot_probability=float(prob),
                is_hotspot=bool(prob >= threshold) # Apply the threshold here
            )
            for loc, prob in zip(locations, probabilities)
        ]
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)