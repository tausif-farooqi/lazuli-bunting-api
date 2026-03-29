"""
Lazuli Bunting Sighting Prediction API
=======================================
Entry point: FastAPI app initialisation, CORS middleware, lifespan, and router.
All business logic lives in core/, db/, services/, and api/.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Avoid PermissionError on Windows: ssl module tries to write SSLKEYLOGFILE.
os.environ.pop("SSLKEYLOGFILE", None)

# Use a writable working directory on Windows for libraries that write to cwd.
if os.name == "nt":
    _safe_cwd = os.environ.get("TEMP") or os.path.expanduser("~")
    try:
        os.chdir(_safe_cwd)
    except OSError:
        pass

# Ensure src/ packages (core, db, services, api, features) are importable.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import services.ml_engine as ml_engine
from api.router import router as api_router
from core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_engine.load_artifacts()
    yield
    ml_engine.clear_artifacts()


app = FastAPI(
    lifespan=lifespan,
    title="Lazuli Bunting Prediction API",
    description=(
        "Returns ranked probabilities of observing Lazuli Buntings "
        "at eBird hotspots near a given coordinate for a given month."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
