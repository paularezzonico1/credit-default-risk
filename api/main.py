"""FastAPI scoring service for the credit-default-risk model.

Exposes three endpoints:

* ``POST /predict``     score a borrower (PD, decision, top SHAP factors).
* ``GET  /health``      liveness/readiness; confirms artifacts are loaded.
* ``GET  /model/info``  model governance metadata (version, AUC, threshold...).

The model is loaded once at startup, not per request. Malformed requests are
rejected by pydantic with a 422; unexpected scoring errors are caught and
returned as a structured 500 so nothing fails silently.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api import __version__
from api.config import get_settings
from api.model_service import ModelService
from api.schemas import (
    BorrowerInput,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logger = logging.getLogger("credit_risk_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Credit Default Risk API",
    version=__version__,
    description="Explainable retail-loan default scoring service.",
)

# Single shared service instance; populated at startup.
settings = get_settings()
service = ModelService(
    models_dir=settings.models_dir,
    threshold_override=settings.threshold_override,
)


@app.on_event("startup")
def _load_model() -> None:
    """Load model artifacts once when the process starts."""
    try:
        service.load()
        logger.info("Model loaded: version=%s", service.model_version)
    except FileNotFoundError as exc:
        # Don't crash the process: /health will report unhealthy and surface why.
        logger.error("Model artifacts not loaded: %s", exc)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured 500 for any unexpected error (never fail silently)."""
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error while processing the request.",
                 "error": exc.__class__.__name__},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness check; healthy only when artifacts are loaded."""
    loaded = service.is_loaded
    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        model_version=service.model_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Return model governance metadata."""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return ModelInfoResponse(**service.model_info())


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(borrower: BorrowerInput) -> PredictionResponse:
    """Score a borrower and return PD, decision, and top SHAP risk factors."""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    request_id = str(uuid.uuid4())
    result = service.predict(borrower.dict())
    return PredictionResponse(request_id=request_id, **result)
