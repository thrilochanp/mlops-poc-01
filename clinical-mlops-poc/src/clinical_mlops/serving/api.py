"""
Inference API for the clinical document-type classifier.

Loads the current Production model from the MLflow Model Registry once at
startup, serves /predict, and logs each prediction for the monitoring
dashboard / drift check. This is the piece Engineering DevOps would put
behind a load balancer / autoscaler in a real deployment (see the docker
compose / CI setup for how this POC packages and tests that boundary).

Run with: uvicorn clinical_mlops.serving.api:app --reload
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from clinical_mlops.config import settings
from clinical_mlops.models import registry
from clinical_mlops.monitoring.logger import log_prediction
from clinical_mlops.serving.schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse
from clinical_mlops.utils.logging_config import get_logger

logger = get_logger(__name__)

_state: dict = {"model": None, "info": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["info"] = registry.get_production_version_info()
        if _state["info"] is not None:
            _state["model"] = registry.load_production_model()
            logger.info("Loaded model %s v%s (%s)", settings.model_name,
                        _state["info"]["version"], settings.model_stage)
        else:
            logger.warning("No Production model found yet — /predict will 503 until one is trained.")
    except Exception:
        logger.exception("Failed to load model at startup; /predict will 503 until reload.")
    yield


app = FastAPI(title="Clinical Document Classifier API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_state["model"] is not None)


@app.post("/reload-model", response_model=ModelInfoResponse)
def reload_model() -> ModelInfoResponse:
    """Hot-reload the Production model without restarting the process —
    call this right after promoting a new model version."""
    info = registry.get_production_version_info()
    if info is None:
        raise HTTPException(status_code=404, detail="No Production model registered.")
    _state["model"] = registry.load_production_model()
    _state["info"] = info
    return ModelInfoResponse(name=info["name"], version=str(info["version"]),
                              stage=info["stage"], metrics=info["metrics"], tags=info["tags"])


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    info = _state["info"]
    if info is None:
        raise HTTPException(status_code=404, detail="No Production model registered.")
    return ModelInfoResponse(name=info["name"], version=str(info["version"]),
                              stage=info["stage"], metrics=info["metrics"], tags=info["tags"])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _state["model"]
    info = _state["info"]
    if model is None or info is None:
        raise HTTPException(status_code=503, detail="No Production model loaded. Train one first.")

    pred = model.predict([req.text])[0]
    proba = model.predict_proba([req.text])[0]
    confidence = float(max(proba))

    log_prediction(doc_type=pred, confidence=confidence,
                    model_version=str(info["version"]), text_length=len(req.text))

    return PredictResponse(doc_type=pred, confidence=confidence,
                            model_name=settings.model_name, model_version=str(info["version"]))
