from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="OCR'd / normalized clinical note text.")


class PredictResponse(BaseModel):
    doc_type: str
    confidence: float
    model_name: str
    model_version: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    stage: str
    metrics: dict
    tags: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
