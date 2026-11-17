"""Prediction logging for the serving API.

Every /predict call appends one JSON line here with the input length, the
predicted class, and confidence. This is deliberately simple (no request
text is stored, to avoid persisting PHI-shaped data in a log file) but it's
enough for monitoring/drift.py and the Streamlit ops dashboard to reconstruct
prediction volume, class-mix drift, and confidence trends over time.
"""
from __future__ import annotations

from clinical_mlops.config import settings
from clinical_mlops.utils.logging_config import append_jsonl, read_jsonl


def _predictions_log_path():
    # Resolved lazily (not at import time) so tests can point `settings` at
    # a throwaway directory and have this immediately follow it.
    return settings.log_path / "predictions.jsonl"


def log_prediction(doc_type: str, confidence: float, model_version: str, text_length: int) -> None:
    append_jsonl(_predictions_log_path(), {
        "doc_type": doc_type,
        "confidence": confidence,
        "model_version": model_version,
        "text_length": text_length,
    })


def load_predictions() -> list[dict]:
    return read_jsonl(_predictions_log_path())
