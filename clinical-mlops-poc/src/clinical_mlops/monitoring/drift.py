"""
Lightweight production monitoring: compare the class mix and text-length
distribution of recent /predict traffic against the training-time baseline
that models/train.py stored, using Population Stability Index (PSI) — a
standard, cheap drift metric that doesn't need a reference dataset shipped
alongside the model.

PSI interpretation (industry rule of thumb):
  < 0.1            no significant shift
  0.1 - 0.25       moderate shift, worth a look
  > 0.25           significant shift, investigate / consider retraining
"""
from __future__ import annotations

import json
import math

from clinical_mlops.config import settings
from clinical_mlops.monitoring.logger import load_predictions


def _baseline_path():
    # Resolved lazily so tests pointing `settings` at a tmp dir take effect.
    return settings.log_path / "training_baseline.json"


PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def _psi(expected: dict[str, float], actual: dict[str, float]) -> float:
    categories = set(expected) | set(actual)
    psi = 0.0
    for cat in categories:
        e = max(expected.get(cat, 0.0001), 0.0001)
        a = max(actual.get(cat, 0.0001), 0.0001)
        psi += (a - e) * math.log(a / e)
    return psi


def compute_drift(min_predictions: int = 10) -> dict:
    predictions = load_predictions()
    baseline_path = _baseline_path()
    if not baseline_path.exists():
        return {"status": "no_baseline", "message": "Train a model first to establish a baseline."}
    if len(predictions) < min_predictions:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(predictions)} predictions logged, need >= {min_predictions}.",
            "n_predictions": len(predictions),
        }

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_dist_raw = baseline["class_distribution"]
    baseline_total = sum(baseline_dist_raw.values())
    baseline_dist = {k: v / baseline_total for k, v in baseline_dist_raw.items()}

    actual_counts: dict[str, int] = {}
    for p in predictions:
        actual_counts[p["doc_type"]] = actual_counts.get(p["doc_type"], 0) + 1
    actual_total = sum(actual_counts.values())
    actual_dist = {k: v / actual_total for k, v in actual_counts.items()}

    class_psi = _psi(baseline_dist, actual_dist)

    avg_len_baseline = baseline["avg_text_length"]
    avg_len_actual = sum(p["text_length"] for p in predictions) / len(predictions)
    len_pct_change = (avg_len_actual - avg_len_baseline) / avg_len_baseline if avg_len_baseline else 0.0

    if class_psi >= PSI_SIGNIFICANT:
        severity = "significant"
    elif class_psi >= PSI_MODERATE:
        severity = "moderate"
    else:
        severity = "none"

    return {
        "status": "ok",
        "n_predictions": len(predictions),
        "class_distribution_psi": round(class_psi, 4),
        "severity": severity,
        "baseline_class_distribution": baseline_dist,
        "current_class_distribution": actual_dist,
        "avg_text_length_baseline": round(avg_len_baseline, 1),
        "avg_text_length_current": round(avg_len_actual, 1),
        "avg_text_length_pct_change": round(len_pct_change * 100, 1),
    }
