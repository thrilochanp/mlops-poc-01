import json

from clinical_mlops.monitoring.drift import compute_drift
from clinical_mlops.monitoring.logger import log_prediction


def _write_baseline(tmp_project):
    baseline = {
        "dataset_version": 1,
        "class_distribution": {
            "discharge_summary": 40, "radiology_report": 40,
            "progress_note": 40, "lab_report": 40,
        },
        "avg_text_length": 500.0,
    }
    from clinical_mlops.config import settings
    (settings.log_path / "training_baseline.json").write_text(json.dumps(baseline), encoding="utf-8")


def test_compute_drift_no_baseline_yet(tmp_project):
    result = compute_drift()
    assert result["status"] == "no_baseline"


def test_compute_drift_insufficient_data(tmp_project):
    _write_baseline(tmp_project)
    log_prediction("lab_report", 0.9, "1", 500)
    result = compute_drift(min_predictions=10)
    assert result["status"] == "insufficient_data"


def test_compute_drift_flags_significant_shift(tmp_project):
    _write_baseline(tmp_project)
    for _ in range(20):
        log_prediction("lab_report", 0.9, "1", 500)
    result = compute_drift(min_predictions=10)
    assert result["status"] == "ok"
    assert result["severity"] == "significant"
    assert result["class_distribution_psi"] > 0.25
