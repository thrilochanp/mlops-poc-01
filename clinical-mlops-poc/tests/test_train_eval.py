import json

from clinical_mlops.config import DOC_TYPES
from clinical_mlops.data_pipeline import dataset_registry

CLASS_TEXT = {
    "discharge_summary": "discharge summary hospital course medications follow up attending physician",
    "radiology_report": "radiology report imaging findings impression modality study clinical indication",
    "progress_note": "progress note subjective objective assessment plan vitals provider visit",
    "lab_report": "laboratory report results interpretation collected resulted specimen values",
}


def _write_normalized(path, n_per_class=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        doc_i = 0
        for doc_type in DOC_TYPES:
            for i in range(n_per_class):
                doc_i += 1
                text = f"{CLASS_TEXT[doc_type]} case number {i}"
                f.write(json.dumps({
                    "doc_id": f"doc_{doc_i:04d}", "doc_type": doc_type, "text": text,
                    "mrn": None, "dob": None, "service_date": None, "provider": None,
                    "char_count": len(text), "ocr_confidence": 92.0,
                }) + "\n")


def test_train_then_evaluate_end_to_end(tmp_project):
    from clinical_mlops.models import evaluate as evaluate_mod
    from clinical_mlops.models import train as train_mod

    normalized_path = tmp_project / "data" / "interim" / "normalized.jsonl"
    _write_normalized(normalized_path, n_per_class=20)
    dataset_registry.build_version(normalized_path, seed=1)

    run_id = train_mod.train(dataset_version=1)
    assert run_id

    result = evaluate_mod.evaluate()
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert result["dataset_version"] == 1
    assert result["regression_detected"] is False

    result2 = evaluate_mod.evaluate()
    assert "regression_detected" in result2
