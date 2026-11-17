import json

from clinical_mlops.config import DOC_TYPES
from clinical_mlops.data_pipeline import dataset_registry
from fastapi.testclient import TestClient

CLASS_TEXT = {
    "discharge_summary": "discharge summary hospital course medications follow up attending physician",
    "radiology_report": "radiology report imaging findings impression modality study clinical indication",
    "progress_note": "progress note subjective objective assessment plan vitals provider visit",
    "lab_report": "laboratory report results interpretation collected resulted specimen values",
}


def _write_normalized(path, n_per_class=15):
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


def _train_a_model(tmp_project):
    from clinical_mlops.models import train as train_mod
    normalized_path = tmp_project / "data" / "interim" / "normalized.jsonl"
    _write_normalized(normalized_path, n_per_class=15)
    dataset_registry.build_version(normalized_path, seed=1)
    train_mod.train(dataset_version=1)


def test_health_ok(tmp_project):
    from clinical_mlops.serving.api import app
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_predict_without_model_returns_503(tmp_project):
    from clinical_mlops.serving.api import app
    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": "some clinical note text"})
        assert resp.status_code == 503


def test_predict_after_training_returns_a_known_label(tmp_project):
    _train_a_model(tmp_project)

    from clinical_mlops.serving.api import app

    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": "radiology report imaging findings impression"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_type"] in DOC_TYPES
        assert 0.0 <= body["confidence"] <= 1.0

        info_resp = client.get("/model-info")
        assert info_resp.status_code == 200
