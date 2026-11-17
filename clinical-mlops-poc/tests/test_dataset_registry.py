import json

from clinical_mlops.data_pipeline import dataset_registry


def _write_normalized(path, n=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc_types = ["discharge_summary", "radiology_report", "progress_note", "lab_report"]
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "doc_id": f"doc_{i:04d}",
                "doc_type": doc_types[i % len(doc_types)],
                "text": f"sample text {i}",
                "mrn": None, "dob": None, "service_date": None, "provider": None,
                "char_count": 20, "ocr_confidence": 90.0,
            }) + "\n")


def test_build_version_creates_splits_and_manifest(tmp_project):
    normalized_path = tmp_project / "data" / "interim" / "normalized.jsonl"
    _write_normalized(normalized_path, n=20)

    manifest = dataset_registry.build_version(normalized_path, seed=1)

    assert manifest["version"] == 1
    assert manifest["n_total"] == 20
    assert manifest["n_train"] + manifest["n_val"] + manifest["n_test"] == 20
    assert set(manifest["class_balance"].keys()) == {
        "discharge_summary", "radiology_report", "progress_note", "lab_report"
    }

    train = dataset_registry.load_split(1, "train")
    assert len(train) == manifest["n_train"]


def test_build_version_is_deterministic_given_same_seed(tmp_project):
    normalized_path = tmp_project / "data" / "interim" / "normalized.jsonl"
    _write_normalized(normalized_path, n=12)

    m1 = dataset_registry.build_version(normalized_path, seed=7)
    m2 = dataset_registry.build_version(normalized_path, seed=7)

    assert m1["content_hash"] == m2["content_hash"]
    assert m2["version"] == m1["version"] + 1


def test_latest_version_and_list_versions(tmp_project):
    normalized_path = tmp_project / "data" / "interim" / "normalized.jsonl"
    _write_normalized(normalized_path, n=10)

    assert dataset_registry.latest_version() is None
    dataset_registry.build_version(normalized_path, seed=1)
    dataset_registry.build_version(normalized_path, seed=2)

    assert dataset_registry.latest_version() == 2
    assert len(dataset_registry.list_versions()) == 2
