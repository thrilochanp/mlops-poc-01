"""
Train a document-type classifier (discharge summary / radiology report /
progress note / lab report) on the OCR'd, normalized, PII-redacted text
produced by the data pipeline — and track the run with MLflow.

Every run logs: the dataset version + content hash it was trained on
(reproducibility), hyperparameters, validation metrics, a confusion-matrix
artifact, and the fitted sklearn Pipeline itself, then registers that model
and promotes it to Production so serving/api.py has something to load.

Usage: python -m clinical_mlops.models.train [--dataset-version N]
"""
from __future__ import annotations

import argparse
import json

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from clinical_mlops.config import DOC_TYPES, settings
from clinical_mlops.data_pipeline import dataset_registry
from clinical_mlops.models import registry
from clinical_mlops.models.features import build_vectorizer
from clinical_mlops.utils.logging_config import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "clinical-doc-classification"


def _to_xy(records: list[dict]):
    x = [r["text"] for r in records]
    y = [r["doc_type"] for r in records]
    return x, y


def train(dataset_version: int | None = None, C: float = 2.0) -> str:
    mlflow.set_tracking_uri(settings.resolved_mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    dataset_version = dataset_version or dataset_registry.latest_version()
    if dataset_version is None:
        raise RuntimeError("No dataset versions found — run the ingestion pipeline first.")

    manifest = dataset_registry.load_manifest(dataset_version)
    train_records = dataset_registry.load_split(dataset_version, "train")
    val_records = dataset_registry.load_split(dataset_version, "val")

    x_train, y_train = _to_xy(train_records)
    x_val, y_val = _to_xy(val_records)

    pipeline = Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf", LogisticRegression(C=C, max_iter=1000)),
    ])

    with mlflow.start_run(run_name=f"train-v{dataset_version}") as run:
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("dataset_content_hash", manifest["content_hash"])
        mlflow.log_param("n_train", len(train_records))
        mlflow.log_param("n_val", len(val_records))
        mlflow.log_param("model_type", "tfidf+logistic_regression")
        mlflow.log_param("C", C)

        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_val)

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_macro", f1)

        cm = confusion_matrix(y_val, preds, labels=DOC_TYPES).tolist()
        cm_path = settings.log_path / "confusion_matrix_val.json"
        cm_path.write_text(json.dumps({"labels": DOC_TYPES, "matrix": cm}, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(cm_path))

        baseline = {
            "dataset_version": dataset_version,
            "class_distribution": manifest["class_balance"],
            "avg_text_length": sum(len(t) for t in x_train) / len(x_train),
        }
        baseline_path = settings.log_path / "training_baseline.json"
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(baseline_path))

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.set_tag("dataset_version", str(dataset_version))

        logger.info("Run %s: val_accuracy=%.3f val_f1_macro=%.3f (dataset v%d)",
                    run.info.run_id, acc, f1, dataset_version)

        registry.register_and_promote(run.info.run_id)
        return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the clinical document-type classifier.")
    parser.add_argument("--dataset-version", type=int, default=None)
    parser.add_argument("--C", type=float, default=2.0)
    args = parser.parse_args()
    train(dataset_version=args.dataset_version, C=args.C)
