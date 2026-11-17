"""
Evaluate the current Production model against its dataset's held-out TEST
split (never seen during training or validation — the "hidden test set" in
the JD) and run a regression check against the last stored baseline.

Logged as its own MLflow run (tagged stage="evaluation") so evaluation
history lives alongside training runs in the same experiment, which is what
the Streamlit ops dashboard reads to plot accuracy over time. Exits non-zero
on a regression so this doubles as a CI/CD gate (see .github/workflows/ci.yml
and scripts/run_eval.sh) — "regression automation" from the JD.

Usage: python -m clinical_mlops.models.evaluate
"""
from __future__ import annotations

import argparse
import json
import sys

import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from clinical_mlops.config import DOC_TYPES, settings
from clinical_mlops.data_pipeline import dataset_registry
from clinical_mlops.models import registry
from clinical_mlops.utils.logging_config import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "clinical-doc-classification"


def _baseline_path():
    # Resolved lazily so tests pointing `settings` at a tmp dir take effect.
    return settings.log_path / "eval_baseline.json"


def _to_xy(records: list[dict]):
    return [r["text"] for r in records], [r["doc_type"] for r in records]


def evaluate(tolerance: float | None = None) -> dict:
    tolerance = tolerance if tolerance is not None else settings.eval_regression_tolerance
    mlflow.set_tracking_uri(settings.resolved_mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    info = registry.get_production_version_info()
    if info is None:
        raise RuntimeError("No Production model found — run models/train.py first.")

    dataset_version = int(info["tags"].get("dataset_version", dataset_registry.latest_version()))
    test_records = dataset_registry.load_split(dataset_version, "test")
    x_test, y_test = _to_xy(test_records)

    model = registry.load_production_model()
    preds = model.predict(x_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds, labels=DOC_TYPES).tolist()

    result = {
        "model_version": info["version"],
        "dataset_version": dataset_version,
        "n_test": len(test_records),
        "test_accuracy": acc,
        "test_f1_macro": f1,
    }

    with mlflow.start_run(run_name=f"eval-model-v{info['version']}"):
        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("evaluated_model_version", info["version"])
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("n_test", len(test_records))
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_macro", f1)

        cm_path = settings.log_path / "confusion_matrix_test.json"
        cm_path.write_text(json.dumps({"labels": DOC_TYPES, "matrix": cm}, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(cm_path))

        regressed = False
        baseline_path = _baseline_path()
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            drop = baseline["test_accuracy"] - acc
            mlflow.log_metric("accuracy_drop_vs_baseline", drop)
            if drop > tolerance:
                regressed = True
                logger.warning(
                    "REGRESSION: test_accuracy dropped %.3f (baseline %.3f -> %.3f, tolerance %.3f)",
                    drop, baseline["test_accuracy"], acc, tolerance,
                )
            else:
                logger.info("No regression: baseline %.3f -> current %.3f (tolerance %.3f)",
                            baseline["test_accuracy"], acc, tolerance)
        else:
            logger.info("No prior baseline found; storing this run as the new baseline.")

        mlflow.set_tag("regression_detected", str(regressed))
        result["regression_detected"] = regressed

    if not regressed:
        baseline_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info("Eval result: %s", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Production model on its hidden test set.")
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--fail-on-regression", action="store_true",
                         help="Exit with a non-zero status if a regression is detected (for CI gating).")
    args = parser.parse_args()
    res = evaluate(tolerance=args.tolerance)
    if args.fail_on_regression and res["regression_detected"]:
        sys.exit(1)
