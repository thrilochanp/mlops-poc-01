#!/usr/bin/env bash
# One-shot end-to-end demo: synthetic scans -> OCR -> normalize -> versioned
# dataset -> train -> evaluate. Run from the project root after `make setup`.
set -euo pipefail

echo "== 1/3 Running ingestion pipeline (generate + OCR + normalize + version) =="
python -m clinical_mlops.data_pipeline.pipeline

echo "== 2/3 Training classifier and promoting to Production =="
python -m clinical_mlops.models.train

echo "== 3/3 Evaluating Production model on hidden test set =="
python -m clinical_mlops.models.evaluate

echo
echo "Demo complete."
echo "  - API:       make serve   (then POST http://localhost:8000/predict)"
echo "  - Streamlit: make ui      (Annotate / Clinical Review / Ops Dashboard)"
echo "  - MLflow UI: make mlflow-ui"
