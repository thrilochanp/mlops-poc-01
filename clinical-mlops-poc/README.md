# Clinical Document MLOps POC

A small, runnable proof-of-concept built to learn the MLOps & Data
Engineering patterns behind an "MLOps & Data Engineer" role supporting a
clinical ML team: process clinical documents end to end, track experiments,
serve and monitor a model, and support annotation/review with lightweight
internal tools.

Everything here uses **synthetic, fake patient data** — there is no real
PHI anywhere in this repo. Documents are generated, rendered as noisy
scanned-page images, and then run through real Tesseract OCR, so the
pipeline experiences genuine OCR noise instead of clean strings.

## What it does

A four-class document-type classifier (`discharge_summary`,
`radiology_report`, `progress_note`, `lab_report`) is trained on OCR'd,
PII-redacted clinical notes, tracked and versioned with MLflow, served
through FastAPI, and monitored for drift — with Streamlit tools for
annotation, clinical review/QA, and an ops dashboard.

```
generate synthetic scans --> OCR (tesseract) --> normalize + redact PII --> versioned dataset
                                                                                    |
                                                                                    v
                                                            train (MLflow tracking + registry)
                                                                                    |
                                                                                    v
                                        evaluate on hidden test set (regression gate) --> Production
                                                                                    |
                                        +-------------------+-----------------------+
                                        v                   v
                                  FastAPI /predict     Streamlit tools
                                        |             (Annotate, Review, Ops Dashboard)
                                        v
                                monitoring: prediction logs -> drift (PSI)
```

## Project layout

```
src/clinical_mlops/
  config.py                  central settings (paths, MLflow URI, thresholds)
  data_pipeline/
    generate_synthetic_docs.py   fake clinical notes -> noisy scanned-page images
    ocr_extract.py                real Tesseract OCR -> raw text + confidence
    normalize.py                  OCR-error cleanup, field extraction, PII redaction
    dataset_registry.py           versioned train/val/test splits + manifest (DVC-lite)
    pipeline.py                   Prefect flow chaining the four steps above
  models/
    features.py                   TF-IDF vectorizer (shared train/serve)
    train.py                      trains + logs to MLflow + registers + promotes
    evaluate.py                   hidden-test-set eval + regression gate
    registry.py                   MLflow Model Registry helpers
  serving/
    api.py, schemas.py            FastAPI inference service
  monitoring/
    logger.py                     JSONL prediction logging
    drift.py                      PSI-based drift detection vs. training baseline
apps/
  streamlit_app.py + pages/       Annotate, Clinical Review, Ops Dashboard
tests/                            pytest suite (pipeline, training, API, drift)
.github/workflows/ci.yml          lint + test + full pipeline/train/eval smoke test + docker build
Dockerfile, docker-compose.yml    one-command local stack (MLflow, API, Streamlit)
Makefile                          shortcuts for every command below
```

## Quickstart (local, no Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup                 # pip install -r requirements.txt
                            # requires the tesseract-ocr system package —
                            # apt: `sudo apt-get install tesseract-ocr`
                            # mac: `brew install tesseract`

make demo                   # generate -> OCR -> normalize -> version -> train -> evaluate
# or step by step:
make generate-data
make pipeline               # OCR + normalize + build a dataset version
make train                  # trains + registers + promotes to Production in MLflow
make evaluate                # scores the hidden test split, gates on regression

make serve                  # FastAPI on :8000
make ui                     # Streamlit on :8501 (Annotate / Clinical Review / Ops Dashboard)
make mlflow-ui              # MLflow UI on :5000
make test                   # pytest
make lint                   # ruff
```

Try the API once `make serve` is running:

```bash
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "RADIOLOGY REPORT ... FINDINGS: no acute abnormality ... IMPRESSION: normal study"}'
```

## Quickstart (Docker Compose)

Run the local quickstart above first (or at least `make pipeline && make
train`) so there's a Production model in `./mlruns` for the containers to
load, then:

```bash
make docker-up     # mlflow:5000, api:8000, streamlit:8501
```

## How this maps to the job description

| JD requirement | Where it lives |
|---|---|
| Clinical document pipelines, OCR outputs, text extraction, metadata normalization | `data_pipeline/generate_synthetic_docs.py`, `ocr_extract.py` (real Tesseract), `normalize.py` |
| Dataset preparation & versioning | `data_pipeline/dataset_registry.py` (content-hashed, split, versioned) |
| Training / inference / evaluation infrastructure | `models/train.py`, `models/evaluate.py`, `serving/api.py` |
| Experiment tracking, model versioning, reproducibility | MLflow tracking + Model Registry; every run tags its exact dataset version/hash |
| CI/CD | `.github/workflows/ci.yml` — lint, unit tests, a full pipeline→train→eval smoke run, and a Docker build |
| Monitoring, observability | `monitoring/drift.py` (PSI vs. training baseline), `monitoring/logger.py`, Ops Dashboard |
| Internal tools for annotation, clinical review, evaluation, QA, data inspection, ops | `apps/pages/1_Annotate.py`, `2_Clinical_Review.py`, `3_Ops_Dashboard.py` |
| Automating recurring ML workflows | `data_pipeline/pipeline.py` (Prefect flow), `scripts/demo.sh`, `Makefile` |
| Productionizing prototypes / deployment support | Dockerfile + docker-compose, FastAPI service boundary that DevOps would front with a load balancer |
| Regression automation, hidden test set runs | `models/evaluate.py --fail-on-regression`, wired into CI |
| PHI/PII-aware data handling | `normalize.py` PII redaction pass; prediction logs store only length/label/confidence, never raw text |

## Known simplifications (it's a POC)

- The classifier is TF-IDF + logistic regression, not an LLM — the point is
  the surrounding infrastructure, not model sophistication.
- Model promotion to "Production" in `train.py` is automatic; a real
  pipeline would gate that behind a review/approval step in CI.
- MLflow runs against a local file store by default (`./mlruns`); pointing
  `MLFLOW_TRACKING_URI` at a real tracking server needs no code changes.
- Drift detection compares against a single static training-time baseline
  rather than a rolling window — enough to demonstrate the mechanism.
- `anyio` is pinned to `4.4.0` in requirements.txt: newer anyio releases
  break Prefect 2.x's flow-run teardown (`GatherTaskGroup` abstract-class
  error) even though Prefect's own package metadata allows them. If you
  upgrade Prefect, re-check whether this pin is still needed.
