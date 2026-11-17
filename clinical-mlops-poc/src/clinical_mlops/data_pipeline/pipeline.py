"""
Orchestration entry point: chains generate -> OCR -> normalize -> version
as a Prefect flow ("workflow orchestration" from the JD's nice-to-haves,
without needing a running Prefect server — `python -m clinical_mlops...pipeline`
executes it locally and still gets retries, logging, and a run graph you can
view with `prefect server start` if you want the UI).

Run with: python -m clinical_mlops.data_pipeline.pipeline
"""
from __future__ import annotations

import argparse

from prefect import flow, task

from clinical_mlops.data_pipeline import dataset_registry, generate_synthetic_docs, normalize, ocr_extract
from clinical_mlops.utils.logging_config import get_logger

logger = get_logger(__name__)


@task(retries=1, retry_delay_seconds=2, log_prints=True)
def generate_task(n_per_type: int, seed: int):
    return generate_synthetic_docs.generate(n_per_type=n_per_type, seed=seed)


@task(retries=1, retry_delay_seconds=2, log_prints=True)
def ocr_task(ground_truth_path):
    return ocr_extract.run(ground_truth_path)


@task(retries=1, retry_delay_seconds=2, log_prints=True)
def normalize_task(ocr_manifest_path):
    return normalize.run(ocr_manifest_path)


@task(log_prints=True)
def version_task(normalized_path, seed: int):
    return dataset_registry.build_version(normalized_path, seed=seed)


@flow(name="clinical-document-ingestion")
def ingestion_flow(n_per_type: int = 40, seed: int = 42, skip_generate: bool = False):
    """End-to-end: synthetic scans -> OCR -> normalize/redact -> versioned dataset."""
    if skip_generate:
        from clinical_mlops.config import settings
        gt_path = settings.raw_dir / "ground_truth.jsonl"
    else:
        gt_path = generate_task(n_per_type=n_per_type, seed=seed)

    ocr_manifest = ocr_task(gt_path)
    normalized = normalize_task(ocr_manifest)
    manifest = version_task(normalized, seed=seed)
    logger.info("Pipeline complete. Dataset version: %s", manifest["version"])
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the clinical document ingestion pipeline.")
    parser.add_argument("--n-per-type", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-generate", action="store_true",
                         help="Reuse existing data/raw/ instead of regenerating synthetic docs.")
    args = parser.parse_args()
    ingestion_flow(n_per_type=args.n_per_type, seed=args.seed, skip_generate=args.skip_generate)
