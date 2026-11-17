"""
Thin wrapper around the MLflow Client for the handful of model-registry
operations this POC needs: register a run's model, promote a version to
Production, and load whatever is currently in Production for serving.

Centralizing this here means api.py, evaluate.py, and the Streamlit ops
dashboard all agree on what "the production model" means.
"""
from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from clinical_mlops.config import settings
from clinical_mlops.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.resolved_mlflow_uri)
    return MlflowClient()


def get_or_create_experiment(name: str) -> str:
    mlflow.set_tracking_uri(settings.resolved_mlflow_uri)
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(name)


def register_and_promote(run_id: str, model_name: str = settings.model_name,
                          stage: str = settings.model_stage) -> "mlflow.entities.model_registry.ModelVersion":
    """Register the model logged under this run and promote it straight to
    `stage`. A real team would gate the promotion on a manual review /
    approval step in CI; this POC promotes automatically so the demo has
    something in Production to serve end-to-end."""
    client = get_client()
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage=stage,
        archive_existing_versions=True,
    )
    logger.info("Registered %s v%s and promoted to %s", model_name, result.version, stage)
    return result


def load_production_model(model_name: str = settings.model_name, stage: str = settings.model_stage):
    mlflow.set_tracking_uri(settings.resolved_mlflow_uri)
    uri = f"models:/{model_name}/{stage}"
    return mlflow.sklearn.load_model(uri)


def get_production_version_info(model_name: str = settings.model_name, stage: str = settings.model_stage):
    client = get_client()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None
    mv = versions[0]
    run = client.get_run(mv.run_id)
    return {
        "name": model_name,
        "version": mv.version,
        "stage": mv.current_stage,
        "run_id": mv.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }
