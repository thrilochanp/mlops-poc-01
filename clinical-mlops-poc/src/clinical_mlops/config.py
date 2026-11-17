"""
Central configuration for the POC.

Everything is a plain, overridable setting sourced from environment
variables / a local .env file, so the same code runs unchanged on a laptop,
inside Docker Compose, or in CI.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=("settings_",))

    # --- storage locations ---
    # explicit aliases because these two are prefixed (CLINICAL_MLOPS_*) in
    # .env.example to avoid colliding with generic DATA_DIR/LOG_DIR names a
    # shell or CI environment might already use for something else.
    data_dir: str = Field(default="data", validation_alias="CLINICAL_MLOPS_DATA_DIR")
    log_dir: str = Field(default="logs", validation_alias="CLINICAL_MLOPS_LOG_DIR")

    # --- MLflow ---
    mlflow_tracking_uri: str = ""  # empty -> local ./mlruns folder
    model_name: str = "clinical-doc-classifier"
    model_stage: str = "Production"

    # --- evaluation ---
    eval_regression_tolerance: float = 0.03

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def data_path(self) -> Path:
        return (PROJECT_ROOT / self.data_dir).resolve()

    @property
    def raw_dir(self) -> Path:
        return self.data_path / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_path / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_path / "processed"

    @property
    def log_path(self) -> Path:
        return (PROJECT_ROOT / self.log_dir).resolve()

    @property
    def resolved_mlflow_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        # local file-based tracking store, good enough for a laptop POC
        return f"file:{(PROJECT_ROOT / 'mlruns').resolve()}"


settings = Settings()

DOC_TYPES = [
    "discharge_summary",
    "radiology_report",
    "progress_note",
    "lab_report",
]

for _p in (settings.raw_dir, settings.interim_dir, settings.processed_dir, settings.log_path):
    _p.mkdir(parents=True, exist_ok=True)
