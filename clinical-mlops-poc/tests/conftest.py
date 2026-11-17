import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_mlops.config import settings  # noqa: E402


@pytest.fixture()
def tmp_project(tmp_path, monkeypatch):
    """Point the shared `settings` singleton's data/log/MLflow locations at
    a throwaway tmp directory for the duration of one test.

    Every module in the app does `from clinical_mlops.config import
    settings` and holds a reference to this same object, so mutating its
    attributes here (via monkeypatch, which auto-reverts after the test) is
    visible everywhere immediately — no need to delete modules from
    sys.modules and hope things get re-imported in the right order.
    """
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "data"))
    monkeypatch.setattr(settings, "log_dir", str(tmp_path / "logs"))
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"file:{tmp_path / 'mlruns'}")

    for p in (settings.raw_dir, settings.interim_dir, settings.processed_dir, settings.log_path):
        p.mkdir(parents=True, exist_ok=True)

    yield tmp_path
