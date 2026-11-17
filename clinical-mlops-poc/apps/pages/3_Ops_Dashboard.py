"""
Project-ops / monitoring dashboard: experiment history from MLflow, current
Production model, prediction volume + drift signal from the serving logs,
dataset versions, and the review flag-rate from the Clinical Review tool.
The one place a Research Engineer or ML Eval Engineer can look to answer
"is anything broken" without SSH-ing anywhere.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
from clinical_mlops.config import settings
from clinical_mlops.data_pipeline import dataset_registry
from clinical_mlops.models import registry
from clinical_mlops.models.train import EXPERIMENT_NAME
from clinical_mlops.monitoring.drift import compute_drift
from clinical_mlops.monitoring.logger import load_predictions

st.set_page_config(page_title="Ops Dashboard", layout="wide")
st.title("Ops & Monitoring Dashboard")

mlflow.set_tracking_uri(settings.resolved_mlflow_uri)

st.subheader("Current Production model")
info = registry.get_production_version_info()
if info is None:
    st.warning("No Production model registered yet. Run `make train`.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Version", info["version"])
    c2.metric("Val accuracy", f"{float(info['metrics'].get('val_accuracy', 0)):.1%}")
    c3.metric("Val F1 (macro)", f"{float(info['metrics'].get('val_f1_macro', 0)):.1%}")
    c4.metric("Trained on dataset", info["tags"].get("dataset_version", "n/a"))

st.subheader("Experiment history (MLflow)")
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    st.info("No experiment runs yet.")
else:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time ASC"])
    if runs.empty:
        st.info("No runs logged yet.")
    else:
        show_cols = [c for c in [
            "start_time", "tags.mlflow.runName", "tags.stage", "params.dataset_version",
            "metrics.val_accuracy", "metrics.val_f1_macro", "metrics.test_accuracy",
            "metrics.test_f1_macro", "tags.regression_detected",
        ] if c in runs.columns]
        st.dataframe(runs[show_cols], use_container_width=True)

        metric_col = "metrics.val_accuracy" if "metrics.val_accuracy" in runs.columns else None
        if metric_col:
            fig = px.line(runs, x="start_time", y=metric_col, markers=True,
                           title="Validation accuracy over training runs")
            st.plotly_chart(fig, use_container_width=True)

st.subheader("Dataset versions")
versions = dataset_registry.list_versions()
if versions:
    st.dataframe(pd.DataFrame(versions), use_container_width=True)
else:
    st.info("No dataset versions yet.")

st.subheader("Serving traffic & drift")
predictions = load_predictions()
st.metric("Total predictions logged", len(predictions))
if predictions:
    df = pd.DataFrame(predictions)
    fig2 = px.histogram(df, x="doc_type", title="Prediction volume by document type")
    st.plotly_chart(fig2, use_container_width=True)

drift = compute_drift()
if drift["status"] == "ok":
    severity_color = {"none": "green", "moderate": "orange", "significant": "red"}[drift["severity"]]
    st.markdown(f"**Drift severity:** :{severity_color}[{drift['severity'].upper()}]  "
                f"(PSI = {drift['class_distribution_psi']})")
    st.json(drift)
else:
    st.info(drift.get("message", drift["status"]))

st.subheader("Clinical review flag rate")
reviews_path = settings.interim_dir / "reviews.csv"
if reviews_path.exists():
    with reviews_path.open("r", encoding="utf-8") as f:
        reviews = list(csv.DictReader(f))
    n_flagged = sum(1 for r in reviews if r["decision"] == "flag")
    st.metric("Reviews logged", len(reviews))
    st.metric("Flag rate", f"{(n_flagged / len(reviews)):.1%}" if reviews else "n/a")
else:
    st.info("No reviews logged yet — use the Clinical Review tool.")
