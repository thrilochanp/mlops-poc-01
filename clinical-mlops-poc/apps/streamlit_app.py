"""Home page for the internal-tools Streamlit app (multipage app; see
apps/pages/ for Annotate, Clinical Review, and Ops Dashboard).

Run with: streamlit run apps/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from clinical_mlops.config import settings
from clinical_mlops.data_pipeline import dataset_registry

st.set_page_config(page_title="Clinical MLOps POC", page_icon=":hospital:", layout="wide")

st.title("Clinical Document MLOps — Internal Tools")
st.write(
    "Lightweight internal tools built for Research Engineers, ML Evaluation "
    "Engineers, and Clinical AI Data Specialists to annotate, review, and "
    "monitor the document-classification pipeline without needing to touch "
    "the underlying code."
)

st.subheader("What's here")
st.markdown(
    "- **Annotate** — label / correct raw document-type ground truth for the next training round.\n"
    "- **Clinical Review** — side-by-side OCR text + model prediction, for QA and spot-checks.\n"
    "- **Ops Dashboard** — experiment history, current Production model, and drift monitoring.\n"
)

st.subheader("Current dataset versions")
versions = dataset_registry.list_versions()
if not versions:
    st.info("No dataset versions yet — run `make pipeline` (or "
            "`python -m clinical_mlops.data_pipeline.pipeline`) first.")
else:
    st.dataframe(
        [
            {
                "version": v["version"],
                "created_at": v["created_at"],
                "n_total": v["n_total"],
                "n_train": v["n_train"],
                "n_val": v["n_val"],
                "n_test": v["n_test"],
                "content_hash": v["content_hash"],
            }
            for v in versions
        ],
        use_container_width=True,
    )

st.caption(f"Data directory: {settings.data_path}")
