"""
Clinical review / QA tool: shows the normalized (PII-redacted) OCR text next
to the model's prediction and confidence, plus the OCR confidence score from
the extraction step, so a reviewer can spot low-quality OCR or a
questionable prediction and flag it. Decisions are appended to
data/interim/reviews.csv, which the Ops Dashboard rolls up into a
flag-rate metric.
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
from clinical_mlops.config import settings
from clinical_mlops.models import registry

st.set_page_config(page_title="Clinical Review", layout="wide")
st.title("Clinical Review & QA")

NORMALIZED_PATH = settings.interim_dir / "normalized.jsonl"
REVIEWS_PATH = settings.interim_dir / "reviews.csv"


@st.cache_data
def load_normalized():
    if not NORMALIZED_PATH.exists():
        return []
    return [json.loads(line) for line in NORMALIZED_PATH.read_text(encoding="utf-8").splitlines() if line]


@st.cache_resource
def load_model():
    try:
        return registry.load_production_model(), registry.get_production_version_info()
    except Exception:
        return None, None


def load_reviewed_ids() -> set:
    if not REVIEWS_PATH.exists():
        return set()
    with REVIEWS_PATH.open("r", encoding="utf-8") as f:
        return {row["doc_id"] for row in csv.DictReader(f)}


def save_review(doc_id: str, predicted: str, confidence: float, decision: str, corrected_label: str, reviewer: str):
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not REVIEWS_PATH.exists()
    with REVIEWS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "doc_id", "predicted", "confidence", "decision", "corrected_label", "reviewer", "reviewed_at",
        ])
        if is_new:
            writer.writeheader()
        writer.writerow({
            "doc_id": doc_id, "predicted": predicted, "confidence": confidence,
            "decision": decision, "corrected_label": corrected_label,
            "reviewer": reviewer or "anonymous", "reviewed_at": datetime.now(timezone.utc).isoformat(),
        })


docs = load_normalized()
if not docs:
    st.warning("No normalized documents found. Run `make pipeline` first.")
    st.stop()

model, info = load_model()
if model is None:
    st.error("No Production model available yet — train one with `make train`.")
    st.stop()

reviewer = st.sidebar.text_input("Reviewer name", value="")
reviewed_ids = load_reviewed_ids()
pending = [d for d in docs if d["doc_id"] not in reviewed_ids]

st.sidebar.metric("Total documents", len(docs))
st.sidebar.metric("Reviewed", len(reviewed_ids))
st.sidebar.metric("Pending", len(pending))

if not pending:
    st.success("All documents reviewed.")
    st.stop()

doc = pending[0]
pred = model.predict([doc["text"]])[0]
proba = model.predict_proba([doc["text"]])[0]
confidence = float(max(proba))

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"**Doc ID:** `{doc['doc_id']}`  |  **OCR confidence:** {doc.get('ocr_confidence', 'n/a')}")
    st.text_area("Normalized, PII-redacted text", doc["text"], height=350)
with col2:
    st.metric("Predicted document type", pred)
    st.metric("Model confidence", f"{confidence:.1%}")
    st.caption(f"Model version: {info['version']}")

    decision = st.radio("Review decision", ["approve", "flag"])
    corrected = ""
    if decision == "flag":
        from clinical_mlops.config import DOC_TYPES
        corrected = st.selectbox("Correct label (if known)", ["unsure"] + DOC_TYPES)
    if st.button("Submit review", type="primary"):
        save_review(doc["doc_id"], pred, confidence, decision, corrected, reviewer)
        st.cache_data.clear()
        st.rerun()
