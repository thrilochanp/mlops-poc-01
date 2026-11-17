"""
Annotation tool: a human reviews the scanned document image (not the noisy
OCR text) and confirms or corrects its document-type label. This is the
"ground truth" a real pipeline would use to retrain/expand the dataset —
here it writes to data/interim/annotations.csv, kept separate from the
auto-generated ground_truth.jsonl so you can see human corrections
independently of the synthetic generator's labels.
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st
from clinical_mlops.config import DOC_TYPES, settings

st.set_page_config(page_title="Annotate", layout="wide")
st.title("Document Annotation")

GT_PATH = settings.raw_dir / "ground_truth.jsonl"
ANNOTATIONS_PATH = settings.interim_dir / "annotations.csv"


@st.cache_data
def load_documents():
    if not GT_PATH.exists():
        return []
    return [json.loads(line) for line in GT_PATH.read_text(encoding="utf-8").splitlines() if line]


def load_existing_annotations() -> dict:
    if not ANNOTATIONS_PATH.exists():
        return {}
    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as f:
        return {row["doc_id"]: row for row in csv.DictReader(f)}


def save_annotation(doc_id: str, doc_type: str, annotator: str, note: str) -> None:
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_annotations()
    existing[doc_id] = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "annotator": annotator or "anonymous",
        "note": note,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }
    with ANNOTATIONS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "doc_type", "annotator", "note", "annotated_at"])
        writer.writeheader()
        writer.writerows(existing.values())


docs = load_documents()
if not docs:
    st.warning("No documents found. Run `make generate-data` first.")
    st.stop()

annotator = st.sidebar.text_input("Your name", value="")
annotations = load_existing_annotations()
unannotated = [d for d in docs if d["doc_id"] not in annotations]

st.sidebar.metric("Total documents", len(docs))
st.sidebar.metric("Annotated", len(annotations))
st.sidebar.metric("Remaining", len(unannotated))

if not unannotated:
    st.success("All documents annotated.")
    st.dataframe(list(annotations.values()), use_container_width=True)
    st.stop()

doc = unannotated[0]
col1, col2 = st.columns([1, 1])
with col1:
    st.image(str(settings.data_path / doc["image_path"]), caption=doc["doc_id"], use_container_width=True)
with col2:
    st.markdown("**Assign / confirm document type:**")
    choice = st.radio("Document type", DOC_TYPES, index=DOC_TYPES.index(doc["doc_type"]))
    note = st.text_area("Note (optional)")
    if st.button("Save annotation and next", type="primary"):
        save_annotation(doc["doc_id"], choice, annotator, note)
        st.cache_data.clear()
        st.rerun()
