"""
Step 2 of the pipeline: clean up raw OCR text and normalize it into
structured records, with a lightweight PII-redaction pass.

Three things happen here, each mapping to a JD line item:
  1. "text extraction ... metadata normalization" -> regex field extraction
     (MRN, DOB, service date) into a flat schema.
  2. common OCR error correction -> a small find/replace table for the
     character confusions Tesseract makes most often on this kind of
     monospace-ish clinical text (0/O, 1/l/I, rn/m).
  3. "PHI/PII-aware data handling" -> MRN / DOB / bare-name-line patterns are
     masked before the text is allowed into the modeling dataset.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from clinical_mlops.config import settings
from clinical_mlops.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- OCR error correction -------------------------------------------------
_OCR_FIXES = [
    (re.compile(r"\bl\b"), "I"),
    (re.compile(r"(?<=[A-Za-z])0(?=[A-Za-z])"), "o"),
    (re.compile(r"rn"), "m"),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def clean_text(raw: str) -> str:
    text = raw
    for pattern, repl in _OCR_FIXES:
        text = pattern.sub(repl, text)
    return text.strip()


# --- field extraction -------------------------------------------------------
_FIELD_PATTERNS = {
    "mrn": re.compile(r"MRN[-:\s]*([A-Za-z0-9-]+)"),
    "dob": re.compile(r"DOB[:\s]*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})"),
    "service_date": re.compile(
        r"(?:Date of Service|Study Date|Admit Date|Collected)[:\s]*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})"
    ),
    "provider": re.compile(r"Dr\.?\s+([A-Za-z]+)"),
}


def extract_fields(text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, pattern in _FIELD_PATTERNS.items():
        match = pattern.search(text)
        fields[name] = match.group(1) if match else None
    return fields


# --- PII redaction -----------------------------------------------------------
# Order matters: the patient-name pattern must run FIRST, while "MRN"/"DOB"
# are still literally present as stop-anchors — otherwise a name pattern
# that matches on generic word characters can swallow an adjacent field
# label (or a previous redaction placeholder) into one garbled match.
_PII_PATTERNS = [
    (re.compile(r"Patient:\s*.+?(?=\s*MRN|\s*DOB|\n|$)"), "Patient: [REDACTED]"),
    (re.compile(r"MRN[-:\s]*[A-Za-z0-9-]+"), "MRN-[REDACTED]"),
    (re.compile(r"DOB[:\s]*[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}"), "DOB-[REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN-REDACTED]"),
]


def redact_pii(text: str) -> str:
    redacted = text
    for pattern, repl in _PII_PATTERNS:
        redacted = pattern.sub(repl, redacted)
    return redacted


def normalize_document(doc_id: str, doc_type: str, raw_text: str) -> dict[str, Any]:
    cleaned = clean_text(raw_text)
    fields = extract_fields(cleaned)
    body = redact_pii(cleaned)
    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        **fields,
        "text": body,
        "char_count": len(body),
    }


def run(ocr_manifest_path: Path | None = None) -> Path:
    ocr_manifest_path = ocr_manifest_path or (settings.interim_dir / "ocr_manifest.jsonl")
    if not ocr_manifest_path.exists():
        raise FileNotFoundError(f"{ocr_manifest_path} not found — run ocr_extract.py first.")

    out_path = settings.interim_dir / "normalized.jsonl"
    records = []
    with ocr_manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            raw_text = (settings.data_path / row["text_path"]).read_text(encoding="utf-8")
            norm = normalize_document(row["doc_id"], row["doc_type"], raw_text)
            norm["ocr_confidence"] = row.get("ocr_confidence")
            records.append(norm)

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_missing_mrn = sum(1 for r in records if r.get("mrn") is None)
    logger.info(
        "Normalized %d documents -> %s (%d had no MRN extracted, likely low-confidence OCR)",
        len(records), out_path, n_missing_mrn,
    )
    return out_path


if __name__ == "__main__":
    run()
