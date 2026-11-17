"""
Step 1 of the pipeline: real OCR text extraction from the scanned page images
produced by generate_synthetic_docs.py, using pytesseract (Tesseract OCR).

This is the piece that stands in for "process OCR outputs" in the job
description. Output: one .txt per document under data/interim/<doc_type>/,
plus an ocr_manifest.jsonl carrying per-document OCR confidence so low
quality scans can be triaged later (that confidence signal also feeds the
Streamlit clinical-review tool).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytesseract
from PIL import Image

from clinical_mlops.config import settings
from clinical_mlops.utils.logging_config import get_logger

logger = get_logger(__name__)


def _ocr_with_confidence(image_path: Path) -> tuple[str, float]:
    """Single Tesseract invocation via image_to_data (word boxes + confidence),
    reconstructing line-broken text from the word list. Doing this instead of
    calling image_to_string() *and* image_to_data() avoids running OCR twice
    per document, which matters once the corpus is more than a handful of
    pages (each Tesseract call takes a couple of seconds)."""
    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    confs = [int(c) for c in data.get("conf", []) if str(c) not in ("-1",)]
    avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0

    lines: dict[tuple, list[str]] = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)

    text = "\n".join(" ".join(words) for words in lines.values())
    return text.strip(), avg_conf


def run(ground_truth_path: Path | None = None) -> Path:
    ground_truth_path = ground_truth_path or (settings.raw_dir / "ground_truth.jsonl")
    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"{ground_truth_path} not found — run generate_synthetic_docs.py first."
        )

    manifest_path = settings.interim_dir / "ocr_manifest.jsonl"
    records = []
    with ground_truth_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            image_path = settings.data_path / row["image_path"]
            text, confidence = _ocr_with_confidence(image_path)

            out_txt = settings.interim_dir / row["doc_type"] / f"{row['doc_id']}.txt"
            out_txt.parent.mkdir(parents=True, exist_ok=True)
            out_txt.write_text(text, encoding="utf-8")

            records.append({
                "doc_id": row["doc_id"],
                "doc_type": row["doc_type"],
                "text_path": str(out_txt.relative_to(settings.data_path)),
                "ocr_confidence": confidence,
                "char_count": len(text),
            })

    with manifest_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    avg_conf = round(sum(r["ocr_confidence"] for r in records) / len(records), 1) if records else 0.0
    logger.info("OCR'd %d documents, avg confidence %.1f -> %s", len(records), avg_conf, manifest_path)
    return manifest_path


if __name__ == "__main__":
    run()
