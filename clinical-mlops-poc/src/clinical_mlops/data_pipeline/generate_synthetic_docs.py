"""
Step 0 of the pipeline: manufacture a fake "scanned clinical documents" corpus.

We don't have (and don't want) real PHI, so this generates entirely synthetic
clinical notes with a Faker-driven patient/provider header plus templated
body text for four document types, then RENDERS each note as a noisy PNG
image (blur, rotation, speckle) to stand in for a scanned page. That gives
the rest of the pipeline a genuine image -> OCR -> text step instead of a
recycled string, which is what step 1 (ocr_extract.py) actually consumes.

Ground-truth text and the true label are stashed in raw/ground_truth.jsonl
purely so evaluate.py has something to score classification against and so
you can see how much OCR noise degraded the text. A real ingestion pipeline
obviously would not have this — it's a POC convenience.
"""
from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from clinical_mlops.config import DOC_TYPES, settings
from clinical_mlops.utils.logging_config import get_logger

logger = get_logger(__name__)
fake = Faker()

TEMPLATES = {
    "discharge_summary": (
        "DISCHARGE SUMMARY\n"
        "Patient: {name}   MRN: {mrn}   DOB: {dob}\n"
        "Admit Date: {date1}   Discharge Date: {date2}\n"
        "Attending: Dr. {doctor}\n\n"
        "HISTORY OF PRESENT ILLNESS:\n{hpi}\n\n"
        "HOSPITAL COURSE:\n{course}\n\n"
        "DISCHARGE MEDICATIONS: {meds}\n"
        "FOLLOW UP: {followup}\n"
    ),
    "radiology_report": (
        "RADIOLOGY REPORT\n"
        "Patient: {name}   MRN: {mrn}   DOB: {dob}\n"
        "Study Date: {date1}   Modality: {modality}\n"
        "Ordering Physician: Dr. {doctor}\n\n"
        "CLINICAL INDICATION: {indication}\n\n"
        "FINDINGS:\n{findings}\n\n"
        "IMPRESSION: {impression}\n"
    ),
    "progress_note": (
        "PROGRESS NOTE\n"
        "Patient: {name}   MRN: {mrn}   DOB: {dob}\n"
        "Date of Service: {date1}\n"
        "Provider: Dr. {doctor}\n\n"
        "SUBJECTIVE: {subjective}\n\n"
        "OBJECTIVE: Vitals stable. {objective}\n\n"
        "ASSESSMENT/PLAN: {plan}\n"
    ),
    "lab_report": (
        "LABORATORY REPORT\n"
        "Patient: {name}   MRN: {mrn}   DOB: {dob}\n"
        "Collected: {date1}   Resulted: {date2}\n"
        "Ordering Physician: Dr. {doctor}\n\n"
        "RESULTS:\n{results}\n\n"
        "INTERPRETATION: {interpretation}\n"
    ),
}

_HPI = [
    "Patient presents with {days}-day history of {symptom} and {symptom2}.",
    "Patient reports worsening {symptom} over the past week, denies {symptom2}.",
    "{age} year old {sex} with history of {condition} presenting with {symptom}.",
]
_SYMPTOMS = ["shortness of breath", "chest pain", "fatigue", "abdominal pain",
             "fever", "cough", "nausea", "dizziness", "lower back pain"]
_CONDITIONS = ["hypertension", "type 2 diabetes", "COPD", "atrial fibrillation",
               "chronic kidney disease", "asthma"]


def _rand_choice(seq):
    return random.choice(seq)


def _fill_note(doc_type: str) -> str:
    dob = fake.date_of_birth(minimum_age=18, maximum_age=95)
    ctx = {
        "name": fake.name(),
        "mrn": f"MRN-{random.randint(100000, 999999)}",
        "dob": dob.strftime("%m/%d/%Y"),
        "date1": fake.date_this_year().strftime("%m/%d/%Y"),
        "date2": fake.date_this_year().strftime("%m/%d/%Y"),
        "doctor": fake.last_name(),
        "age": 2026 - dob.year,
        "sex": _rand_choice(["male", "female"]),
        "symptom": _rand_choice(_SYMPTOMS),
        "symptom2": _rand_choice(_SYMPTOMS),
        "condition": _rand_choice(_CONDITIONS),
        "days": random.randint(1, 14),
        "modality": _rand_choice(["CT Chest", "MRI Brain", "X-Ray Abdomen", "Ultrasound Renal"]),
        "meds": ", ".join(fake.words(nb=3)).title() + " " + f"{random.randint(1,2)}0mg daily",
        "followup": f"Follow up with Dr. {fake.last_name()} in {random.randint(1,4)} weeks.",
    }
    ctx["hpi"] = _rand_choice(_HPI).format(**ctx)
    ctx["course"] = (f"Patient was treated with supportive care and improved. "
                      f"{ctx['condition'].capitalize()} was managed per protocol.")
    ctx["indication"] = f"Evaluate for {_rand_choice(_SYMPTOMS)}."
    ctx["findings"] = (f"No acute abnormality identified. Mild changes consistent with "
                        f"{_rand_choice(_CONDITIONS)}.")
    ctx["impression"] = _rand_choice(["No acute findings.", "Findings as described above.",
                                       "Recommend clinical correlation."])
    ctx["subjective"] = f"Patient reports {_rand_choice(_SYMPTOMS)}, improving since last visit."
    ctx["objective"] = f"No acute distress. {_rand_choice(_SYMPTOMS).capitalize()} not reproduced on exam."
    ctx["plan"] = f"Continue current management for {_rand_choice(_CONDITIONS)}. {ctx['followup']}"
    ctx["results"] = (f"WBC {round(random.uniform(4,11),1)}  Hgb {round(random.uniform(11,16),1)}  "
                       f"Na {random.randint(133,144)}  K {round(random.uniform(3.4,5.1),1)}  "
                       f"Creatinine {round(random.uniform(0.6,1.4),2)}")
    ctx["interpretation"] = _rand_choice(["Within normal limits.", "Mildly abnormal, correlate clinically.",
                                           "Critical value called to provider."])
    return TEMPLATES[doc_type].format(**ctx)


def _render_noisy_page(text: str, out_path: Path, seed: int) -> None:
    """Render text onto a page-like image with scan artifacts (rotation,
    blur, speckle noise) so pytesseract has to do real OCR work."""
    rng = random.Random(seed)
    width, height = 1000, 1300
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    y = 40
    for para in text.split("\n"):
        for line in textwrap.wrap(para, width=70) or [""]:
            draw.text((40, y), line, fill=0, font=font)
            y += 26
        y += 6

    angle = rng.uniform(-1.2, 1.2)
    img = img.rotate(angle, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))

    pixels = img.load()
    n_speckles = int(width * height * 0.002)
    for _ in range(n_speckles):
        x, yy = rng.randrange(width), rng.randrange(height)
        pixels[x, yy] = rng.choice([0, 255])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def generate(n_per_type: int = 40, seed: int = 42) -> Path:
    random.seed(seed)
    Faker.seed(seed)
    gt_path = settings.raw_dir / "ground_truth.jsonl"
    gt_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    counter = 0
    for doc_type in DOC_TYPES:
        for i in range(n_per_type):
            counter += 1
            text = _fill_note(doc_type)
            doc_id = f"{doc_type}_{i:04d}"
            img_path = settings.raw_dir / doc_type / f"{doc_id}.png"
            _render_noisy_page(text, img_path, seed=counter)
            records.append({
                "doc_id": doc_id,
                "doc_type": doc_type,
                "image_path": str(img_path.relative_to(settings.data_path)),
                "ground_truth_text": text,
            })

    with gt_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info("Generated %d synthetic scanned documents across %d types -> %s",
                len(records), len(DOC_TYPES), settings.raw_dir)
    return gt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic clinical document images.")
    parser.add_argument("--n-per-type", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(n_per_type=args.n_per_type, seed=args.seed)
