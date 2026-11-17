"""
Step 3 of the pipeline: turn the normalized records into a versioned,
split dataset — a tiny stand-in for a DVC/LakeFS-style dataset registry.

Each call to `build_version()` creates data/processed/v<N>/ containing
train.jsonl / val.jsonl / test.jsonl plus a manifest.json recording a
content hash, row counts, class balance, and the timestamp. Training
(models/train.py) records which dataset version it consumed so every model
run is traceable back to the exact data snapshot that produced it —
"reproducibility of experiments and releases" from the JD.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinical_mlops.config import settings
from clinical_mlops.utils.logging_config import get_logger

logger = get_logger(__name__)


def _hash_records(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _next_version() -> int:
    existing = [
        int(p.name[1:]) for p in settings.processed_dir.glob("v*") if p.name[1:].isdigit()
    ]
    return max(existing, default=0) + 1


def _split(records: list[dict[str, Any]], seed: int = 42,
           ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)):
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def build_version(normalized_path: Path | None = None, seed: int = 42) -> dict[str, Any]:
    normalized_path = normalized_path or (settings.interim_dir / "normalized.jsonl")
    if not normalized_path.exists():
        raise FileNotFoundError(f"{normalized_path} not found — run normalize.py first.")

    records = [json.loads(line) for line in normalized_path.read_text(encoding="utf-8").splitlines() if line]
    train, val, test = _split(records, seed=seed)

    version = _next_version()
    version_dir = settings.processed_dir / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    for name, split in (("train", train), ("val", val), ("test", test)):
        with (version_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")

    class_balance: dict[str, int] = {}
    for r in records:
        class_balance[r["doc_type"]] = class_balance.get(r["doc_type"], 0) + 1

    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(normalized_path.relative_to(settings.data_path)),
        "content_hash": _hash_records(records),
        "n_total": len(records),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "class_balance": class_balance,
        "seed": seed,
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Built dataset v%d: %d train / %d val / %d test -> %s",
                version, len(train), len(val), len(test), version_dir)
    return manifest


def latest_version() -> int | None:
    versions = [int(p.name[1:]) for p in settings.processed_dir.glob("v*") if p.name[1:].isdigit()]
    return max(versions) if versions else None


def load_split(version: int, split: str) -> list[dict[str, Any]]:
    path = settings.processed_dir / f"v{version}" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No such dataset split: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_manifest(version: int) -> dict[str, Any]:
    path = settings.processed_dir / f"v{version}" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_versions() -> list[dict[str, Any]]:
    return [
        load_manifest(int(p.name[1:]))
        for p in sorted(settings.processed_dir.glob("v*"))
        if p.name[1:].isdigit()
    ]


if __name__ == "__main__":
    build_version()
