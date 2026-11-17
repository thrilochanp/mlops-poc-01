"""Structured (JSON-lines) logging shared by the API, pipeline, and CLIs.

Kept intentionally small: one function that returns a configured logger, and
one helper for appending JSONL "events" (used by monitoring/logger.py for
prediction logs). Real deployments would ship these lines to something like
CloudWatch/Elastic; here they land on disk so the monitoring dashboard can
read them straight back.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object per line, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
