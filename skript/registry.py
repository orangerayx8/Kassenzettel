import csv
import re

from config import REGISTRY_FILE
from utils import utc_now


def ensure_registry() -> None:
    if not REGISTRY_FILE.exists():
        with REGISTRY_FILE.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                ["assigned_at", "receipt_id", "original_filename"]
            )


def next_receipt_id(filename: str) -> str:
    ensure_registry()
    last_id = 0
    with REGISTRY_FILE.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = re.match(r"R(\d{6})$", (r.get("receipt_id") or ""))
            if m:
                last_id = max(last_id, int(m.group(1)))
    rid = f"R{last_id + 1:06d}"
    with REGISTRY_FILE.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([utc_now(), rid, filename])
    return rid
