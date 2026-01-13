from datetime import datetime, timezone
from typing import List

from config import (
    CSV_DIR,
    DEBUG_LLM_DIR,
    DEBUG_OCR_DIR,
    DEBUG_TABLE_DIR,
    PROCESSED_DIR,
    SCRIPT_DIR,
)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_lines(path) -> List[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ensure_dirs() -> None:
    for d in [
        CSV_DIR,
        PROCESSED_DIR,
        DEBUG_TABLE_DIR,
        DEBUG_OCR_DIR,
        DEBUG_LLM_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
