#!/usr/bin/env python3
"""
process_receipts.py

- Jede Datei im uploads-Ordner wird IMMER verarbeitet
- Jede Verarbeitung bekommt IMMER eine neue globale receipt_id
- EXIF-Normalisierung vor OCR (kritisch für stabile Koordinaten)
- Prompt, Kategorien und Händler extern ausgelagert
- Statusmeldungen für CMD / Excel
- CSV / Excel abwärtskompatibel
"""

import csv
import io
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from google.cloud import vision
from openai import OpenAI
from PIL import Image, ImageOps


# =========================================================
# PFADE / DATEIEN
# =========================================================

BASE_DIR = Path(r"C:\Users\micha\Dropbox\Apps\Server Kassenzettel")

INCOMING_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
CSV_DIR = BASE_DIR / "csv"

SCRIPT_DIR = BASE_DIR / "skript"

PROMPT_FILE = SCRIPT_DIR / "prompt.txt"
CATEGORIES_FILE = SCRIPT_DIR / "allowed_categories.txt"
MERCHANTS_FILE = SCRIPT_DIR / "allowed_merchants.txt"

REGISTRY_FILE = SCRIPT_DIR / "receipt_id_registry.csv"

VISION_KEY_FILE = SCRIPT_DIR / "vision_key.json"
OPENAI_KEY_FILE = SCRIPT_DIR / "openai_key.txt"

OPENAI_MODEL = "gpt-4o-mini"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


# =========================================================
# CSV
# =========================================================

CSV_HEADER = [
    "receipt_id",
    "source_image",
    "imported_at",
    "merchant",
    "receipt_date",
    "item_index",
    "name",
    "category",
    "quantity",
    "unit_price",
    "total_price",
]


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    m = re.match(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})", s)
    if m:
        d, mth, y = m.groups()
        if len(y) == 2:
            y = "20" + y
        return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
    if re.match(r"\d{4}-\d{2}-\d{2}", s):
        return s
    return None


def safe_float(v: Any) -> Optional[float]:
    try:
        return float(str(v).replace(",", "."))
    except Exception:
        return None


def read_lines_strict(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"Fehlende Datei: {path}")
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l]


# =========================================================
# ID-REGISTRY
# =========================================================

def ensure_registry():
    if not REGISTRY_FILE.exists():
        REGISTRY_FILE.parent.mkdir(exist_ok=True)
        with REGISTRY_FILE.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                ["assigned_at", "receipt_id", "original_filename"]
            )


def get_next_receipt_id(original_filename: str) -> str:
    ensure_registry()

    last_id = 0
    with REGISTRY_FILE.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.match(r"R(\d{6})", row["receipt_id"])
            if m:
                last_id = max(last_id, int(m.group(1)))

    new_id = f"R{last_id + 1:06d}"

    with REGISTRY_FILE.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([utc_now(), new_id, original_filename])

    return new_id


# =========================================================
# OCR – EXIF-NORMALISIERT
# =========================================================

def setup_clients():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(VISION_KEY_FILE)
    return (
        vision.ImageAnnotatorClient(),
        OpenAI(api_key=OPENAI_KEY_FILE.read_text().strip()),
    )


def normalize_image_for_ocr(image_path: Path) -> bytes:
    """
    Öffnet ein Bild, wendet EXIF-Orientation korrekt an
    und liefert PNG-Bytes der normalisierten Pixel.
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ocr_debug_text(client, normalized_png_bytes: bytes) -> str:
    image = vision.Image(content=normalized_png_bytes)
    r = client.text_detection(image=image)

    lines = []
    for page in r.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                words = []
                for w in para.words:
                    txt = "".join(s.text for s in w.symbols)
                    words.append((txt, w.bounding_box))
                if not words:
                    continue

                y = sum(v.y for _, b in words for v in b.vertices) / (4 * len(words))
                x_min = min(v.x for _, b in words for v in b.vertices)
                x_max = max(v.x for _, b in words for v in b.vertices)
                text = " ".join(t for t, _ in words)

                lines.append(f"Y={y:.1f} X=[{x_min}-{x_max}] {text}")

    return "\n".join(lines)


# =========================================================
# LLM
# =========================================================

def load_prompt() -> str:
    prompt = PROMPT_FILE.read_text(encoding="utf-8")

    categories = read_lines_strict(CATEGORIES_FILE)
    merchants = read_lines_strict(MERCHANTS_FILE)

    prompt = prompt.replace(
        "{{ALLOWED_CATEGORIES}}",
        "\n".join(f"- {c}" for c in categories),
    )
    prompt = prompt.replace(
        "{{ALLOWED_MERCHANTS}}",
        "\n".join(f"- {m}" for m in merchants),
    )

    if "{{" in prompt or "}}" in prompt:
        raise RuntimeError("Nicht ersetzte Platzhalter im Prompt")

    return prompt


def parse_with_llm(client: OpenAI, prompt: str, ocr_text: str) -> dict:
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Du bist ein Kassenzettel-Parser."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": ocr_text},
        ],
    )
    return json.loads(r.choices[0].message.content)


# =========================================================
# HAUPTLOGIK
# =========================================================

def process_image(img: Path, vision_client, openai_client, prompt: str):
    receipt_id = get_next_receipt_id(img.name)
    print(f"[{receipt_id}] Verarbeitung startet")

    print(f"[{receipt_id}] Bild-Normalisierung …")
    normalized_png = normalize_image_for_ocr(img)

    print(f"[{receipt_id}] OCR läuft …")
    ocr_text = ocr_debug_text(vision_client, normalized_png)

    print(f"[{receipt_id}] LLM läuft …")
    parsed = parse_with_llm(openai_client, prompt, ocr_text)

    receipt = parsed.get("receipt", {})
    merchant = receipt.get("merchant", "")
    receipt_date = normalize_date(receipt.get("date"))
    receipt_total = safe_float(receipt.get("receipt_total"))

    rows = [CSV_HEADER]

    rows.append([
        receipt_id, "", "", "", "", 0, "", "", "", "",
        receipt_total if receipt_total is not None else ""
    ])

    idx = 1
    for item in parsed.get("items", []):
        rows.append([
            receipt_id,
            f"{receipt_id}{img.suffix.lower()}",
            utc_now(),
            merchant,
            receipt_date or "",
            idx,
            item.get("name", ""),
            item.get("category", ""),
            item.get("quantity", 1),
            "",
            safe_float(item.get("total_price")),
        ])
        idx += 1

    CSV_DIR.mkdir(exist_ok=True)
    csv_path = CSV_DIR / f"{receipt_id}.csv"
    with csv_path.open("w", encoding="utf-16", newline="") as f:
        csv.writer(f, delimiter=";", lineterminator="\r\n").writerows(rows)

    PROCESSED_DIR.mkdir(exist_ok=True)
    shutil.move(str(img), PROCESSED_DIR / f"{receipt_id}{img.suffix.lower()}")

    print(f"[{receipt_id}] CSV geschrieben")
    print(f"[{receipt_id}] Fertig\n")


def main():
    vision_client, openai_client = setup_clients()
    prompt = load_prompt()

    images = [p for p in INCOMING_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    print(f"{len(images)} Datei(en) im Upload-Ordner gefunden")

    for img in images:
        process_image(img, vision_client, openai_client, prompt)

    print("Alle Kassenzettel verarbeitet.")


if __name__ == "__main__":
    main()
