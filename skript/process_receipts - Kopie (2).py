#!/usr/bin/env python3
"""
process_receipts.py

OCR-Pipeline mit expliziter Spaltenprojektion:

- Vision OCR auf Wortebene
- Header dient als Spaltenanker (X-Positionen)
- Wörter werden zeilen- UND spaltenweise zugeordnet
- Ausgabe als echte textuelle Tabelle (| getrennt)
- LLM erhält NUR diese Tabelle
- Debug-Ausgaben:
    - finaler Prompt
    - finale Tabellenrepräsentation
- Bei Fehlern: ABBRUCH
"""

import csv
import io
import json
import os
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from google.cloud import vision
from openai import OpenAI
from PIL import Image, ImageOps


# =========================================================
# PFADE
# =========================================================

BASE_DIR = Path(r"C:\Users\micha\Dropbox\Apps\Server Kassenzettel")

INCOMING_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
CSV_DIR = BASE_DIR / "csv"

SCRIPT_DIR = BASE_DIR / "skript"
DEBUG_DIR = SCRIPT_DIR / "debug_prompts"

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
# UTILS
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
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l]


# =========================================================
# RECEIPT ID
# =========================================================

def ensure_registry():
    if not REGISTRY_FILE.exists():
        with REGISTRY_FILE.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                ["assigned_at", "receipt_id", "original_filename"]
            )


def get_next_receipt_id(filename: str) -> str:
    ensure_registry()
    last_id = 0
    with REGISTRY_FILE.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = re.match(r"R(\d{6})", r["receipt_id"])
            if m:
                last_id = max(last_id, int(m.group(1)))
    new_id = f"R{last_id + 1:06d}"
    with REGISTRY_FILE.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([utc_now(), new_id, filename])
    return new_id


# =========================================================
# OCR
# =========================================================

def setup_clients():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(VISION_KEY_FILE)
    return (
        vision.ImageAnnotatorClient(),
        OpenAI(api_key=OPENAI_KEY_FILE.read_text().strip()),
    )


def normalize_image(image_path: Path) -> bytes:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ocr_words(client, png_bytes: bytes) -> list[dict]:
    r = client.text_detection(image=vision.Image(content=png_bytes))
    words = []
    for a in r.text_annotations[1:]:
        xs = [v.x for v in a.bounding_poly.vertices]
        ys = [v.y for v in a.bounding_poly.vertices]
        words.append({
            "text": a.description,
            "x": sum(xs) / 4,
            "y": sum(ys) / 4,
            "x_min": min(xs),
            "x_max": max(xs),
            "y_min": min(ys),
            "y_max": max(ys),
            "height": max(ys) - min(ys),
        })
    if not words:
        raise RuntimeError("OCR leer")
    return words


# =========================================================
# SPALTENPROJEKTION
# =========================================================

def cluster_lines(words: list[dict]) -> list[list[dict]]:
    heights = [w["height"] for w in words if w["height"] > 0]
    h = statistics.median(heights)
    threshold = h * 0.6

    words = sorted(words, key=lambda w: (w["y"], w["x"]))
    lines = []
    current = []

    for w in words:
        if not current:
            current = [w]
            continue
        if abs(w["y"] - current[0]["y"]) <= threshold:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
    if current:
        lines.append(current)
    return lines


def detect_header_line(lines: list[list[dict]]) -> int:
    for i, line in enumerate(lines[:10]):
        texts = " ".join(w["text"].lower() for w in line)
        if any(k in texts for k in ["artikel", "bezeich", "menge", "preis", "total", "betrag"]):
            return i
    raise RuntimeError("Header nicht gefunden")


def build_columns(header_line: list[dict]) -> list[float]:
    cols = sorted(w["x"] for w in header_line)
    return cols


def assign_to_columns(words: list[dict], cols: list[float]) -> list[str]:
    cells = [""] * len(cols)
    for w in words:
        idx = min(range(len(cols)), key=lambda i: abs(w["x"] - cols[i]))
        cells[idx] = (cells[idx] + " " + w["text"]).strip()
    return cells


def build_table(words: list[dict]) -> list[str]:
    lines = cluster_lines(words)
    header_idx = detect_header_line(lines)
    header = lines[header_idx]
    cols = build_columns(header)

    table = []
    header_cells = assign_to_columns(header, cols)
    table.append(" | ".join(header_cells))
    table.append("-" * len(table[0]))

    for line in lines[header_idx + 1:]:
        cells = assign_to_columns(line, cols)
        if any(cells):
            table.append(" | ".join(cells))

    return table


# =========================================================
# LLM
# =========================================================

def load_prompt() -> str:
    p = PROMPT_FILE.read_text(encoding="utf-8")
    p = p.replace(
        "{{ALLOWED_CATEGORIES}}",
        "\n".join(f"- {c}" for c in read_lines_strict(CATEGORIES_FILE)),
    )
    p = p.replace(
        "{{ALLOWED_MERCHANTS}}",
        "\n".join(f"- {m}" for m in read_lines_strict(MERCHANTS_FILE)),
    )
    if "{{" in p:
        raise RuntimeError("Prompt-Platzhalter offen")
    return p


def parse_with_llm(client, prompt: str, table_lines: list[str]) -> dict:
    content = "\n".join(table_lines)
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Du bist ein Kassenzettel-Parser."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": content},
        ],
    )
    return json.loads(r.choices[0].message.content)


# =========================================================
# MAIN
# =========================================================

def process_image(img: Path, vision_client, openai_client, prompt: str):
    rid = get_next_receipt_id(img.name)
    print(f"[{rid}] Start")

    png = normalize_image(img)
    words = ocr_words(vision_client, png)
    table = build_table(words)

    DEBUG_DIR.mkdir(exist_ok=True)
    (DEBUG_DIR / f"{rid}_prompt.txt").write_text(prompt, encoding="utf-8")
    (DEBUG_DIR / f"{rid}_table.txt").write_text("\n".join(table), encoding="utf-8")

    parsed = parse_with_llm(openai_client, prompt, table)

    receipt = parsed["receipt"]
    items = parsed["items"]

    rows = [CSV_HEADER]
    rows.append([
        rid, "", "", "", "", 0, "", "", "", "",
        safe_float(receipt.get("receipt_total"))
    ])

    idx = 1
    for it in items:
        rows.append([
            rid,
            f"{rid}{img.suffix.lower()}",
            utc_now(),
            receipt.get("merchant", ""),
            normalize_date(receipt.get("date")),
            idx,
            it.get("name", ""),
            it.get("category", ""),
            it.get("quantity", 1),
            "",
            safe_float(it.get("total_price")),
        ])
        idx += 1

    CSV_DIR.mkdir(exist_ok=True)
    with (CSV_DIR / f"{rid}.csv").open("w", encoding="utf-16", newline="") as f:
        csv.writer(f, delimiter=";").writerows(rows)

    PROCESSED_DIR.mkdir(exist_ok=True)
    shutil.move(str(img), PROCESSED_DIR / f"{rid}{img.suffix.lower()}")

    print(f"[{rid}] Fertig")


def main():
    vc, oc = setup_clients()
    prompt = load_prompt()
    imgs = [p for p in INCOMING_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not imgs:
        print("Keine Dateien")
        return
    for img in imgs:
        process_image(img, vc, oc, prompt)


if __name__ == "__main__":
    main()
