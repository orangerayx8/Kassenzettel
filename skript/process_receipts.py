#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import io
import json
import math
import os
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from google.cloud import vision
from openai import OpenAI
from PIL import Image, ImageOps


# =========================================================
# PFADE / KONFIG
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

DEBUG_TABLE_DIR = SCRIPT_DIR / "debug_tables"
DEBUG_OCR_DIR = SCRIPT_DIR / "debug_vision_ocr"
DEBUG_LLM_DIR = SCRIPT_DIR / "debug_llm_raw"

OPENAI_MODEL = "gpt-4o-mini"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

DESKEW_MIN_ABS_DEG = 0.4
DESKEW_MAX_ABS_DEG = 12.0

LINE_Y_THRESHOLD_MULT = 0.60
X_GAP_MULT = 1.8
MIN_BUCKETS = 4
MAX_BUCKETS = 16


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

def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ensure_dirs() -> None:
    for d in [
        CSV_DIR, PROCESSED_DIR,
        DEBUG_TABLE_DIR, DEBUG_OCR_DIR, DEBUG_LLM_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# RECEIPT ID
# =========================================================

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


# =========================================================
# CLIENTS
# =========================================================

def setup_clients() -> Tuple[Any, Any]:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(VISION_KEY_FILE)
    return (
        vision.ImageAnnotatorClient(),
        OpenAI(api_key=OPENAI_KEY_FILE.read_text(encoding="utf-8").strip()),
    )


# =========================================================
# IMAGE / OCR
# =========================================================

def load_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def rotate_image(img: Image.Image, angle_deg: float) -> Image.Image:
    return img.rotate(-angle_deg, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def vision_ocr_words(client: Any, img: Image.Image) -> List[Dict[str, Any]]:
    png = image_to_png_bytes(img)
    result = client.text_detection(image=vision.Image(content=png))

    words = []
    for ann in result.text_annotations[1:]:
        verts = ann.bounding_poly.vertices
        xs = [v.x for v in verts]
        ys = [v.y for v in verts]

        words.append({
            "text": ann.description,
            "center": {
                "x": sum(xs) / 4.0,
                "y": sum(ys) / 4.0,
            },
            "size": {
                "w": max(xs) - min(xs),
                "h": max(ys) - min(ys),
            },
            "vertices": [{"x": v.x, "y": v.y} for v in verts],
        })

    if not words:
        raise RuntimeError("Vision OCR leer")

    return words


def estimate_skew(words: List[Dict[str, Any]]) -> float:
    angles = []
    for w in words:
        v = w["vertices"]
        if len(v) >= 2:
            dx = v[1]["x"] - v[0]["x"]
            dy = v[1]["y"] - v[0]["y"]
            if abs(dx) > 1e-3:
                ang = math.degrees(math.atan2(dy, dx))
                if abs(ang) <= DESKEW_MAX_ABS_DEG:
                    angles.append(ang)
    return statistics.median(angles) if angles else 0.0


def two_pass_ocr(client: Any, img: Image.Image) -> Tuple[Image.Image, List[Dict[str, Any]], float]:
    words1 = vision_ocr_words(client, img)
    angle = estimate_skew(words1)

    if abs(angle) < DESKEW_MIN_ABS_DEG:
        return img, words1, 0.0

    img2 = rotate_image(img, angle)
    words2 = vision_ocr_words(client, img2)
    return img2, words2, angle


# =========================================================
# GEOMETRIE
# =========================================================

def cluster_lines(words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    heights = [w["size"]["h"] for w in words if w["size"]["h"] > 0]
    med_h = statistics.median(heights) if heights else 10
    thr = med_h * LINE_Y_THRESHOLD_MULT

    words_sorted = sorted(words, key=lambda w: (w["center"]["y"], w["center"]["x"]))
    lines = []
    current = []

    for w in words_sorted:
        if not current or abs(w["center"]["y"] - current[0]["center"]["y"]) <= thr:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
    if current:
        lines.append(current)

    for ln in lines:
        ln.sort(key=lambda w: w["center"]["x"])

    return lines


def build_x_buckets(words: List[Dict[str, Any]]) -> List[float]:
    xs = sorted(w["center"]["x"] for w in words)
    if not xs:
        return []

    widths = [w["size"]["w"] for w in words if w["size"]["w"] > 0]
    med_w = statistics.median(widths) if widths else 20
    gap = med_w * X_GAP_MULT

    clusters = [[xs[0]]]
    for x in xs[1:]:
        if abs(x - clusters[-1][-1]) <= gap:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    centers = [sum(c) / len(c) for c in clusters]
    centers.sort()

    while len(centers) > MAX_BUCKETS:
        i = min(range(len(centers) - 1), key=lambda j: centers[j + 1] - centers[j])
        centers = centers[:i] + [(centers[i] + centers[i + 1]) / 2] + centers[i + 2:]

    if len(centers) < MIN_BUCKETS:
        mn, mx = min(xs), max(xs)
        step = (mx - mn) / (MIN_BUCKETS - 1) if mx > mn else 1
        centers = [mn + i * step for i in range(MIN_BUCKETS)]

    return centers


def assign_bucket(x: float, centers: List[float]) -> int:
    return min(range(len(centers)), key=lambda i: abs(x - centers[i]))


def build_llm_table(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = build_x_buckets(words)
    lines = cluster_lines(words)

    rows = []
    for i, ln in enumerate(lines):
        cells = {}
        for w in ln:
            b = assign_bucket(w["center"]["x"], buckets)
            cells.setdefault(b, []).append(w["text"])

        rows.append({
            "row_id": i,
            "y_center": round(sum(w["center"]["y"] for w in ln) / len(ln), 2),
            "cells": [
                {"x_bucket": b, "text": " ".join(cells[b])}
                for b in sorted(cells)
            ],
        })

    return {
        "x_buckets": [
            {"x_bucket": i, "x_center": round(c, 2)}
            for i, c in enumerate(buckets)
        ],
        "rows": rows,
    }


def table_to_txt(table: Dict[str, Any]) -> str:
    out = []
    for r in table["rows"]:
        parts = [f"ROW {r['row_id']}"]
        for c in r["cells"]:
            parts.append(f"X{c['x_bucket']}: {c['text']}")
        out.append(" | ".join(parts))
    return "\n".join(out)


# =========================================================
# LLM
# =========================================================

def load_prompt() -> str:
    p = PROMPT_FILE.read_text(encoding="utf-8")
    p = p.replace(
        "{{ALLOWED_CATEGORIES}}",
        "\n".join(f"- {c}" for c in read_lines(CATEGORIES_FILE)),
    )
    p = p.replace(
        "{{ALLOWED_MERCHANTS}}",
        "\n".join(f"- {m}" for m in read_lines(MERCHANTS_FILE)),
    )
    if "{{" in p:
        raise RuntimeError("Offene Prompt-Platzhalter")
    return p


def call_llm_and_debug(
    client: Any,
    receipt_id: str,
    prompt: str,
    table: Dict[str, Any],
) -> Dict[str, Any]:

    payload = json.dumps(table, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Du bist ein Kassenzettel-Interpreter."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": payload},
        ],
    )

    raw = resp.choices[0].message.content

    # >>> NEU: rohe LLM-Antwort speichern <<<
    (DEBUG_LLM_DIR / f"{receipt_id}_llm_raw.txt").write_text(
        raw, encoding="utf-8"
    )

    return json.loads(raw)


# =========================================================
# CSV
# =========================================================

def write_csv(receipt_id: str, source_image: str, parsed: Dict[str, Any]) -> None:
    out = CSV_DIR / f"{receipt_id}.csv"
    rows = [CSV_HEADER]

    receipt = parsed.get("receipt", {})
    items = parsed.get("items", [])

    rows.append([
        receipt_id, "", "", "", "",
        0, "", "", "", "",
        receipt.get("receipt_total"),
    ])

    idx = 1
    for it in items:
        rows.append([
            receipt_id,
            source_image,
            utc_now(),
            receipt.get("merchant", ""),
            receipt.get("date", ""),
            idx,
            it.get("name", ""),
            it.get("category", ""),
            it.get("quantity", 1),
            "",
            it.get("total_price"),
        ])
        idx += 1

    with out.open("w", encoding="utf-16", newline="") as f:
        csv.writer(f, delimiter=";").writerows(rows)


# =========================================================
# PROCESS
# =========================================================

def process_image(
    img_path: Path,
    vision_client: Any,
    openai_client: Any,
    prompt: str,
) -> None:

    rid = next_receipt_id(img_path.name)
    info(f"{rid}: Start")

    img = load_image_rgb(img_path)
    img2, words, angle = two_pass_ocr(vision_client, img)

    table = build_llm_table(words)

    (DEBUG_OCR_DIR / f"{rid}_vision_words.json").write_text(
        json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DEBUG_TABLE_DIR / f"{rid}_table.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DEBUG_TABLE_DIR / f"{rid}_table.txt").write_text(
        table_to_txt(table), encoding="utf-8"
    )

    info(f"{rid}: LLM läuft")
    parsed = call_llm_and_debug(openai_client, rid, prompt, table)

    write_csv(rid, f"{rid}{img_path.suffix.lower()}", parsed)

    shutil.move(str(img_path), PROCESSED_DIR / f"{rid}{img_path.suffix.lower()}")

    info(f"{rid}: Fertig")


def main() -> None:
    ensure_dirs()
    vision_client, openai_client = setup_clients()
    prompt = load_prompt()

    imgs = [p for p in INCOMING_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    for p in imgs:
        process_image(p, vision_client, openai_client, prompt)


if __name__ == "__main__":
    main()
