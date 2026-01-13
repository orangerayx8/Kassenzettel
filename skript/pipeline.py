import json
import shutil
from pathlib import Path
from typing import Any

from clients import setup_clients
from config import (
    DEBUG_OCR_DIR,
    DEBUG_TABLE_DIR,
    IMAGE_EXTS,
    INCOMING_DIR,
    PROCESSED_DIR,
)
from csv_export import write_csv
from geometry import build_llm_table, table_to_txt
from llm import call_llm_and_debug, load_prompt
from ocr import load_image_rgb, two_pass_ocr
from registry import next_receipt_id
from utils import ensure_dirs, info


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
