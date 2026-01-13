import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont, ImageOps


# =========================================================
# PFADE
# =========================================================

BASE_DIR = Path(r"C:\Users\micha\Dropbox\Apps\Server Kassenzettel")

INPUT_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "ocr_canvas"

OCR_JSON_DIR = OUTPUT_DIR / "ocr_json"
CANVAS_DIR = OUTPUT_DIR / "canvas"

VISION_KEY_FILE = BASE_DIR / "skript" / "vision_key.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


# =========================================================
# SETUP
# =========================================================

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(VISION_KEY_FILE)

OCR_JSON_DIR.mkdir(parents=True, exist_ok=True)
CANVAS_DIR.mkdir(parents=True, exist_ok=True)

client = vision.ImageAnnotatorClient()

try:
    FONT = ImageFont.truetype("arial.ttf", 14)
except OSError:
    FONT = ImageFont.load_default()


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def normalize_image_for_ocr(image_path: Path) -> Tuple[Image.Image, bytes]:
    """
    Öffnet ein Bild, wendet EXIF-Orientation korrekt an (exif_transpose),
    und liefert:
      - PIL Image (RGB) in 'sichtbarer' Orientierung
      - Bytes (PNG) exakt dieser normalisierten Pixel (für Vision)
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")  # PNG = stabile, EXIF-freie Pixel
    return img, buf.getvalue()


def bbox_min_xy(vertices: List[Dict[str, int]]) -> Tuple[int, int]:
    """
    Nimmt 4 vertices (x,y) und gibt robust den linken oberen Anker zurück.
    (Vision garantiert nicht immer eine perfekte Reihenfolge der Punkte.)
    """
    xs = [v.get("x", 0) for v in vertices]
    ys = [v.get("y", 0) for v in vertices]
    return int(min(xs)), int(min(ys))


def bbox_height(vertices: List[Dict[str, int]]) -> int:
    ys = [v.get("y", 0) for v in vertices]
    return int(max(ys) - min(ys))


def pick_font_for_box(base_font: ImageFont.ImageFont, box_h: int) -> ImageFont.ImageFont:
    """
    Optional: font grob an Boxhöhe anpassen (damit Canvas näher am Original wirkt).
    Wenn arial.ttf nicht verfügbar ist, bleibt default font.
    """
    # Sehr konservativ, weil Receipt-Fonts schmal sind.
    target = max(10, min(28, int(box_h * 0.6)))
    try:
        return ImageFont.truetype("arial.ttf", target)
    except OSError:
        return base_font


# =========================================================
# OCR (Wörter + Koordinaten)
# =========================================================

def ocr_image_words(normalized_png_bytes: bytes) -> Dict:
    """
    Nutzt Vision text_detection und extrahiert Wörter mit bounding_poly.
    Wir verwenden weiterhin text_annotations[1:], weil du explizit "Wörter" willst.
    (full_text_annotation wäre auch möglich, aber hier minimal-invasiv.)
    """
    vimg = vision.Image(content=normalized_png_bytes)
    response = client.text_detection(image=vimg)

    if response.error.message:
        raise RuntimeError(response.error.message)

    annotations = response.text_annotations

    words = []
    for ann in annotations[1:]:
        vertices = [{"x": int(v.x), "y": int(v.y)} for v in ann.bounding_poly.vertices]
        words.append({
            "text": ann.description,
            "bounding_box": vertices
        })

    return {"words": words}


# =========================================================
# CANVAS RENDERING
# =========================================================

def draw_canvas(normalized_img: Image.Image, ocr_data: Dict, output_path: Path) -> None:
    """
    Legt einen weißen Canvas in exakt gleicher Größe wie das normalisierte Bild an
    und zeichnet jedes Wort am (min_x, min_y) der Bounding Box.
    """
    canvas = Image.new("RGB", normalized_img.size, "white")
    draw = ImageDraw.Draw(canvas)

    for w in ocr_data.get("words", []):
        vertices = w.get("bounding_box", [])
        if len(vertices) != 4:
            continue

        x, y = bbox_min_xy(vertices)
        h = bbox_height(vertices)
        font = pick_font_for_box(FONT, h)

        draw.text((x, y), w.get("text", ""), fill="black", font=font)

    canvas.save(output_path)


# =========================================================
# MAIN
# =========================================================

def process_all_images() -> None:
    for img_path in INPUT_DIR.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        print(f"PROCESS: {img_path.name}")

        normalized_img, normalized_bytes = normalize_image_for_ocr(img_path)
        ocr_result = ocr_image_words(normalized_bytes)

        # Metadaten dazu (hilft beim Debuggen)
        ocr_result["image"] = img_path.name
        ocr_result["normalized_size"] = {"width": normalized_img.size[0], "height": normalized_img.size[1]}

        json_path = OCR_JSON_DIR / f"{img_path.stem}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)

        canvas_path = CANVAS_DIR / f"{img_path.stem}_canvas.png"
        draw_canvas(normalized_img, ocr_result, canvas_path)


if __name__ == "__main__":
    process_all_images()
