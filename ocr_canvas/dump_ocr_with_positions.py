from pathlib import Path
import os
from collections import defaultdict

from google.cloud import vision


# =========================================================
# KONFIGURATION
# =========================================================

BASE_DIR = Path(r"C:\Users\micha\Dropbox\Apps\Server Kassenzettel")
INCOMING_DIR = BASE_DIR / "uploads"
OUT_DIR = BASE_DIR / "ocr_debug_pos"
VISION_KEY_FILE = BASE_DIR / "skript" / "vision_key.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(VISION_KEY_FILE)

OUT_DIR.mkdir(exist_ok=True)

client = vision.ImageAnnotatorClient()


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def bbox_center_y(bbox):
    ys = [v.y for v in bbox.vertices]
    return sum(ys) / len(ys)


def bbox_min_x(bbox):
    return min(v.x for v in bbox.vertices)


def bbox_max_x(bbox):
    return max(v.x for v in bbox.vertices)


# =========================================================
# HAUPTLOGIK
# =========================================================

for img in INCOMING_DIR.iterdir():
    if img.suffix.lower() not in IMAGE_EXTS:
        continue

    print(f"Verarbeite {img.name}")

    image = vision.Image(content=img.read_bytes())
    response = client.text_detection(image=image)

    if not response.full_text_annotation:
        print("  Keine full_text_annotation")
        continue

    lines = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:

                words = []
                y_positions = []

                for word in para.words:
                    text = "".join(s.text for s in word.symbols)
                    words.append((text, word.bounding_box))
                    y_positions.append(bbox_center_y(word.bounding_box))

                if not words:
                    continue

                line_text = " ".join(w[0] for w in words)
                avg_y = sum(y_positions) / len(y_positions)

                min_x = min(bbox_min_x(w[1]) for w in words)
                max_x = max(bbox_max_x(w[1]) for w in words)

                lines.append({
                    "y": round(avg_y, 1),
                    "x_min": min_x,
                    "x_max": max_x,
                    "text": line_text,
                })

    # nach Y sortieren (oben → unten)
    lines.sort(key=lambda l: l["y"])

    out_txt = OUT_DIR / f"{img.stem}_lines.txt"

    with out_txt.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(
                f"Y={l['y']:>7}  "
                f"X=[{l['x_min']:>4}-{l['x_max']:<4}]  "
                f"{l['text']}\n"
            )

    print(f"  → gespeichert: {out_txt.name}")
