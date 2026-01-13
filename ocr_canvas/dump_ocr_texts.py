from pathlib import Path
import os
from google.cloud import vision

BASE_DIR = Path(r"C:\Users\micha\Dropbox\Apps\Server Kassenzettel")
INCOMING_DIR = BASE_DIR / "uploads"
OUT_DIR = BASE_DIR / "ocr_debug"
VISION_KEY_FILE = BASE_DIR / "skript" / "vision_key.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(VISION_KEY_FILE)

OUT_DIR.mkdir(exist_ok=True)

client = vision.ImageAnnotatorClient()

for img in INCOMING_DIR.iterdir():
    if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"]:
        continue

    content = img.read_bytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    text = ""
    if response.text_annotations:
        text = response.text_annotations[0].description

    out_file = OUT_DIR / f"{img.stem}.txt"
    out_file.write_text(text, encoding="utf-8")

    print(f"OCR gespeichert: {out_file.name}")
