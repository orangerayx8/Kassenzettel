import io
import math
import statistics
from typing import Any, Dict, List, Tuple

from google.cloud import vision
from PIL import Image, ImageOps

from config import DESKEW_MAX_ABS_DEG, DESKEW_MIN_ABS_DEG


def load_image_rgb(path) -> Image.Image:
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
