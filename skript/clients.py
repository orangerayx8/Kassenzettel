import os
from typing import Any, Tuple

from google.cloud import vision
from openai import OpenAI

from config import OPENAI_KEY_FILE, VISION_KEY_FILE


def setup_clients() -> Tuple[Any, Any]:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(VISION_KEY_FILE)
    return (
        vision.ImageAnnotatorClient(),
        OpenAI(api_key=OPENAI_KEY_FILE.read_text(encoding="utf-8").strip()),
    )
