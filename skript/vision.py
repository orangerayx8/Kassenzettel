#!/usr/bin/env python3
"""
vision_token_test.py

Testet:
- Vision-LLM mit echtem Kassenzettelbild
- realen Prompt
- misst tatsächlich verbrauchte Tokens
"""

import base64
from pathlib import Path
from openai import OpenAI


# =========================================================
# KONFIGURATION
# =========================================================

OPENAI_API_KEY_FILE = Path(r"C:\Users\micha\Dropbox\Apps\Server Kassenzettel\skript\openai_key.txt")

IMAGE_PATH = Path(r"C:\Users\micha\Dropbox\Apps\Server Kassenzettel\uploads\TEST_RECEIPT.jpg")

MODEL = "gpt-4o-mini"   # Vision-fähig, günstiger als full gpt-4o


# =========================================================
# HILFSFUNKTIONEN
# =========================================================

def load_api_key() -> str:
    return OPENAI_API_KEY_FILE.read_text(encoding="utf-8").strip()


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


# =========================================================
# MAIN
# =========================================================

def main():
    client = OpenAI(api_key=load_api_key())

    image_b64 = image_to_base64(IMAGE_PATH)

    prompt = """
Du siehst einen Kassenzettel.

AUFGABE:
- Lies den Kassenzettel vollständig
- Extrahiere alle gekauften Artikel
- Gib für jeden Artikel aus:
  - name
  - quantity
  - total_price

Antworte NUR als JSON.
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
    )

    print("=== MODELL ===")
    print(MODEL)
    print()

    print("=== TOKEN VERBRAUCH ===")
    usage = response.usage
    print(f"Prompt Tokens     : {usage.prompt_tokens}")
    print(f"Completion Tokens : {usage.completion_tokens}")
    print(f"TOTAL Tokens      : {usage.total_tokens}")
    print()

    print("=== MODELL ANTWORT ===")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
