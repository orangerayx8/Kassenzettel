import json
from typing import Any, Dict

from config import (
    CATEGORIES_FILE,
    DEBUG_LLM_DIR,
    MERCHANTS_FILE,
    OPENAI_MODEL,
    PROMPT_FILE,
)
from utils import read_lines


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

    (DEBUG_LLM_DIR / f"{receipt_id}_llm_raw.txt").write_text(
        raw, encoding="utf-8"
    )

    return json.loads(raw)
