import csv
from typing import Any, Dict

from config import CSV_DIR, CSV_HEADER
from utils import utc_now


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
