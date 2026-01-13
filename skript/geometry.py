import statistics
from typing import Any, Dict, List

from config import LINE_Y_THRESHOLD_MULT, MAX_BUCKETS, MIN_BUCKETS, X_GAP_MULT


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
