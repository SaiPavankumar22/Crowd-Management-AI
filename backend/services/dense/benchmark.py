"""Score the counter on the hand-labelled India test set, photo by photo and scene by scene.

Truth is every hand-marked head outside the photo's ignore zones (boxes drawn over people too
small or blurry to mark one by one). The model is judged on the same area: its raw head points
inside ignore zones are dropped before the count correction is applied, exactly as for the
truth. Accuracy is 1 - total error / total true count, the same measure as on ShanghaiTech.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import REPO_ROOT
from .points import corrected_count

INDIA = REPO_ROOT / "data" / "india_test"

# Scene of each photo, from words in its Commons title (first match wins).
SCENES = [
    ("Melas and pilgrimages", ("kumbh", "mela", "gangasagar", "pilgrim", "simhastha", "kumbhmela")),
    ("Stations and transport", ("station", "railway", "platform", "metro", "wdm", "train")),
    ("Temples and processions", ("temple", "mandir", "procession", "shrine", "golden", "rath", "chariot",
                                 "jagannath", "meenakshi", "brihadisvara", "chitirai", "chithirai", "pongala",
                                 "ponkala", "sabarimala", "kavadi", "deities", "march of the chariots")),
    ("Festivals", ("puja", "ganesh", "holi", "chhath", "onam", "bonalu", "jallikattu", "pookalam", "festival",
                   "diwali", "eid", "muharram", "durga")),
    ("Stadiums, ghats and rallies", ("stadium", "cricket", "ghat", "varanasi", "trump", "boat race", "rally",
                                     "beach", "parade", "clinton", "pride", "gateway", "mahotsav", "drama")),
    ("Streets and markets", ("market", "street", "shopping", "nagar", "charminar", "crowd", "bazaar", "stall")),
]


def scene_of(title: str) -> str:
    t = title.lower()
    for scene, words in SCENES:
        if any(w in t for w in words):
            return scene
    return "Streets and markets"


def _outside(points, zones):
    return [p for p in points
            if not any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, y0, x1, y1 in zones)]


def points_rc(points) -> list[tuple]:
    """Stored head points are (x, y); the drawing code takes (row, column), i.e. (y, x)."""
    return [(y, x) for x, y in points]


def load(root: Path = INDIA) -> list[dict]:
    """One row per photo with hand-checked labels: true count, model count, error and scene."""
    labels_file = root / "labels.json"
    if not labels_file.exists():
        return []
    labels = json.loads(labels_file.read_text(encoding="utf-8"))["images"]
    sources = {r["image"]: r for r in csv.DictReader(open(root / "sources.csv", encoding="utf-8"))}
    rows = []
    for image, lab in sorted(labels.items()):
        pred_file = root / "predictions" / (Path(image).stem + ".json")
        if not pred_file.exists():
            continue
        pred = json.loads(pred_file.read_text(encoding="utf-8"))
        zones = lab.get("ignore") or []
        truth = _outside([p[:2] for p in lab["points"]], zones)
        model_raw = _outside(pred["points"], zones)
        model = corrected_count(len(model_raw)) if model_raw else 0.0
        src = sources.get(image, {})
        rows.append({
            "image": image, "title": src.get("title", image), "scene": src.get("scene") or scene_of(src.get("title", "")),
            "author": src.get("author", ""), "license": src.get("license", ""), "source": src.get("source", ""),
            "true": len(truth), "model": model, "error": model - len(truth),
            "truth_points": truth, "model_points": model_raw, "ignore": zones,
        })
    return rows


def accuracy(rows: list[dict]) -> float | None:
    """1 - total absolute error / total true count, or None with nothing to score."""
    total = sum(r["true"] for r in rows)
    if not total:
        return None
    return 1 - sum(abs(r["error"]) for r in rows) / total


def by_scene(rows: list[dict]) -> list[dict]:
    """Accuracy and mean error per scene, biggest scenes first."""
    scenes = {}
    for r in rows:
        scenes.setdefault(r["scene"], []).append(r)
    out = [{"scene": s, "photos": len(rs), "people": sum(r["true"] for r in rs),
            "accuracy": accuracy(rs), "mae": sum(abs(r["error"]) for r in rs) / len(rs)}
           for s, rs in scenes.items()]
    return sorted(out, key=lambda d: -d["photos"])
