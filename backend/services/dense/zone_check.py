"""Check zone boxes against the heads in a picture, so a box doesn't quietly mix two separate crowds.

People on a balcony and people on the floor below can sit close together in a picture, yet they are
separate crowds in separate spaces. What gives them away is an empty band across the box: a railing, a wall,
a stretch with no heads. Boxes are for a person to draw, since a picture alone can't say where one space ends;
this says when they look wrong.
"""

from __future__ import annotations

import numpy as np

MIN_GAP = 0.06  # an empty band at least this share of the picture separates two areas
MIN_SIDE = 0.15  # each side needs this share of the box's heads, so a few stragglers don't count
WIDE_BOX = 0.9  # a box this much of the picture can't separate anything, gap or no gap


def has_gap(points: np.ndarray, box: tuple, width: int, height: int) -> bool:
    """Whether an empty band runs across `box` (x0, y0, x1, y1 in pixels) with a real crowd on both sides."""
    x0, y0, x1, y1 = box
    inside = points[(points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)]
    if len(inside) < 20:
        return False
    enough = MIN_SIDE * len(inside)
    for axis, size in ((1, height), (0, width)):
        values = np.sort(inside[:, axis])
        gaps = np.diff(values)
        wide = np.flatnonzero(gaps / size >= max(MIN_GAP, 6 / len(inside)))  # few heads leave wide gaps by chance
        if any(min(i + 1, len(values) - i - 1) >= enough for i in wide):
            return True
    return False


def check_zones(points, zones: list[tuple], width: int, height: int) -> dict:
    """How boxes fit the heads (x, y). `zones`: (name, polygon). Returns {"heads": {name: heads in the box},
    "outside": heads in no box, "total": all heads, "mixed": boxes that may hold two separate areas, "wide": boxes
    covering nearly the whole picture, "overlap": pairs of boxes sharing heads, which those heads count twice in}."""
    points = np.asarray(points, float).reshape(-1, 2)
    covered = np.zeros(len(points), bool)
    heads, mixed, wide, masks = {}, [], [], {}
    for name, poly in zones:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        box = (min(xs), min(ys), max(xs), max(ys))
        inside = (points[:, 0] >= box[0]) & (points[:, 0] < box[2]) & (points[:, 1] >= box[1]) & (points[:, 1] < box[3])
        covered |= inside
        heads[name] = int(inside.sum())
        masks[name] = inside
        if (box[2] - box[0]) * (box[3] - box[1]) >= WIDE_BOX * width * height:
            wide.append(name)
        if has_gap(points, box, width, height):
            mixed.append(name)
    names = list(masks)
    overlap = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if (masks[a] & masks[b]).any()]
    return {"heads": heads, "outside": int((~covered).sum()), "total": len(points), "mixed": mixed, "wide": wide,
            "overlap": overlap}
