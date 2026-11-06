"""Crowd-safety measures in real units: people per square metre and crowd pressure.

Density levels (people per m²):
- about 2.2: Fruin's walkway Level of Service F, where normal flow breaks down
- 5: Still, "significantly increased risk to life and limb" above this
- 7: Fruin, quoted by Helbing et al.: "the crowd becomes almost a fluid mass",
  and shock waves can travel through it

Crowd pressure (Helbing, Johansson and Al-Abideen, Phys. Rev. E 75, 046109, 2007):
P = local density x local velocity variance, in 1/s². In their video of the
2006 Hajj disaster, "turbulent" motion began when P passed 0.02/s², and the
crush began at its peak. They note the critical value depends on the averaging
window, so 0.02 is a reference line, not a calibrated alarm level.
"""

from __future__ import annotations

import numpy as np

from .geometry import GroundCalibration

DENSITY_LEVELS = [  # (lower bound in people per m², label)
    (0.0, "Normal"),
    (2.2, "Crowded"),
    (5.0, "Dangerous"),
    (7.0, "Crush risk"),
]
HELBING_TURBULENCE_PRESSURE = 0.02  # 1/s²


def density_level(people_per_m2: float) -> str:
    level = DENSITY_LEVELS[0][1]
    for bound, label in DENSITY_LEVELS:
        if people_per_m2 >= bound:
            level = label
    return level


def _box_sum(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over each k x k neighbourhood (k odd, zero outside the array), same shape as a."""
    if k <= 1:
        return a.astype(float)
    pad = k // 2
    c = np.pad(np.pad(a.astype(float), pad).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = a.shape
    return c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]


def cell_areas(calib: GroundCalibration, image_shape, grid_shape, cell: int = 8) -> np.ndarray:
    """Ground area in m² of each density-map cell (cell x cell pixels), sky counted as 0."""
    area = np.nan_to_num(calib.pixel_area(image_shape[:2]))
    gh, gw = grid_shape
    h, w = min(gh * cell, area.shape[0]), min(gw * cell, area.shape[1])
    hc, wc = h // cell, w // cell
    blocks = area[:hc * cell, :wc * cell].reshape(hc, cell, wc, cell).sum(axis=(1, 3))
    out = np.zeros(grid_shape)
    out[:hc, :wc] = blocks
    return out


def density_per_m2(density: np.ndarray, calib: GroundCalibration, image_shape,
                   window: int = 5) -> np.ndarray:
    """People per m² around each density-map cell, averaged over a window x window patch.

    Both people and ground area are summed over the patch before dividing, so a
    single far-away cell can't produce a wild ratio. Cells with no ground get NaN.
    """
    people = _box_sum(density, window)
    area = _box_sum(cell_areas(calib, image_shape, density.shape), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(area > 0, people / area, np.nan)


def _inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon test, vectorised over points."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    px, py = polygon[:, 0], polygon[:, 1]
    for i in range(len(polygon)):
        x1, y1, x2, y2 = px[i - 1], py[i - 1], px[i], py[i]
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(invalid="ignore", divide="ignore"):
            x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (x < x_at)
    return inside


def zone_density(density: np.ndarray, calib: GroundCalibration, polygon_px,
                 cell: int = 8) -> dict:
    """People, ground area and people per m² inside a zone drawn on the image."""
    poly = np.asarray(polygon_px, float)
    rows, cols = np.mgrid[0:density.shape[0], 0:density.shape[1]]
    centres = np.c_[(cols.ravel() + 0.5) * cell, (rows.ravel() + 0.5) * cell]
    mask = _inside(centres, poly).reshape(density.shape)
    people = float(density[mask].sum())
    area = calib.polygon_area(poly)
    per_m2 = people / area if area > 0 else float("nan")
    return {"people": people, "area_m2": area, "per_m2": per_m2,
            "level": density_level(per_m2)}


def crowd_pressure(people_per_m2: float, velocities_mps) -> float:
    """Helbing's crowd pressure for one region and time window, in 1/s².

    `velocities_mps` holds ground velocities (N x 2, m/s) sampled across the
    region and window; their variance is the mean squared deviation from the
    average velocity.
    """
    v = np.atleast_2d(np.asarray(velocities_mps, float))
    if len(v) == 0:
        return 0.0
    variance = float(np.mean(np.sum((v - v.mean(axis=0)) ** 2, axis=1)))
    return people_per_m2 * variance
