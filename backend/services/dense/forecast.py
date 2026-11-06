"""Early warning: how long until a zone reaches a dangerous density, at its current trend.

A straight line is fitted to the zone's recent people-per-m² readings with the
Theil-Sen estimator (the median of all pairwise slopes), so one bad count
can't swing the forecast. If the line is rising, the time until it crosses the
danger level is the warning.
"""

from __future__ import annotations

import numpy as np

DANGER_PER_M2 = 5.0  # Still: significantly increased risk above 5 people per m²


def theil_sen_slope(times, values) -> float:
    t, v = np.asarray(times, float), np.asarray(values, float)
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    keep = dt > 0
    if not keep.any():
        return 0.0
    return float(np.median((v[j] - v[i])[keep] / dt[keep]))


def seconds_until(times, values, level: float = DANGER_PER_M2, window_s: float = 60.0,
                  min_points: int = 4) -> float | None:
    """Seconds until the trend reaches `level`, judged on the last `window_s` seconds.

    0 means it is already there. None means there is no warning: too few
    readings, or the density is steady or falling.
    """
    t, v = np.asarray(times, float), np.asarray(values, float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) == 0:
        return None
    recent = t >= t[-1] - window_s
    t, v = t[recent], v[recent]
    current = float(np.median(v[-3:]))
    if current >= level:
        return 0.0
    if len(t) < min_points:
        return None
    slope = theil_sen_slope(t, v)
    if slope <= 0:
        return None
    return (level - current) / slope
