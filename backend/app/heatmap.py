"""Crowd density heatmap: an exponentially-decaying occupancy grid.

Every detected person's foot point is splatted with a small Gaussian kernel
into a low-resolution grid. The grid decays on each inference tick, so it
visualises where people *accumulate over time* -- choke points, queue lanes,
standing areas -- rather than a single frame's boxes.

Encoding notes
--------------
* 48 x 27 cells keeps the payload tiny (1296 bytes -> ~1.7 KB base64) while
  still looking smooth once the dashboard upscales it bilinearly + blurs.
* The grid is normalised against a slowly-decaying running peak, so a quiet
  scene still shows faint structure instead of everything reading "max".
* Pure NumPy, no OpenCV: this runs inside the per-camera worker at the
  inference rate, so it must stay cheap.
"""
import base64

import numpy as np


class HeatmapAccumulator:
    """Decaying occupancy grid fed by normalised foot points (x, y in 0..1)."""

    def __init__(self, cols: int = 48, rows: int = 27,
                 decay: float = 0.994, sigma: float = 1.1):
        self.cols = int(cols)
        self.rows = int(rows)
        self.decay = float(decay)
        self.sigma = float(sigma)
        self.grid = np.zeros((self.rows, self.cols), np.float32)
        self.peak = 1.0
        radius = max(1, int(round(self.sigma * 2)))
        self._offsets = [
            (dx, dy, float(np.exp(-(dx * dx + dy * dy) / (2 * self.sigma ** 2))))
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]

    # ------------------------------------------------------------------ #
    def add_points(self, pts_norm: np.ndarray) -> None:
        """Splat normalised points into the grid, after decaying it."""
        self.grid *= self.decay
        if pts_norm is None or len(pts_norm) == 0:
            return
        cols, rows = self.cols, self.rows
        for x, y in pts_norm:
            cx = int(x * cols)
            cy = int(y * rows)
            for dx, dy, weight in self._offsets:
                gx = cx + dx
                gy = cy + dy
                if 0 <= gx < cols and 0 <= gy < rows:
                    self.grid[gy, gx] += weight

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.grid[:] = 0.0
        self.peak = 1.0

    def encode(self) -> dict:
        """Normalised 0..255 grid as base64 bytes + metadata for the UI."""
        current = float(self.grid.max())
        self.peak = max(self.peak * 0.997, current, 1e-3)
        # gamma < 1 lifts the mid-tones so thin trails stay visible
        norm = np.clip(self.grid / self.peak, 0.0, 1.0) ** 0.65
        data = (norm * 255.0).astype(np.uint8)
        return {
            "cols": self.cols,
            "rows": self.rows,
            "b64": base64.b64encode(data.tobytes()).decode("ascii"),
            "peak": round(current, 3),
        }
