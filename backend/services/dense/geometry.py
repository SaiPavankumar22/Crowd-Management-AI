"""Map camera pixels onto the ground in metres, so counts become people per square metre.

A camera sees the ground in perspective: a pixel near the bottom of the picture
covers far less floor than one near the top. Four points on the ground whose
real positions are known (the corners of floor tiles, a marked area, a court)
are enough to undo that perspective with a homography: a 3 x 3 matrix H that
sends an image point (u, v) to a ground point (x, y) in metres.

Crowd-safety limits are set in people per square metre (Still: significant
risk above about 5 per m²), so this is what turns the AI's counts into
numbers a safety officer can act on.
"""

from __future__ import annotations

import numpy as np


def _normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centre the points and scale them to mean distance sqrt(2)."""
    centre = points.mean(axis=0)
    scale = np.sqrt(2) / max(np.linalg.norm(points - centre, axis=1).mean(), 1e-12)
    t = np.array([[scale, 0, -scale * centre[0]],
                  [0, scale, -scale * centre[1]],
                  [0, 0, 1]])
    return (points - centre) * scale, t


def homography(src, dst) -> np.ndarray:
    """3 x 3 matrix H with dst ~ H @ src, from 4 or more point pairs (direct linear transform)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
        raise ValueError("need at least 4 matching (x, y) point pairs")
    s, ts = _normalise(src)
    d, td = _normalise(dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, singular, vt = np.linalg.svd(np.asarray(rows))
    if singular[-2] < 1e-9:
        raise ValueError("points are degenerate (three or more in a line)")
    h = np.linalg.inv(td) @ vt[-1].reshape(3, 3) @ ts
    return h / h[2, 2]


class GroundCalibration:
    """Pixel-to-ground mapping for one fixed camera."""

    def __init__(self, image_points, ground_points_m):
        self.H = homography(image_points, ground_points_m)
        # Pick the sign that makes the projective weight positive on the calibrated area.
        w = self._weights(np.asarray(image_points, float))
        if np.median(w) < 0:
            self.H = -self.H

    @classmethod
    def from_rectangle(cls, corners_px, width_m: float, length_m: float) -> GroundCalibration:
        """Corners of a ground rectangle as clicked in the image, in order: top-left,
        top-right, bottom-right, bottom-left (as they appear on the ground plan)."""
        ground = [(0, 0), (width_m, 0), (width_m, length_m), (0, length_m)]
        return cls(corners_px, ground)

    def _weights(self, points: np.ndarray) -> np.ndarray:
        return points @ self.H[2, :2] + self.H[2, 2]

    def to_ground(self, points_px) -> np.ndarray:
        """Image points (N x 2, pixels) to ground points (N x 2, metres)."""
        p = np.atleast_2d(np.asarray(points_px, float))
        mapped = np.c_[p, np.ones(len(p))] @ self.H.T
        return mapped[:, :2] / mapped[:, 2:]

    def pixel_area(self, shape) -> np.ndarray:
        """Ground area in m² covered by each pixel of an image of the given (height, width).

        For a homography, the local area scale is |det H| / w³, where w is the
        projective weight at that pixel. Pixels at or above the horizon (w <= 0)
        don't lie on the ground and get NaN.
        """
        h, w = shape[:2]
        v, u = np.mgrid[0:h, 0:w].astype(float)
        weight = self.H[2, 0] * (u + 0.5) + self.H[2, 1] * (v + 0.5) + self.H[2, 2]
        area = abs(np.linalg.det(self.H)) / np.where(weight > 0, weight, np.nan) ** 3
        return area

    def polygon_area(self, polygon_px) -> float:
        """Ground area in m² of a polygon drawn on the image (shoelace formula on the ground)."""
        g = self.to_ground(polygon_px)
        x, y = g[:, 0], g[:, 1]
        return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)

    def ground_velocity(self, points_px, flow_px, fps: float) -> np.ndarray:
        """Turn optical flow at image points (pixels per frame) into ground velocity (m/s)."""
        p = np.atleast_2d(np.asarray(points_px, float))
        moved = p + np.atleast_2d(np.asarray(flow_px, float))
        return (self.to_ground(moved) - self.to_ground(p)) * fps
