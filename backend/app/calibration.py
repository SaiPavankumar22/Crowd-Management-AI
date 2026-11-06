"""Camera ground-plane calibration helpers."""
from __future__ import annotations

import cv2
import numpy as np


def validate_calibration(data: dict | None) -> dict:
    if not isinstance(data, dict):
        raise ValueError("calibration body must be an object")
    image_points = data.get("image_points")
    world_points = data.get("world_points")
    if not isinstance(image_points, list) or not isinstance(world_points, list):
        raise ValueError("calibration needs image_points and world_points")
    if len(image_points) < 4 or len(image_points) != len(world_points):
        raise ValueError("calibration needs at least four matching points")
    image = _points(image_points)
    world = _points(world_points)
    if not np.isfinite(image).all() or not np.isfinite(world).all():
        raise ValueError("calibration points must be finite numbers")
    try:
        matrix, mask = cv2.findHomography(image, world, method=0)
    except cv2.error as exc:
        raise ValueError("calibration points are invalid or degenerate") from exc
    if matrix is None:
        raise ValueError("calibration points are degenerate")
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("calibration points produce an invalid homography")
    return {
        "image_points": image.tolist(),
        "world_points": world.tolist(),
        "homography": matrix.tolist(),
        "inliers": int(mask.sum()) if mask is not None else 0,
    }


def project_points(points, calibration: dict | None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if not calibration or not calibration.get("homography"):
        return pts.reshape(-1, 2)
    matrix = np.asarray(calibration["homography"], dtype=np.float64)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def polygon_area_world(polygon, calibration: dict | None) -> float | None:
    if not calibration:
        return None
    projected = project_points(polygon, calibration).astype(np.float32)
    return float(abs(cv2.contourArea(projected)))


def _points(value) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("calibration points must be numeric [x, y] pairs") from exc
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("calibration points must be [x, y] pairs")
    return points
