"""Tiled / region-of-interest inference helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    w: int
    h: int

def compute_tiles(width: int, height: int, rows: int, cols: int,
                  overlap: float = 0.15) -> list[Tile]:
    """Split a frame into an overlapping grid.

    `overlap` is the fraction of each tile that is shared with its neighbour,
    which is what stops a person standing exactly on a cut line from being
    clipped in half in every tile and detected in none.
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    overlap = min(max(0.0, float(overlap)), 0.45)
    tile_w = width / (cols - (cols - 1) * overlap)
    tile_h = height / (rows - (rows - 1) * overlap)
    step_x = tile_w * (1.0 - overlap)
    step_y = tile_h * (1.0 - overlap)

    tiles: list[Tile] = []
    for r in range(rows):
        for c in range(cols):
            x = int(round(c * step_x))
            y = int(round(r * step_y))
            w = int(round(tile_w))
            h = int(round(tile_h))
            w = min(w, width - x)
            h = min(h, height - y)
            if w > 8 and h > 8:
                tiles.append(Tile(x, y, w, h))
    return tiles

def crop(frame: np.ndarray, tile: Tile) -> np.ndarray:
    return frame[tile.y:tile.y + tile.h, tile.x:tile.x + tile.w]

def offset_boxes(boxes: np.ndarray, tile: Tile) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    shifted = boxes.astype(np.float32, copy=True)
    shifted[:, [0, 2]] += tile.x
    shifted[:, [1, 3]] += tile.y
    return shifted

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.55,
        max_det: int = 1000) -> np.ndarray:
    """Greedy NMS returning the kept indices, ordered by descending score."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if len(keep) >= max_det or order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=int)

def merge_tile_results(boxes: np.ndarray, scores: np.ndarray,
                       iou_threshold: float = 0.55,
                       max_det: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Merge tile detections after their boxes have been offset to frame space."""
    keep = nms(boxes, scores, iou_threshold=iou_threshold, max_det=max_det)
    if len(keep) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
    return (boxes[keep].astype(np.float32), scores[keep].astype(np.float32))

@dataclass
class TilingPolicy:
    """Only pay the 9x compute when the scene actually needs it.

    A person shorter than `min_box_height_px` in the current view is the
    reliable signal: that is when detection recall collapses, and it is cheap
    to measure from the boxes the detector already returned.
    """
    enabled: bool = True
    rows: int = 3
    cols: int = 3
    overlap: float = 0.15
    min_box_height_px: float = 40.0
    min_boxes: int = 3

    def should_tile(self, boxes: np.ndarray) -> bool:
        if not self.enabled or len(boxes) == 0:
            return False
        heights = boxes[:, 3] - boxes[:, 1]
        if len(heights) < self.min_boxes:
            return False
        return float(np.median(heights)) < self.min_box_height_px
