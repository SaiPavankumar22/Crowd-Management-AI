"""Crowd counting with head points: PET, fine-tuned on ShanghaiTech and UCF-QNRF.

PET (Liu et al., ICCV 2023) puts one point on each head it finds, so it gives
both the count and where every person is. On the ShanghaiTech test photos it
was the most accurate counter we tried (see results/shanghaitech).

This is the second version (pet-qnrf-v1): the first (pet-shab-v1) fine-tuned once
more on ShanghaiTech A (twice over) + B, the 1,201 dense UCF-QNRF photos and 23
people-free Indian railway photos. It counts Indian crowds far better (see
results/model_v2) and is level with the first on ShanghaiTech.

The raw count is passed through one correction, true ≈ 1.340 × raw^0.947,
fitted on held-out training photos (never on test photos). It trims
PET's overcounting in packed crowds and its undercounting on streets.
"""

from __future__ import annotations

import os
from argparse import Namespace
from contextlib import contextmanager
from functools import lru_cache

import numpy as np

from .config import MIN_SIDE, REPO_ROOT, default_max_side, default_work_side, fit_max_side
from .weights import RELEASES, fetch

WEIGHTS = REPO_ROOT / "models" / "pet_qnrf.pth"
WEIGHTS_URL = f"{RELEASES}/pet-qnrf-v1/pet_qnrf.pth"
WEIGHTS_SHA256 = "e8476157d314606f143947b0f94d4f915e5f22ef11e26e69d5188ce0f0a2ed92"
CORRECTION = (1.340, 0.947)  # true ≈ a * raw ** b, fitted on held-out training photos
MAX_SIDE = default_max_side()  # kept for callers that read it; counting uses WORK_SIDE below
WORK_SIDE = default_work_side()  # the detail to count at; a small server does it in tiles
TILE_MARGIN = 48  # context around each tile's core, in pixels, so heads at a seam are still seen whole
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The settings PET was trained with (its train.sh).
_ARGS = dict(backbone="vgg16_bn", position_embedding="sine", dec_layers=2,
             dim_feedforward=512, hidden_dim=256, dropout=0.0, nheads=8,
             set_cost_class=1.0, set_cost_point=0.05, ce_loss_coef=1.0,
             point_loss_coef=5.0, eos_coef=0.5)


_progress = None  # called as fn(done, total) after each tile; set with on_tile_progress()


@contextmanager
def on_tile_progress(fn):
    """Run counting with `fn(done, total)` called after every tile, so a page can show real progress
    during the minute a big photo takes on a small server."""
    global _progress
    previous, _progress = _progress, fn
    try:
        yield
    finally:
        _progress = previous


def corrected_count(raw: float) -> float:
    a, b = CORRECTION
    return float(a * raw ** b) if raw > 0 else 0.0


def build_model(device: str = "cpu"):
    from .pet.models import build_pet

    model, _ = build_pet(Namespace(**_ARGS, device=device))
    return model


@lru_cache(maxsize=1)
def load_point_model():
    """The trained PET model on the GPU if there is one, else the CPU."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = fetch(WEIGHTS, WEIGHTS_URL, WEIGHTS_SHA256)
    model = build_model(device)
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), device


def points_from_model(model, device: str, img_bgr: np.ndarray) -> np.ndarray:
    """Head points (N x 2, row then column, in the given image's pixels)."""
    import torch

    from .pet.util.misc import nested_tensor_from_tensor_list

    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    x = torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy())
    with torch.inference_mode():  # same numbers as no_grad, a little less overhead
        samples = nested_tensor_from_tensor_list([x]).to(device)
        out = model(samples, test=True)
    h, w = samples.tensors.shape[-2:]
    pts = out["pred_points"][0].cpu().numpy().reshape(-1, 2)
    return pts * np.array([h, w], dtype=float)


PAD = 256  # PET pads its input up to a multiple of this (pet/util/misc.py), so that is the real size it computes
TILE_PLAN = "padded"  # "padded" plans tiles by PET's padded size; "square" is the original layout, kept to compare


def _padded(n: int) -> int:
    return -(-n // PAD) * PAD


def tile_layout(height: int, width: int, tile_side: int, margin: int = TILE_MARGIN) -> list[tuple]:
    """Split an image into cores (y0, y1, x0, x1) that cover it exactly once.

    A tile is its core plus up to `margin` pixels of context on each inner side. PET pads every input
    up to a multiple of 256 px, and its memory follows that padded size, so the plan keeps every tile's
    padded area within the padded area of a `tile_side` square (the size that fits in memory) and picks
    the split with the least padded area in total: for a 1024 x 768 photo, two 1024-wide strips instead
    of four squares that are mostly padding.
    """
    if TILE_PLAN == "square":
        core = max(64, tile_side - 2 * margin)
        rows, cols = max(1, -(-height // core)), max(1, -(-width // core))
    else:
        budget = _padded(tile_side) ** 2
        best = None
        for rows in range(1, max(1, height // 64) + 1):
            tile_h = -(-height // rows) + margin * min(2, rows - 1)
            for cols in range(1, max(1, width // 64) + 1):
                tile_w = -(-width // cols) + margin * min(2, cols - 1)
                area = _padded(tile_h) * _padded(tile_w)
                if area > budget:
                    continue
                cost = (rows * cols * area, rows * cols)
                if best is None or cost < best[0]:
                    best = (cost, rows, cols)
        if best is None:  # even 64 px cores don't fit: fall back to the smallest squares
            core = 64
            rows, cols = max(1, -(-height // core)), max(1, -(-width // core))
        else:
            _, rows, cols = best
    ys = np.linspace(0, height, rows + 1).round().astype(int)
    xs = np.linspace(0, width, cols + 1).round().astype(int)
    return [(ys[r], ys[r + 1], xs[c], xs[c + 1]) for r in range(rows) for c in range(cols)]


def points_tiled(model, device: str, img_bgr: np.ndarray, tile_side: int,
                 margin: int = TILE_MARGIN) -> np.ndarray:
    """Head points for an image counted one tile at a time, so memory is set by the tile, not the photo.

    Each tile is run with `margin` pixels of context, and a head is kept only by the tile whose core
    contains it: nobody is counted twice at a seam, and nobody is dropped.
    """
    h, w = img_bgr.shape[:2]
    found = []
    layout = tile_layout(h, w, tile_side, margin)
    for done, (y0, y1, x0, x1) in enumerate(layout, start=1):
        ya, yb, xa, xb = max(0, y0 - margin), min(h, y1 + margin), max(0, x0 - margin), min(w, x1 + margin)
        pts = points_from_model(model, device, img_bgr[ya:yb, xa:xb]) + np.array([ya, xa], dtype=float)
        keep = (pts[:, 0] >= y0) & (pts[:, 0] < y1) & (pts[:, 1] >= x0) & (pts[:, 1] < x1)
        found.append(pts[keep])
        if _progress:
            _progress(done, len(layout))
    return np.concatenate(found) if found else np.zeros((0, 2))


def count_points(img_bgr: np.ndarray) -> dict:
    """Count people in a BGR photo: corrected count, raw count and head points.

    The photo is shrunk to WORK_SIDE. If the server has the memory to run that in one pass it does;
    otherwise the photo is counted in tiles that fit, which keeps the same detail.
    """
    import cv2

    model, device = load_point_model()
    scale = min(1.0, WORK_SIDE / max(img_bgr.shape[:2]))
    small = img_bgr if scale == 1.0 else cv2.resize(img_bgr, None, fx=scale, fy=scale,
                                                    interpolation=cv2.INTER_AREA)
    longest = max(small.shape[:2])
    fit = fit_max_side(ceiling=max(MIN_SIDE, longest))
    if fit >= longest:
        pts = points_from_model(model, device, small)
    else:
        pts = points_tiled(model, device, small, fit)
    pts = pts / scale
    return {"count": corrected_count(len(pts)), "raw_count": len(pts), "points": pts}


def point_grid(frame_bgr: np.ndarray) -> np.ndarray:
    """Counter for the video monitor: a 1/8-size grid of head points, scaled to the corrected count."""
    from .monitor import points_to_grid

    out = count_points(frame_bgr)
    grid = points_to_grid(out["points"], frame_bgr.shape)
    if out["raw_count"]:
        grid *= out["count"] / out["raw_count"]
    return grid
