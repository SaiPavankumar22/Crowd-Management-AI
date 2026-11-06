"""Counting-error metrics, as reported in crowd-counting papers."""

from __future__ import annotations

import numpy as np

# Ground-truth crowd sizes used to break errors down: sparse groups to packed squares.
DEFAULT_BUCKETS = (0, 50, 100, 250, 500, 1000, float("inf"))


def mae(pred, true) -> float:
    """Mean absolute error in people per image."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    return float(np.mean(np.abs(pred - true)))


def rmse(pred, true) -> float:
    """Root mean squared error. Crowd-counting papers usually label this "MSE"."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def bias(pred, true) -> float:
    """Mean signed error: negative means the method undercounts."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    return float(np.mean(pred - true))


def by_bucket(pred, true, edges=DEFAULT_BUCKETS) -> list[dict]:
    """Errors for each crowd-size band of the ground truth; empty bands are skipped."""
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (true >= low) & (true < high)
        if not mask.any():
            continue
        rows.append({
            "range":     f"{int(low)}+" if np.isinf(high) else f"{int(low)}-{int(high)}",
            "images":    int(mask.sum()),
            "mean_true": float(true[mask].mean()),
            "mean_pred": float(pred[mask].mean()),
            "mae":       mae(pred[mask], true[mask]),
            "bias":      bias(pred[mask], true[mask]),
        })
    return rows


def summarise(pred, true) -> dict:
    return {
        "mae":     mae(pred, true),
        "rmse":    rmse(pred, true),
        "bias":    bias(pred, true),
        "buckets": by_bucket(pred, true),
    }
