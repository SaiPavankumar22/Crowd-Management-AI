"""Load ShanghaiTech-style crowd datasets: photos plus .mat files of head positions."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _head_points(obj) -> np.ndarray | None:
    """Depth-first search of a loaded .mat structure for the N x 2 array of head points."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not key.startswith("__"):
                found = _head_points(value)
                if found is not None:
                    return found
        return None
    if not isinstance(obj, np.ndarray):
        return None
    if obj.dtype.names:
        for name in obj.dtype.names:
            for item in obj[name].flat:
                found = _head_points(item)
                if found is not None:
                    return found
        return None
    if obj.dtype == object:
        for item in obj.flat:
            found = _head_points(item)
            if found is not None:
                return found
        return None
    if obj.size == 0:
        return np.zeros((0, 2))
    if obj.ndim == 2 and obj.shape[1] == 2:
        return obj.astype(float)
    return None


def _natural_key(path: Path):
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[-1]) if numbers else -1, path.name)


def find_part_dir(root: Path, part: str) -> Path:
    """Locate part_A / part_A_final (or B) anywhere under root, as mirrors nest it differently."""
    matches = [p for p in Path(root).rglob(f"part_{part}*")
               if p.is_dir() and (p / "test_data").is_dir()]
    if not matches:
        raise FileNotFoundError(f"No part_{part} folder with test_data under {root}")
    return min(matches, key=lambda p: len(p.parts))


def load_shanghaitech(root, part: str = "A", split: str = "test") -> list[tuple[Path, np.ndarray]]:
    """Return (image path, head points) pairs in natural order, IMG_2 before IMG_10."""
    from scipy.io import loadmat

    data_dir = find_part_dir(Path(root), part) / f"{split}_data"
    gt_dirs = [d for d in (data_dir / "ground_truth", data_dir / "ground-truth") if d.is_dir()]
    if not gt_dirs:
        raise FileNotFoundError(f"No ground_truth folder in {data_dir}")

    samples = []
    for image in sorted((data_dir / "images").glob("*.jpg"), key=_natural_key):
        mat_path = gt_dirs[0] / f"GT_{image.stem}.mat"
        points = _head_points(loadmat(mat_path))
        if points is None:
            raise ValueError(f"No head points found in {mat_path}")
        samples.append((image, points))
    return samples
