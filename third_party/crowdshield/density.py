"""Density-map crowd counting with DM-Count (Wang et al., NeurIPS 2020, MIT licence).

Instead of drawing a box around each person, the model predicts a density map:
every pixel holds a fraction of a person, and the whole map sums to the head
count. That keeps working when bodies overlap, which is where detectors fail.
The network mirrors DM-Count's models.py so its trained weights load as-is.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, default_max_side, fit_max_side
from .weights import RELEASES, fetch

# Trained on ShanghaiTech A + B with an honest held-out split; see the release notes.
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "dmcount_shab.pth"
WEIGHTS_URL = f"{RELEASES}/dmcount-shab-v1/dmcount_shab.pth"
WEIGHTS_SHA256 = "d6f72bffcd1ea986011796b43a63e2738fe8c97d19d230ef99286b603ea207b9"
# Very large photos are shrunk first: the network's memory grows with pixel count.
# Small free servers can set CROWDSHIELD_MAX_SIDE lower (e.g. 1280).
MAX_SIDE = default_max_side()  # 2048, or less on a small server (see config.py)

# VGG19 convolution layers ("M" = 2x2 max-pool), configuration E without batch norm.
_VGG19 = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
          512, 512, 512, 512, "M", 512, 512, 512, 512]
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model():
    """DM-Count's network: VGG19 features, 2x upsampling, then a small regression head."""
    import torch.nn as nn
    import torch.nn.functional as F

    class DMCount(nn.Module):
        def __init__(self):
            super().__init__()
            layers, channels = [], 3
            for v in _VGG19:
                if v == "M":
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers += [nn.Conv2d(channels, v, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                    channels = v
            self.features = nn.Sequential(*layers)
            self.reg_layer = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            )
            self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

        def forward(self, x):
            x = F.interpolate(self.features(x), scale_factor=2, mode="bilinear",
                              align_corners=True)
            return self.density_layer(self.reg_layer(x))

    return DMCount()


def ensure_weights(path: Path = DEFAULT_WEIGHTS) -> Path:
    """Download the trained weights from the GitHub release on first use, and check them."""
    path = Path(path)
    if path != DEFAULT_WEIGHTS and not path.exists():
        raise FileNotFoundError(f"Density model weights not found: {path}")
    return fetch(path, WEIGHTS_URL, WEIGHTS_SHA256)


@lru_cache(maxsize=2)
def load_density_model(weights: str = str(DEFAULT_WEIGHTS)):
    """Load trained weights onto the GPU if there is one, else the CPU."""
    import torch

    path = ensure_weights(Path(weights))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval(), device


def density_from_model(model, device: str, img_bgr: np.ndarray) -> np.ndarray:
    """Run the network on a BGR image; returns the density map at 1/8 resolution."""
    import torch

    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    x = torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy())
    with torch.no_grad():
        return model(x.unsqueeze(0).to(device))[0, 0].cpu().numpy()


def estimate_density(img_bgr: np.ndarray, weights: str | None = None) -> dict:
    """Count people in a BGR photo; the density map sums to the count."""
    import cv2

    model, device = load_density_model(str(weights or DEFAULT_WEIGHTS))
    scale = min(1.0, fit_max_side(ceiling=MAX_SIDE) / max(img_bgr.shape[:2]))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    density = density_from_model(model, device, img_bgr)
    return {"count": float(density.sum()), "density": density, "scale": scale}


def density_grid(frame_bgr: np.ndarray) -> np.ndarray:
    """Counter for the video monitor: a 1/8-size grid whose sum is the head count."""
    import cv2

    out = estimate_density(frame_bgr)
    density = out["density"]
    if out["scale"] < 1.0:  # back onto the full frame's 1/8 grid, keeping the total
        h, w = frame_bgr.shape[0] // 8, frame_bgr.shape[1] // 8
        total = density.sum()
        density = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
        density *= total / max(density.sum(), 1e-9)
    return density
