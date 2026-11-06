"""PET and DM-Count, pinned to the checkpoints in ``models/``.

plan.md §5.5 requires the *original* network definitions from the approved
source, the local checkpoints, verified checksums and a recorded version.  The
definitions are vendored under ``third_party/crowdshield`` (Crowd Shield,
academic-use licence); this module owns everything that differs in a
deployment: which checkpoint is loaded, checksum verification, the count
correction and the device.

Neither model is allowed to become an operational count on its own.  The
adapters in ``app/adapters.py`` wrap these calls in the advisory contracts and
the router keeps them unselected until a per-camera validation is signed off.
"""
from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path

import numpy as np

from . import config

_THIRD_PARTY = Path(config.BASE_DIR) / "third_party"
MODEL_DIR = Path(config.MODELS_DIR)

PET_WEIGHTS = MODEL_DIR / "pet_shab.pth"
DMCOUNT_WEIGHTS = MODEL_DIR / "dmcount_shab.pth"

# Recorded once from the downloaded files (plan.md step 2) and checked on every
# load, so a swapped or truncated checkpoint fails loudly instead of miscounting.
PET_SHA256 = "5a7c6590e61938e42c6b3a2bbde5fbec34ee7eb8b2c9cae0fdba2a45d8437595"
DMCOUNT_SHA256 = "d6f72bffcd1ea986011796b43a63e2738fe8c97d19d230ef99286b603ea207b9"

PET_VERSION = "pet-shab-vgg16bn"
DMCOUNT_VERSION = "dmcount-shab-vgg19"

# true ≈ a × raw ** b, fitted by Crowd Shield on held-out ShanghaiTech photos
# (model card, and plan.md §5.5).  Off by default: plan.md says it must not
# drive operational numbers until it is validated on local camera views.
PET_CORRECTION = (1.375, 0.942)

_lock = threading.Lock()
_loaded: dict[str, object] = {}
_hashes: dict[str, str] = {}


def _crowdshield():
    """Import the vendored package without making it a hard startup dependency."""
    if str(_THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(_THIRD_PARTY))
    from crowdshield import density as cs_density
    from crowdshield import points as cs_points

    return cs_points, cs_density


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> Path:
    """Fail early when a checkpoint is missing, truncated or simply another file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found in {path.parent}; place the Crowd Shield "
            "checkpoint there or set CROWD_MODELS_DIR")
    actual = _sha256(path)
    if expected and actual != expected:
        raise RuntimeError(
            f"{path.name} checksum mismatch (expected {expected[:12]}…, "
            f"got {actual[:12]}…) — refusing to load an unverified checkpoint")
    _hashes[path.name] = actual
    return path


def _torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _pet():
    with _lock:
        if "pet" not in _loaded:
            cs_points, _ = _crowdshield()
            # The vendored package defaults to its newer pet_qnrf.pth and the
            # correction fitted for that checkpoint; point both at the shab
            # checkpoint this deployment actually ships.
            cs_points.WEIGHTS = verify(PET_WEIGHTS, PET_SHA256)
            cs_points.CORRECTION = PET_CORRECTION
            _loaded["pet"] = cs_points.load_point_model()
        return _loaded["pet"]


def _dmcount():
    with _lock:
        if "dmcount" not in _loaded:
            _, cs_density = _crowdshield()
            cs_density.DEFAULT_WEIGHTS = verify(DMCOUNT_WEIGHTS, DMCOUNT_SHA256)
            _loaded["dmcount"] = cs_density.load_density_model(str(DMCOUNT_WEIGHTS))
        return _loaded["dmcount"]


def count_points(frame_bgr: np.ndarray) -> dict:
    """Head points for one BGR frame.

    Returns ``{"count", "raw_count", "points"}`` where ``points`` is N×2 in
    (row, column) source-frame pixels — Crowd Shield's convention.  ``count``
    carries the ShanghaiTech correction only when ``CROWD_PET_CORRECTION`` is
    on; otherwise it equals ``raw_count`` so nothing is silently adjusted.
    """
    cs_points, _ = _crowdshield()
    _pet()
    result = cs_points.count_points(frame_bgr)
    points = np.asarray(result.get("points", []), dtype=np.float32).reshape(-1, 2)
    raw = int(result.get("raw_count", len(points)))
    if not config.PET_CORRECTION_ENABLED:
        return {"count": float(raw), "raw_count": raw, "points": points}
    count = result.get("count")
    return {"count": float(count) if count is not None else float(raw),
            "raw_count": raw, "points": points}


def estimate_density(frame_bgr: np.ndarray) -> dict:
    """Density map for one BGR frame: ``{"count", "density", "scale"}``.

    ``density`` is the 1/8-resolution map whose sum is ``count``; zones are
    integrated from that map rather than from the image area.
    """
    _, cs_density = _crowdshield()
    _dmcount()
    return cs_density.estimate_density(frame_bgr, weights=str(DMCOUNT_WEIGHTS))


def info() -> dict:
    """What is installed, for /api/system and for the operator audit trail."""
    return {
        "backend": "crowdshield",
        "licence": "academic-use (Khushi-7/crowd-shield-counting, Crowd Shield)",
        "models_dir": str(MODEL_DIR),
        "pet": {
            "weights": PET_WEIGHTS.name,
            "version": PET_VERSION,
            "sha256": _hashes.get(PET_WEIGHTS.name, PET_SHA256),
            "correction": list(PET_CORRECTION),
            "correction_enabled": config.PET_CORRECTION_ENABLED,
            "loaded": "pet" in _loaded,
        },
        "dmcount": {
            "weights": DMCOUNT_WEIGHTS.name,
            "version": DMCOUNT_VERSION,
            "sha256": _hashes.get(DMCOUNT_WEIGHTS.name, DMCOUNT_SHA256),
            "loaded": "dmcount" in _loaded,
        },
    }


def available() -> tuple[bool, str | None]:
    """Whether both checkpoints and the vendored definitions can be used."""
    try:
        _crowdshield()
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return False, f"vendored crowdshield import failed: {exc}"
    for path, digest in ((PET_WEIGHTS, PET_SHA256), (DMCOUNT_WEIGHTS, DMCOUNT_SHA256)):
        if not path.exists():
            return False, f"{path.name} not found in {path.parent}"
        try:
            verify(path, digest)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
    return True, None
