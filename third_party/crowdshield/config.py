"""Shared constants: zones, risk tiers, density standards and detection models."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Photos are shrunk so their longer side is at most this many pixels before counting. The
# networks' memory grows with pixel count, so a small server needs a smaller limit.
FULL_MAX_SIDE = 2048
# Measured peak memory of PET alone on a Kaggle CPU (results/pet_memory.json): 640 px 1.5 GB,
# 800 px 2.4 GB, 1024 px 2.8 GB, 1280 px 3.6 GB, 2048 px 8.4 GB. Streamlit Community Cloud gives
# an app about 2.7 GB, so it counts at 640 px; counts stay within a few percent.
SMALL_SERVER_MAX_SIDE = 640
# Counted in tiles on a small server, so photos keep this much detail (the same accuracy on
# ShanghaiTech as 2048 px: results/size_accuracy.json) instead of shrinking to fit in memory.
SMALL_SERVER_WORK_SIDE = 1024
SMALL_SERVER_BYTES = 6 * 1024 ** 3


def memory_limit_bytes() -> int | None:
    """The memory this process may use: the container's limit if it has one, else total RAM."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:  # "max" or a huge number means no limit
            return int(raw)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def on_streamlit_cloud() -> bool:
    """Streamlit Community Cloud checks apps out under /mount/src."""
    return str(REPO_ROOT).startswith("/mount/src")


class NotEnoughMemory(RuntimeError):
    """The server can't hold the counting model for a photo of any size we'd accept."""


# Peak memory (MB) of the whole app while PET counts one photo at each size, measured on a
# Kaggle CPU (results/pet_memory.json, pet_memory_small.json, app_memory.json). Loading the weights
# alone takes about 0.9 GB, so no photo size runs in much less than that. Sizes in between are interpolated.
PET_PEAK_MB = {320: 920, 384: 1140, 448: 1140, 512: 1150, 576: 1520, 640: 1520,
               800: 2400, 1024: 2800, 1280: 3600, 1600: 5500, 2048: 8500}
MIN_SIDE = 320
# Use at most this share of the memory limit. The Streamlit Cloud server showed a 3.2 GB limit and was
# still killed at an estimated 2.7 GB (85%), because the numbers above come from a different machine.
SAFETY = 0.6


def estimated_peak_mb(side: int) -> float:
    sizes = sorted(PET_PEAK_MB)
    side = max(sizes[0], min(side, sizes[-1]))
    for lo, hi in zip(sizes, sizes[1:]):
        if lo <= side <= hi:
            return PET_PEAK_MB[lo] + (PET_PEAK_MB[hi] - PET_PEAK_MB[lo]) * (side - lo) / (hi - lo)
    return float(PET_PEAK_MB[sizes[-1]])


def memory_in_use_bytes() -> int | None:
    """Memory the container is using now (anonymous memory, not reclaimable file cache)."""
    try:
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            if line.startswith("anon "):
                return int(line.split()[1])
    except OSError:
        pass
    try:
        return int(Path("/sys/fs/cgroup/memory.current").read_text())
    except OSError:
        return None


def own_rss_bytes() -> int:
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError, IndexError):
        return 0


def fit_max_side(limit_bytes: int | None = None, in_use_bytes: int | None = None,
                 own_bytes: int | None = None, ceiling: int | None = None) -> int:
    """The largest photo size whose estimated peak fits in the memory left, never above `ceiling`.

    Raises NotEnoughMemory when even the smallest size would not fit, so the app can say so
    instead of being killed by the server. With no memory limit to read (a laptop), the ceiling is used.
    """
    ceiling = ceiling or default_max_side()
    limit = memory_limit_bytes() if limit_bytes is None else limit_bytes
    if limit is None or limit >= 64 * 1024 ** 3:
        return ceiling
    used = memory_in_use_bytes() if in_use_bytes is None else in_use_bytes
    own = own_rss_bytes() if own_bytes is None else own_bytes
    others = max(0, (used or 0) - own)           # memory held by other processes in the container
    budget_mb = (SAFETY * limit - others) / 1e6
    for side in range(ceiling, MIN_SIDE - 1, -32):
        if estimated_peak_mb(side) <= budget_mb:
            return side
    raise NotEnoughMemory(
        f"This server has about {limit / 1e9:.1f} GB of memory, and counting needs roughly "
        f"{estimated_peak_mb(MIN_SIDE) / 1000:.1f} GB even for a small photo. Run Crowd Shield on "
        "your own computer, or use the sample photos.")


def default_work_side() -> int:
    """The photo size the point model should see: the most detail worth paying for.

    A server too small to run that size in one pass counts it in tiles instead (see points.py), so
    the size no longer has to shrink with memory. CROWDSHIELD_MAX_SIDE overrides it.
    """
    if os.environ.get("CROWDSHIELD_MAX_SIDE"):
        return int(os.environ["CROWDSHIELD_MAX_SIDE"])
    limit = memory_limit_bytes()
    if on_streamlit_cloud() or (limit is not None and limit < SMALL_SERVER_BYTES):
        return SMALL_SERVER_WORK_SIDE
    return FULL_MAX_SIDE


def default_max_side() -> int:
    """CROWDSHIELD_MAX_SIDE if set; otherwise smaller on Streamlit Cloud or a small machine."""
    if os.environ.get("CROWDSHIELD_MAX_SIDE"):
        return int(os.environ["CROWDSHIELD_MAX_SIDE"])
    limit = memory_limit_bytes()
    if on_streamlit_cloud() or (limit is not None and limit < SMALL_SERVER_BYTES):
        return SMALL_SERVER_MAX_SIDE
    return FULL_MAX_SIDE

SEGMENTS = ["A", "B", "C", "D", "E"]
MAX_IMAGES_PER_SEGMENT = 4

RISK_LEVELS = ["Safe", "Moderate", "High Risk", "Critical"]

DEFAULT_THRESHOLDS = {
    "moderate": 30,
    "high":     60,
    "critical": 90,
}

# People per square metre at which each risk tier starts, after Fruin's
# Level of Service scale for pedestrian spaces.
FRUIN_PRESETS = {
    "Conservative (outdoor open areas)": {
        "moderate": 0.5,
        "high":     1.0,
        "critical": 2.0,
    },
    "Standard (mixed venues)": {
        "moderate": 1.0,
        "high":     2.0,
        "critical": 3.5,
    },
    "Crowded events / concerts": {
        "moderate": 1.5,
        "high":     3.0,
        "critical": 4.5,
    },
    "Enclosed corridors / gates": {
        "moderate": 0.5,
        "high":     1.0,
        "critical": 1.5,
    },
}

MODEL_REGISTRY = {
    "crowd_yolo": {
        "path":    "models/crowd_yolo.pt",
        "classes": [0, 1],
        "label":   "Custom Crowd YOLO (VisDrone best.pt)",
    },
    "yolov8l": {
        "path":    "models/yolov8l.pt",
        "classes": [0],
        "label":   "YOLOv8l (COCO standard large)",
    },
}
DEFAULT_MODEL_KEY = "crowd_yolo"

# Who counted a photo, as shown in the app.
COUNTER_LABELS = {
    "points":   "main counting model (PET)",
    "density":  "backup counting model (DM-Count)",
    "detector": "basic person detector (YOLO)",
}

# Photo source -> model key; None means "use the globally selected model".
SOURCE_MODEL_MAP = {
    "auto":   None,
    "drone":  "crowd_yolo",
    "ground": "yolov8l",
}
