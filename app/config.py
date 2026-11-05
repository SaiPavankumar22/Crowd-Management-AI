"""Central configuration. Every value can be overridden with an env var."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Heavy folders (uploads / samples / alert DB) can live anywhere -- point them
# at a drive with real free space via env vars (see start_dev.sh).
def _dir(env_var: str, default: str) -> str:
    path = os.environ.get(env_var, default)
    os.makedirs(path, exist_ok=True)
    return path

DATA_DIR = _dir("CROWD_DATA_DIR", os.path.join(BASE_DIR, "data"))
UPLOADS_DIR = _dir("CROWD_UPLOADS_DIR", os.path.join(BASE_DIR, "uploads"))
SAMPLES_DIR = _dir("CROWD_SAMPLES_DIR", os.path.join(BASE_DIR, "samples"))
STATIC_DIR = os.path.join(BASE_DIR, "static")


def resolve_media(path: str) -> str:
    """Return a usable local path for a stored camera source.

    Camera configs persist absolute paths, so moving the data folder (or
    the whole project) would otherwise leave every saved camera dead. If the
    recorded file is gone we look for the same filename in the current
    samples/uploads folders before giving up.
    """
    if not path or "://" in str(path) or str(path).strip().isdigit():
        return path
    if os.path.isfile(path):
        return path
    name = os.path.basename(str(path).replace("\\", "/"))
    for base in (SAMPLES_DIR, UPLOADS_DIR):
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            return candidate
    return path


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


# --- Model -----------------------------------------------------------------
# YOLO26-nano (Ultralytics, released Jan 2026): NMS-free end-to-end head,
# small-object aware - the current real-time pick for dense crowd counting.
MODEL_NAME = os.environ.get("CROWD_MODEL", "yolo26n.pt")
FALLBACK_MODELS = ["yolo11n.pt", "yolov8n.pt"]
CONF_THRESHOLD = float(os.environ.get("CROWD_CONF", "0.30"))
IMGSZ = int(os.environ.get("CROWD_IMGSZ", "640"))

# --- Frame / throughput ----------------------------------------------------
# Per-camera inference rate. 5 FPS is plenty for crowd counting and keeps
# CPU usage sane with several cameras running at once.
TARGET_INFER_FPS = float(os.environ.get("CROWD_TARGET_INFER_FPS", "5"))
MAX_FRAME_WIDTH = int(os.environ.get("CROWD_MAX_FRAME_WIDTH", "1280"))
JPEG_QUALITY = int(os.environ.get("CROWD_JPEG_QUALITY", "65"))
LOOP_FILE_SOURCES = env_bool("CROWD_LOOP", True)

# --- Alerts ----------------------------------------------------------------
DEFAULT_WARN_PCT = float(os.environ.get("CROWD_WARN_PCT", "75"))
DEFAULT_ZONE_CAPACITY = int(os.environ.get("CROWD_ZONE_CAPACITY", "100"))
DEFAULT_ALERT_SUSTAIN_S = float(os.environ.get("CROWD_ALERT_SUSTAIN", "5"))
DEFAULT_ALERT_COOLDOWN_S = float(os.environ.get("CROWD_ALERT_COOLDOWN", "60"))

# --- Dashboard -------------------------------------------------------------
WS_PERIOD_S = float(os.environ.get("CROWD_WS_PERIOD", "0.6"))
STREAM_FPS = int(os.environ.get("CROWD_STREAM_FPS", "12"))

# --- Reliability / quality -----------------------------------------------
FRAME_STALE_S = float(os.environ.get("CROWD_FRAME_STALE_S", "8"))
FROZEN_FRAME_S = float(os.environ.get("CROWD_FROZEN_FRAME_S", "10"))
QUALITY_SAMPLE_PERIOD_S = float(os.environ.get("CROWD_QUALITY_SAMPLE_PERIOD_S", "1"))
MIN_BRIGHTNESS = float(os.environ.get("CROWD_MIN_BRIGHTNESS", "12"))
MIN_CONTRAST = float(os.environ.get("CROWD_MIN_CONTRAST", "8"))
MIN_BLUR_SCORE = float(os.environ.get("CROWD_MIN_BLUR_SCORE", "18"))

# --- Inference / decoding -------------------------------------------------
BATCH_SIZE = int(os.environ.get("CROWD_BATCH_SIZE", "1"))
MAX_BATCH_WAIT_MS = float(os.environ.get("CROWD_MAX_BATCH_WAIT_MS", "8"))
FRAME_QUEUE_SIZE = int(os.environ.get("CROWD_FRAME_QUEUE_SIZE", "2"))
DECODER = os.environ.get("CROWD_DECODER", "auto").lower()
# Stale results are worse than dropped frames: anything older than this is
# discarded rather than analysed (improvement.md §8).
MAX_FRAME_AGE_MS = float(os.environ.get("CROWD_MAX_FRAME_AGE_MS", "500"))

# --- Tiled inference (gated: only when people are small in frame) ----------
TILING = os.environ.get("CROWD_TILING", "auto").lower()  # auto | off
TILE_ROWS = int(os.environ.get("CROWD_TILE_ROWS", "3"))
TILE_COLS = int(os.environ.get("CROWD_TILE_COLS", "3"))
TILE_OVERLAP = float(os.environ.get("CROWD_TILE_OVERLAP", "0.15"))
TILE_MIN_BOX_HEIGHT_PX = float(os.environ.get("CROWD_TILE_MIN_BOX_HEIGHT_PX", "40"))

# --- Detail escalation (recover people the full-frame pass cannot resolve) --
# When a frame yields almost nobody, a cheap tiled probe checks whether that is
# the truth or a resolution failure; only a clear disagreement pays for the
# high-resolution pass. See app/detail.py for the measurements behind this.
DETAIL_PASS = env_bool("CROWD_DETAIL_PASS", True)
DETAIL_PROBE_BELOW = int(os.environ.get("CROWD_DETAIL_PROBE_BELOW", "6"))
DETAIL_PROBE_IMGSZ = int(os.environ.get("CROWD_DETAIL_PROBE_IMGSZ", "640"))
DETAIL_PROBE_GRID = int(os.environ.get("CROWD_DETAIL_PROBE_GRID", "2"))
DETAIL_IMGSZ = int(os.environ.get("CROWD_DETAIL_IMGSZ", "1280"))
DETAIL_GRID = int(os.environ.get("CROWD_DETAIL_GRID", "3"))
DETAIL_IDLE_PROBE_EVERY = int(os.environ.get("CROWD_DETAIL_IDLE_PROBE_EVERY", "4"))
# Leaving detail mode needs sustained evidence, not one quiet frame.
DETAIL_EXIT_BELOW = int(os.environ.get("CROWD_DETAIL_EXIT_BELOW", "3"))
DETAIL_EXIT_FRAMES = int(os.environ.get("CROWD_DETAIL_EXIT_FRAMES", "10"))

# --- Mode router / degraded states ----------------------------------------
MODE_ROUTER = env_bool("CROWD_MODE_ROUTER", True)
DENSE_BOX_HEIGHT_PX = float(os.environ.get("CROWD_DENSE_BOX_HEIGHT_PX", "32"))
MODE_HYSTERESIS_FRAMES = int(os.environ.get("CROWD_MODE_HYSTERESIS_FRAMES", "3"))
# Density output is advisory until an operator signs off a validation gate.
DENSITY_MODEL = os.environ.get("CROWD_DENSITY_MODEL", "").strip()
DENSITY_VALIDATED = env_bool("CROWD_DENSITY_VALIDATED", False)
DENSITY_PERIOD_S = float(os.environ.get("CROWD_DENSITY_PERIOD_S", "1.5"))
CALIBRATED_DENSITY_ENABLED = env_bool("CROWD_CALIBRATED_DENSITY", False)

# --- Alerts / calibration -------------------------------------------------
WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
WEBHOOK_SECRET = os.environ.get("ALERT_WEBHOOK_SECRET", "").strip()
WEBHOOK_TIMEOUT_S = float(os.environ.get("ALERT_WEBHOOK_TIMEOUT_S", "5"))
WEBHOOK_RETRIES = int(os.environ.get("ALERT_WEBHOOK_RETRIES", "3"))
CALIBRATION_ENABLED = env_bool("CROWD_CALIBRATION_ENABLED", True)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".wmv")
