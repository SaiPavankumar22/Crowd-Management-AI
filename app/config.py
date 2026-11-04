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

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".wmv")
