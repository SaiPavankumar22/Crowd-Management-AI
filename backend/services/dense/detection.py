"""Count people in one photo and draw what was found.

`count_people` uses the most accurate counter that can run: the PET point model, then the
DM-Count density model, then YOLO boxes (optionally with faces whose bodies are hidden).
No Streamlit here: models load lazily and stay cached per process, so the same
code serves the app, the evaluation scripts and the tests.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import DEFAULT_MODEL_KEY, MODEL_REGISTRY, REPO_ROOT, NotEnoughMemory

# Ultralytics keeps at most 300 boxes per image by default, which silently caps
# counts in dense crowds. Counting needs a far higher ceiling.
MAX_DETECTIONS = 3000

# A Git LFS pointer (repo cloned without LFS) is a tiny text file, not weights.
_LFS_POINTER_MAX_BYTES = 1024


def _weights_source(model_key: str) -> str:
    cfg = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])
    path = REPO_ROOT / cfg["path"]
    if path.exists() and path.stat().st_size > _LFS_POINTER_MAX_BYTES:
        return str(path)
    # Official Ultralytics weights download by name; custom ones fall back to yolov8n.
    return path.name if path.name.startswith("yolov8") else "yolov8n.pt"


@lru_cache(maxsize=4)
def load_detector(model_key: str):
    """Return a YOLO model, or None when Ultralytics isn't installed."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    return YOLO(_weights_source(model_key))


@lru_cache(maxsize=1)
def _face_cascades():
    import cv2

    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("Haar face cascades need OpenCV 4.x; OpenCV 5 removed them.")
    root = cv2.data.haarcascades
    return (cv2.CascadeClassifier(root + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(root + "haarcascade_profileface.xml"))


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def _contains_centre(outer, inner) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    cx, cy = ix + iw // 2, iy + ih // 2
    return ox <= cx <= ox + ow and oy <= cy <= oy + oh


def _hog_people(gray: np.ndarray) -> list[tuple]:
    import cv2

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
    return [tuple(int(v) for v in box) for box in boxes]


def detect_people(img: np.ndarray, model_key: str = DEFAULT_MODEL_KEY, *,
                  use_faces: bool = False, conf: float = 0.25, imgsz: int = 640,
                  max_det: int = MAX_DETECTIONS, allow_fallback: bool = True) -> dict:
    """Count people in a BGR image. Boxes are (x, y, width, height).

    YOLO finds bodies. If YOLO is missing or fails and `allow_fallback` is set,
    OpenCV's HOG person detector is used instead.

    `use_faces` restores the original app's add-on: Haar cascades add faces that
    don't sit inside a detected body. It is off by default: OpenCV 5 dropped the
    cascades, and not processing faces is better for privacy. The evaluation
    still runs it (on OpenCV 4) to measure what it contributed.
    """
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cfg = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY[DEFAULT_MODEL_KEY])

    detector = model_key
    try:
        model = load_detector(model_key)
        if model is None:
            raise RuntimeError("Ultralytics is not installed")
        result = model(img, classes=cfg["classes"], conf=conf, imgsz=imgsz,
                       max_det=max_det, verbose=False)[0]
        bodies = [(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                  for x1, y1, x2, y2 in result.boxes.xyxy.tolist()]
    except Exception:
        # OpenCV 5 dropped the HOG people detector, so there may be no fallback.
        if not allow_fallback or not hasattr(cv2, "HOGDescriptor"):
            raise
        bodies, detector = _hog_people(gray), "hog"

    extra_heads = []
    if use_faces:
        h, w = gray.shape
        min_side = max(20, min(h, w) // 20)
        faces = [tuple(int(v) for v in face)
                 for cascade in _face_cascades()
                 for face in cascade.detectMultiScale(gray, 1.1, 4,
                                                      minSize=(min_side, min_side))]
        extra_heads = [face for face in faces
                       if not any(_contains_centre(body, face) or _iou(body, face) > 0.15
                                  for body in bodies)]

    return {
        "count":        len(bodies) + len(extra_heads),
        "body_count":   len(bodies),
        "head_count":   len(extra_heads),
        "person_boxes": bodies,
        "extra_heads":  extra_heads,
        "detector":     detector,
    }


def render_overlay(img: np.ndarray, person_boxes, extra_heads,
                   density: np.ndarray | None = None, points=None) -> np.ndarray:
    """Heat overlay of where people are.

    With head points (from the point model), each head gets a dot and the heat
    shows where they cluster. Otherwise the heat comes from the density map, or
    from where YOLO's boxes pile up, with bodies in green and extra faces in yellow.
    """
    import cv2

    h, w = img.shape[:2]
    if points is not None:
        from .monitor import points_to_grid

        density = points_to_grid(points, img.shape)
        person_boxes, extra_heads = [], []
    if density is not None:
        heat = cv2.resize(density.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        heat = cv2.GaussianBlur(heat, (0, 0), 4)
    else:
        heat = np.zeros((h, w), dtype=np.float32)
        for x, y, bw, bh in [*person_boxes, *extra_heads]:
            heat[max(0, y):min(y + bh, h), max(0, x):min(x + bw, w)] += 1.0
    if heat.max() > 0:
        heat /= heat.max()
    heat_bgr = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # Colour only where people are, fading in with density, so empty parts of the
    # photo keep their natural colours instead of a blue tint.
    alpha = (0.55 * np.clip(heat / 0.15, 0, 1))[..., None]
    overlay = (img * (1 - alpha) + heat_bgr * alpha).astype(np.uint8)
    for boxes, colour in ((person_boxes, (0, 255, 0)), (extra_heads, (0, 255, 255))):
        for x, y, bw, bh in boxes:
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), colour, 2)
    if points is not None and len(points):
        overlay = draw_head_dots(overlay, points)
    return overlay


def draw_head_dots(img: np.ndarray, points, colour=(0, 255, 0)) -> np.ndarray:
    """A big, see-through dot on every head (points as row, col), with a thin white ring.

    Easy to spot on a small screen, yet the people stay visible underneath: the dots are drawn
    on a copy and blended in. Every page that shows head points uses this, so they all match.
    """
    import cv2

    h, w = img.shape[:2]
    radius = max(5, round(min(h, w) / 110))
    dots = img.copy()
    for row, col in points:
        cv2.circle(dots, (int(col), int(row)), radius, colour, -1, cv2.LINE_AA)
    out = cv2.addWeighted(dots, 0.5, img, 0.5, 0)
    for row, col in points:
        cv2.circle(out, (int(col), int(row)), radius, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def count_people(img: np.ndarray, model_key: str = DEFAULT_MODEL_KEY) -> dict:
    """The app's count for one photo.

    In order of accuracy on the ShanghaiTech test photos: the point model (PET),
    then the density model (DM-Count), then YOLO boxes, using the first one that
    can run here (the first two need PyTorch and a one-time weights download).
    YOLO only runs as the last resort: it finds a small share of the people in
    a dense crowd, and running it alongside would only slow the app down.
    """
    base = {"body_count": 0, "head_count": 0, "person_boxes": [], "extra_heads": [],
            "density": None, "points": None, "check": None}
    try:
        from .points import count_points

        pts = count_points(img)
    except NotEnoughMemory:
        raise  # falling back to a model that needs as much memory would only crash the server
    except Exception:
        pts = None
    if pts is not None:
        from .crosscheck import second_opinion

        return {**base, "count": int(round(pts["count"])), "points": pts["points"],
                "counter": "points", "check": second_opinion(img, pts["count"])}
    try:
        from .density import estimate_density

        dens = estimate_density(img)
        return {**base, "count": int(round(dens["count"])), "density": dens["density"],
                "counter": "density"}
    except NotEnoughMemory:
        raise
    except Exception:
        pass
    det = detect_people(img, model_key)
    return {**base, **det, "counter": "detector"}


def analyse_image_bytes(image_bytes: bytes, model_key: str = DEFAULT_MODEL_KEY) -> dict:
    """Decode an uploaded photo, count people and draw the overlay."""
    import cv2

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"count": 0, "body_count": 0, "head_count": 0,
                "counter": None, "check": None, "overlay": None, "model_key": model_key}
    res = count_people(img, model_key)
    return {
        "count":          res["count"],
        "body_count":     res["body_count"],
        "head_count":     res["head_count"],
        "counter":        res["counter"],
        "check":          res.get("check"),
        "overlay":        render_overlay(img, res["person_boxes"], res["extra_heads"],
                                         res["density"], res["points"]),
        "model_key":      model_key,
    }
