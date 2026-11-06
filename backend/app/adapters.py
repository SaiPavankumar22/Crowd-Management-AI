"""Detector and density adapter boundaries."""
from __future__ import annotations

import abc
import importlib
import importlib.util
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, metrics
from .tiling import (TilingPolicy, compute_tiles, crop, merge_tile_results,
                     offset_boxes)

PERSON_CLASS = 0

# Result types
@dataclass(frozen=True)
class DetectionResult:
    """The only thing the pipeline is allowed to know about a detector."""
    boxes: np.ndarray            # (n, 4) xyxy in frame pixels
    confidences: np.ndarray      # (n,)
    class_ids: np.ndarray        # (n,)
    inference_ms: float
    model_name: str
    model_version: str
    image_size: tuple[int, int]
    tiled: bool = False
    tile_count: int = 0
    # "primary" | "detail": how this result was produced. A detail result is a
    # lower bound recovered at higher resolution, not a validated count.
    tier: str = "primary"
    # What the plain full-frame pass said, so the router can see the jump.
    primary_count: int | None = None

    @property
    def count(self) -> int:
        return int(len(self.boxes))

    def median_box_height(self) -> float:
        if len(self.boxes) == 0:
            return 0.0
        return float(np.median(self.boxes[:, 3] - self.boxes[:, 1]))

    def low_confidence_ratio(self, threshold: float = 0.5) -> float:
        if len(self.confidences) == 0:
            return 0.0
        return float(np.mean(self.confidences < threshold))

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "median_box_height": round(self.median_box_height(), 1),
            "inference_ms": round(self.inference_ms, 2),
            "model": self.model_name,
            "model_version": self.model_version,
            "image_size": list(self.image_size),
            "tiled": self.tiled,
            "tiles": self.tile_count,
        }

@dataclass(frozen=True)
class DensityResult:
    """Advisory density estimate. Never merged into the detector count."""
    total_count: float
    zone_counts: dict
    confidence: str
    inference_ms: float
    model_name: str
    model_version: str
    density_map: np.ndarray | None = None

@dataclass(frozen=True)
class PointResult:
    """Head-point estimate in original-frame ``(x, y)`` pixel coordinates.

    Point models intentionally have a separate contract from box detectors:
    points support occupancy and calibrated density in occluded crowds, but do
    not imply an identity that can be followed between frames.
    """
    points: np.ndarray              # (n, 2), x/y pixels in the source frame
    raw_count: int
    corrected_count: float | None
    zone_counts: dict
    confidence: str
    inference_ms: float
    model_name: str
    model_version: str

    @property
    def total_count(self) -> float:
        return (float(self.corrected_count) if self.corrected_count is not None
                else float(self.raw_count))

# Model metadata / licence registry
@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    version: str
    license: str
    task: str = "person-detection"
    imgsz: int = 640
    expected_vram_mb: int = 0
    commercial_use: bool = True
    notes: str = ""
    # Optional Python package this family needs at runtime. Empty means the
    # weights run on the dependencies the service already ships.
    requires: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "backend": self.backend,
            "version": self.version,
            "license": self.license,
            "task": self.task,
            "imgsz": self.imgsz,
            "expected_vram_mb": self.expected_vram_mb,
            "commercial_use": self.commercial_use,
            "notes": self.notes,
            "requires": self.requires,
        }

# Licence terms checked before deployment, not assumed: Ultralytics (YOLOv8,
# YOLO11, YOLO26) ships AGPL-3.0; RF-DETR Nano-Small-Large weights are
# Apache-2.0; RT-DETR is Apache-2.0. Any replacement must be re-checked.
KNOWN_MODELS: dict[str, ModelSpec] = {
    "yolo26n.pt": ModelSpec("yolo26n.pt", "ultralytics", "26-nano", "AGPL-3.0",
                            imgsz=640, expected_vram_mb=700,
                            commercial_use=False,
                            notes="NMS-free head; AGPL network-service obligation"),
    "yolo11n.pt": ModelSpec("yolo11n.pt", "ultralytics", "11-nano", "AGPL-3.0",
                            imgsz=640, expected_vram_mb=650,
                            commercial_use=False,
                            notes="Fallback"),
    "yolov8n.pt": ModelSpec("yolov8n.pt", "ultralytics", "8-nano", "AGPL-3.0",
                            imgsz=640, expected_vram_mb=600,
                            commercial_use=False,
                            notes="Fallback"),
    "yolov8x.pt": ModelSpec("yolov8x.pt", "ultralytics", "8-xlarge", "AGPL-3.0",
                            imgsz=960, expected_vram_mb=1800,
                            commercial_use=False,
                            notes="Accuracy reference for benchmarking"),
    # --- RT-DETR (Apache-2.0): transformer decoder, no NMS ----------------
    "rtdetr-l.pt": ModelSpec("rtdetr-l.pt", "rtdetr", "rtdetr-l", "Apache-2.0",
                             imgsz=640, expected_vram_mb=2200, commercial_use=True,
                             notes="DETR decoder without NMS; better on overlapping "
                                   "people, ~2-3x the compute of YOLO26n"),
    "rtdetr-x.pt": ModelSpec("rtdetr-x.pt", "rtdetr", "rtdetr-x", "Apache-2.0",
                             imgsz=640, expected_vram_mb=3200, commercial_use=True,
                             notes="Largest RT-DETR; accuracy reference"),
    # --- RF-DETR (Apache-2.0), optional `rfdetr` package -------------------
    "rf-detr-base.pth": ModelSpec("rf-detr-base.pth", "rfdetr", "rf-detr-base",
                                  "Apache-2.0", imgsz=640, expected_vram_mb=2600,
                                  commercial_use=True, requires="rfdetr",
                                  notes="DINOv2 backbone; `pip install rfdetr`"),
    "rf-detr-large.pth": ModelSpec("rf-detr-large.pth", "rfdetr", "rf-detr-large",
                                   "Apache-2.0", imgsz=640, expected_vram_mb=4200,
                                   commercial_use=True, requires="rfdetr",
                                   notes="DINOv2 backbone; more accurate, slower"),
    # --- D-FINE (Apache-2.0), optional `dfine` package ---------------------
    "dfine_m_coco.pt": ModelSpec("dfine_m_coco.pt", "dfine", "dfine-m", "Apache-2.0",
                                 imgsz=640, expected_vram_mb=1800,
                                 commercial_use=True, requires="dfine",
                                 notes="D-FINE medium; needs the `dfine` package"),
    "dfine_l_coco.pt": ModelSpec("dfine_l_coco.pt", "dfine", "dfine-l", "Apache-2.0",
                                 imgsz=640, expected_vram_mb=2400,
                                 commercial_use=True, requires="dfine",
                                 notes="D-FINE large; needs the `dfine` package"),
}

def _weight_path(name: str) -> str:
    """Prefer the organized local model directory while preserving model IDs.

    API callers and persisted camera profiles continue to use names such as
    ``yolo26n.pt``; only the filesystem lookup moves into ``models/``.
    Missing weights still fall through to Ultralytics' normal download path.
    """
    candidate = Path(config.MODELS_DIR) / Path(name).name
    return str(candidate) if candidate.is_file() else name

def model_available(name: str) -> bool:
    """True when this model's backend dependencies are importable.

    Weights for the Ultralytics families download on first load, so they are
    available wherever the service runs; RF-DETR and D-FINE live in separate
    packages and must be installed before the picker offers them.
    """
    requires = spec_for(name).requires
    if not requires:
        return True
    try:
        return importlib.util.find_spec(requires) is not None
    except (ImportError, ValueError):
        return False

def spec_for(name: str, imgsz: int | None = None) -> ModelSpec:
    spec = KNOWN_MODELS.get(name)
    if spec is None:
        backend = "onnx" if name.endswith(".onnx") else "ultralytics"
        spec = ModelSpec(name, backend, "unregistered", "unknown",
                         imgsz=imgsz or 640, commercial_use=False,
                         notes="Unregistered weights: verify licence before deploying")
    if imgsz and imgsz != spec.imgsz:
        return ModelSpec(spec.name, spec.backend, spec.version, spec.license,
                         spec.task, imgsz, spec.expected_vram_mb,
                         spec.commercial_use, spec.notes, spec.requires)
    return spec

# Detector adapters
class DetectorAdapter(abc.ABC):
    """Common surface for every detector backend."""

    backend = "abstract"

    @property
    @abc.abstractmethod
    def spec(self) -> ModelSpec: ...

    @abc.abstractmethod
    def detect_batch(self, frames: list[np.ndarray], *,
                     imgsz: int | None = None,
                     camera_ids: list[str] | None = None,
                     ) -> list[DetectionResult]: ...

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return self.detect_batch([frame])[0]

    @abc.abstractmethod
    def info(self) -> dict: ...

class UltralyticsDetector(DetectorAdapter):
    """YOLO-family adapter, also used for Ultralytics-exported .onnx weights."""

    def __init__(self, name: str, conf: float = 0.30, imgsz: int = 640,
                 device=None, half: bool = False):
        self.name = name
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.backend = "onnx" if name.endswith(".onnx") else "ultralytics"
        self._model = None
        self._lock = threading.RLock()
        self._spec = spec_for(name, imgsz)

    # lifecycle ---------------------------------------------------------
    def load(self):
        from ultralytics import YOLO

        with self._lock:
            self._model = YOLO(_weight_path(self.name))
        return self

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def set_profile(self, conf: float | None = None, imgsz: int | None = None):
        with self._lock:
            if conf is not None:
                self.conf = float(conf)
            if imgsz is not None:
                self.imgsz = int(imgsz)
            self._spec = spec_for(self.name, self.imgsz)

    # inference ---------------------------------------------------------
    def detect_batch(self, frames: list[np.ndarray], *,
                     imgsz: int | None = None,
                     camera_ids: list[str] | None = None,
                     ) -> list[DetectionResult]:
        if self._model is None:
            raise RuntimeError("detector not loaded")
        started = time.perf_counter()
        # `half` is deprecated in ultralytics 8.4 (it warns and will be removed);
        # fp32 on this class of GPU is not the bottleneck, decode is.
        kwargs = dict(conf=self.conf, classes=[PERSON_CLASS],
                      imgsz=imgsz or self.imgsz,
                      device=self.device, verbose=False)
        with self._lock:
            results = self._model.predict(frames, **kwargs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics.inc("godavari_inference_batches_total")
        metrics.inc("godavari_inference_frames_total", len(frames))
        metrics.inc("godavari_inference_seconds_total", elapsed_ms / 1000.0)
        return [self._to_result(result, frames[i], elapsed_ms / max(1, len(frames)), i)
                for i, result in enumerate(results)]

    def _to_result(self, result, frame: np.ndarray, per_frame_ms: float,
                   index: int) -> DetectionResult:
        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.zeros((0, 4), np.float32)
            confs = np.zeros((0,), np.float32)
            classes = np.zeros((0,), np.int32)
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            confs = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            classes = (result.boxes.cls.detach().cpu().numpy().astype(np.int32)
                       if result.boxes.cls is not None else
                       np.zeros(len(boxes), np.int32))
        h, w = frame.shape[:2]
        return DetectionResult(boxes, confs, classes, per_frame_ms,
                               self.name, self._spec.version, (w, h))

    def info(self) -> dict:
        base = self._spec.to_dict()
        base.update({"loaded": self.loaded, "conf": self.conf,
                     "model": self.name, "backend": self.backend,
                     "detector": "ultralytics"})
        return base
class RTDETRDetector(UltralyticsDetector):
    """RT-DETR through Ultralytics.

    Same adapter surface as YOLO, but the DETR family's decoder needs no NMS,
    and its weights are Apache-2.0 - the non-YOLO option that does not change
    the service's dependency set.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = "rtdetr"

    def load(self):
        from ultralytics import RTDETR

        with self._lock:
            self._model = RTDETR(_weight_path(self.name))
        return self

_RFDETR_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
    "2xlarge": "RFDETR2XLarge",
}

def _convert_detections(det, frame: np.ndarray, per_frame_ms: float,
                        name: str, version: str) -> DetectionResult:
    """Map a supervision-style Detections object onto DetectionResult."""
    boxes = getattr(det, "xyxy", None)
    boxes = (np.zeros((0, 4), np.float32) if boxes is None
             else np.asarray(boxes, np.float32).reshape(-1, 4))
    count = len(boxes)
    confs = getattr(det, "confidence", None)
    confs = (np.zeros(count, np.float32) if confs is None
             else np.asarray(confs, np.float32).reshape(-1))
    classes = getattr(det, "class_id", None)
    classes = (np.zeros(count, np.int32) if classes is None
               else np.asarray(classes, np.int32).reshape(-1))
    # COCO index 0 is "person" for every COCO-pretrained checkpoint here.
    keep = classes == PERSON_CLASS
    h, w = frame.shape[:2]
    return DetectionResult(boxes[keep], confs[keep], classes[keep], per_frame_ms,
                           name, version, (w, h))

class RFDetrDetector(DetectorAdapter):
    """RF-DETR (Roboflow, Apache-2.0) through the optional ``rfdetr`` package.

    Weights are the COCO checkpoints (``rf-detr-base.pth`` and friends); a local
    file is passed as ``pretrain_weights``, otherwise the package downloads its
    own. Everything below is lazy, so a deployment without the package simply
    reports ``available: false`` instead of failing to start.
    """

    backend = "rfdetr"

    def __init__(self, name: str, conf: float = 0.30, imgsz: int = 640,
                 device=None, half: bool = False):
        self.name = name
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self._model = None
        self._lock = threading.RLock()
        self._spec = spec_for(name, imgsz)

    def load(self):
        try:
            import rfdetr
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RF-DETR is not installed; run `pip install rfdetr` "
                "(Apache-2.0 code and weights) and switch again") from exc
        variant = Path(self.name).stem.split("-")[-1].lower()
        factory = getattr(rfdetr, _RFDETR_CLASSES.get(variant, "RFDETRBase"))
        kwargs = {}
        if os.path.isfile(self.name):
            kwargs["pretrain_weights"] = self.name
        with self._lock:
            self._model = factory(**kwargs)
        return self

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def set_profile(self, conf: float | None = None, imgsz: int | None = None):
        with self._lock:
            if conf is not None:
                self.conf = float(conf)
            if imgsz is not None:
                self.imgsz = int(imgsz)

    def detect_batch(self, frames: list[np.ndarray], *,
                     imgsz: int | None = None,
                     camera_ids: list[str] | None = None,
                     ) -> list[DetectionResult]:
        if self._model is None:
            raise RuntimeError("detector not loaded")
        started = time.perf_counter()
        with self._lock:
            raw = self._model.predict([np.asarray(f) for f in frames],
                                      threshold=self.conf)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results = list(raw) if isinstance(raw, (list, tuple)) else [raw] * len(frames)
        per_frame = elapsed_ms / max(1, len(frames))
        metrics.inc("godavari_inference_batches_total")
        metrics.inc("godavari_inference_frames_total", len(frames))
        metrics.inc("godavari_inference_seconds_total", elapsed_ms / 1000.0)
        return [_convert_detections(det, frames[i], per_frame,
                                    self.name, self._spec.version)
                for i, det in enumerate(results)]

    def info(self) -> dict:
        base = self._spec.to_dict()
        base.update({"loaded": self.loaded, "conf": self.conf,
                     "model": self.name, "backend": self.backend,
                     "detector": "rfdetr"})
        return base

class DFineDetector(DetectorAdapter):
    """D-FINE (Apache-2.0) through the optional ``dfine`` package.

    Upstream D-FINE has no stable pip inference entry point (its repository
    drives inference through ``engine.core.YAMLConfig``), so this adapter calls
    ``dfine.load_model(checkpoint, device=...)`` when the installed package
    exposes it and otherwise fails with an actionable message. A failed load
    never replaces the active model - the manager keeps the previous one.
    """

    backend = "dfine"

    def __init__(self, name: str, conf: float = 0.30, imgsz: int = 640,
                 device=None, half: bool = False):
        self.name = name
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self._model = None
        self._lock = threading.RLock()
        self._spec = spec_for(name, imgsz)

    def load(self):
        try:
            import dfine
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "D-FINE is not installed; install the `dfine` package or put "
                "Peterande/D-FINE on PYTHONPATH, then switch again") from exc
        loader = getattr(dfine, "load_model", None)
        if not callable(loader):
            raise RuntimeError(
                "the installed dfine package exposes no load_model(checkpoint); "
                "see docs/DETECTORS.md for the supported entry points")
        with self._lock:
            self._model = loader(self.name, device=self.device)
        return self

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def set_profile(self, conf: float | None = None, imgsz: int | None = None):
        with self._lock:
            if conf is not None:
                self.conf = float(conf)
            if imgsz is not None:
                self.imgsz = int(imgsz)

    def detect_batch(self, frames: list[np.ndarray], *,
                     imgsz: int | None = None,
                     camera_ids: list[str] | None = None,
                     ) -> list[DetectionResult]:
        if self._model is None:
            raise RuntimeError("detector not loaded")
        started = time.perf_counter()
        with self._lock:
            raw = self._model.predict([np.asarray(f) for f in frames])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results = list(raw) if isinstance(raw, (list, tuple)) else [raw] * len(frames)
        per_frame = elapsed_ms / max(1, len(frames))
        return [_convert_detections(det, frames[i], per_frame,
                                    self.name, self._spec.version)
                for i, det in enumerate(results)]

    def info(self) -> dict:
        base = self._spec.to_dict()
        base.update({"loaded": self.loaded, "conf": self.conf,
                     "model": self.name, "backend": self.backend,
                     "detector": "dfine"})
        return base

def build_detector(name: str, conf: float = 0.30, imgsz: int = 640,
                   device=None, half: bool = False) -> DetectorAdapter:
    """Pick the adapter class for a registered model name.

    The registry says which family a checkpoint belongs to; this function is the
    only place that maps a family onto an implementation.
    """
    spec = spec_for(name, imgsz)
    if spec.backend == "rtdetr":
        return RTDETRDetector(name, conf=conf, imgsz=imgsz, device=device, half=half)
    if spec.backend == "rfdetr":
        return RFDetrDetector(name, conf=conf, imgsz=imgsz, device=device, half=half)
    if spec.backend == "dfine":
        return DFineDetector(name, conf=conf, imgsz=imgsz, device=device, half=half)
    return UltralyticsDetector(name, conf=conf, imgsz=imgsz, device=device, half=half)

def merge_tiled(tile_results: list[DetectionResult], tile_rects: list,
                frame: np.ndarray, spec: ModelSpec, elapsed_ms: float,
                *, tier: str = "primary",
                primary_count: int | None = None) -> DetectionResult:
    """Offset per-tile boxes back into frame coordinates and de-duplicate."""
    all_boxes, all_scores = [], []
    for tile, result in zip(tile_rects, tile_results):
        if result.count:
            all_boxes.append(offset_boxes(result.boxes, tile))
            all_scores.append(result.confidences)
    if not all_boxes:
        merged_boxes = np.zeros((0, 4), np.float32)
        merged_scores = np.zeros((0,), np.float32)
    else:
        merged_boxes, merged_scores = merge_tile_results(
            np.concatenate(all_boxes), np.concatenate(all_scores))
    h, w = frame.shape[:2]
    return DetectionResult(
        merged_boxes, merged_scores, np.zeros(len(merged_boxes), np.int32),
        elapsed_ms, spec.name, spec.version, (w, h),
        tiled=True, tile_count=len(tile_rects), tier=tier,
        primary_count=primary_count,
    )

class TiledDetector(DetectorAdapter):
    """Wraps an adapter with gated tiled inference and cross-tile NMS.

    The gate is data-driven: tile only when the detector's own boxes say people
    are small, which is exactly when recall collapses in wide views.
    """

    def __init__(self, inner: DetectorAdapter, policy: TilingPolicy | None = None):
        self.inner = inner
        self.policy = policy or TilingPolicy()
        self.backend = inner.backend + "+tiled"

    @property
    def spec(self) -> ModelSpec:
        return self.inner.spec

    def info(self) -> dict:
        info = dict(self.inner.info())
        info.update({
            "backend": self.backend,
            "tiling": {"enabled": self.policy.enabled,
                       "grid": [self.policy.rows, self.policy.cols],
                       "overlap": self.policy.overlap,
                       "min_box_height_px": self.policy.min_box_height_px},
        })
        return info

    def detect(self, frame: np.ndarray) -> DetectionResult:
        base = self.inner.detect(frame)
        if not self.policy.should_tile(base.boxes):
            return base
        h, w = frame.shape[:2]
        tiles = compute_tiles(w, h, self.policy.rows, self.policy.cols,
                              self.policy.overlap)
        crops = [crop(frame, tile) for tile in tiles]
        started = time.perf_counter()
        tile_results = self.inner.detect_batch(crops)
        elapsed_ms = (time.perf_counter() - started) * 1000.0 + base.inference_ms
        metrics.inc("godavari_tiled_inferences_total")
        return merge_tiled(tile_results, tiles, frame, self.inner.spec, elapsed_ms)

    def detect_batch(self, frames: list[np.ndarray], *,
                     imgsz: int | None = None,
                     camera_ids: list[str] | None = None,
                     ) -> list[DetectionResult]:
        return [self.detect(frame) for frame in frames]

# Density adapter (advisory only until camera validation)
class DensityEstimator(abc.ABC):
    """Density/point-count adapter. Output is advisory, never operational."""

    name = "none"
    version = "0"
    validated = False

    @abc.abstractmethod
    def estimate(self, frame: np.ndarray, zone_masks: list | None = None) -> DensityResult: ...

    def info(self) -> dict:
        return {"name": self.name, "version": self.version,
                "validated": self.validated, "available": True}

class PointEstimator(abc.ABC):
    """Head-point estimation boundary used by dense-scene routing."""

    name = "none"
    version = "0"
    validated = False

    @abc.abstractmethod
    def estimate(self, frame: np.ndarray, zone_masks: list | None = None) -> PointResult: ...

    def info(self) -> dict:
        return {"name": self.name, "version": self.version,
                "validated": self.validated, "available": True}

class NullPointEstimator(PointEstimator):
    """An explicit unavailable state; absence must never resemble zero heads."""

    name = "head-points-unavailable"
    version = "0"
    validated = False

    def estimate(self, frame: np.ndarray, zone_masks: list | None = None) -> PointResult:
        raise RuntimeError("no head-point model configured")

    def info(self) -> dict:
        return {"name": self.name, "version": self.version, "validated": False,
                "available": False,
                "note": "Set CROWD_POINT_BACKEND=dense after installing dense checkpoints"}

def _point_zone_counts(points: np.ndarray, zone_masks: list | None,
                       frame_shape: tuple[int, int]) -> dict:
    """Count source-frame points inside normalised dashboard zone polygons."""
    if not zone_masks:
        return {}
    import cv2

    h, w = frame_shape
    counts: dict = {}
    for zone in zone_masks:
        key = zone.get("id", zone.get("name", "zone"))
        polygon = np.asarray(zone.get("polygon", []), dtype=np.float32)
        if len(polygon) < 3:
            counts[key] = 0
            continue
        polygon *= np.array([w, h], dtype=np.float32)
        counts[key] = sum(
            cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0
            for x, y in points
        )
    return counts

class DensePointEstimator(PointEstimator):
    """Adapter for the dense PET ``count_points`` service.

    The service package owns the PET network definition and checkpoint loading.
    This adapter imports its public API only, which keeps the app free from
    checkpoint-specific model code and preserves the original BGR input and
    point-coordinate convention.
    """

    name = "dense-pet"
    version = "pet-shab-vgg16bn"

    def __init__(self, validated: bool = False):
        # backend/app/dense.py pins the service definitions to models/pet_shab.pth
        # and verifies its checksum before anything is loaded.
        from . import dense

        self._count_points = dense.count_points
        self.validated = validated
        self._lock = threading.Lock()

    def estimate(self, frame: np.ndarray, zone_masks: list | None = None) -> PointResult:
        h, w = frame.shape[:2]
        started = time.perf_counter()
        with self._lock:
            result = self._count_points(frame)
        if not isinstance(result, dict):
            raise RuntimeError("Dense PET returned an unsupported result")
        raw_points = np.asarray(result.get("points", []), dtype=np.float32).reshape(-1, 2)
        # Service documents points as (row, column); Godavari uses x/y.
        points = raw_points[:, ::-1] if len(raw_points) else raw_points
        raw_count = int(result.get("raw_count", len(points)))
        corrected = result.get("count")
        if corrected is not None:
            corrected = float(corrected)
            # A count equal to the raw points carries no correction; keep
            # corrected_count null so nothing looks calibrated by accident.
            if abs(corrected - raw_count) < 1e-9:
                corrected = None
        return PointResult(
            points=points, raw_count=raw_count, corrected_count=corrected,
            zone_counts=_point_zone_counts(points, zone_masks, (h, w)),
            confidence="validated" if self.validated else "advisory",
            inference_ms=(time.perf_counter() - started) * 1000.0,
            model_name=self.name, model_version=self.version,
        )

class DenseDensityEstimator(DensityEstimator):
    """Adapter for the dense DM-Count service."""

    name = "dense-dm-count"
    version = "dmcount-shab-vgg19"

    def __init__(self, validated: bool = False):
        from . import dense

        self._estimate_density = dense.estimate_density
        self.validated = validated
        self._lock = threading.Lock()

    def estimate(self, frame: np.ndarray, zone_masks: list | None = None) -> DensityResult:
        started = time.perf_counter()
        with self._lock:
            result = self._estimate_density(frame)
        if not isinstance(result, dict) or result.get("count") is None:
            raise RuntimeError("Dense DM-Count returned no count")
        # Service returns the map under "density"; accept the older key too.
        density_map = result.get("density", result.get("density_map"))
        density_map = (np.asarray(density_map, dtype=np.float32)
                       if density_map is not None else None)
        return DensityResult(
            total_count=float(result["count"]),
            zone_counts=_density_zone_counts(density_map, zone_masks),
            confidence="validated" if self.validated else "advisory",
            inference_ms=(time.perf_counter() - started) * 1000.0,
            model_name=self.name, model_version=self.version,
            density_map=density_map,
        )

class NullDensityEstimator(DensityEstimator):
    """Explicit "no density model installed" state.

    Returning a number here would be fabricating a crowd estimate, so the
    router instead reports the density path as unavailable/degraded.
    """

    name = "density-unavailable"
    version = "0"
    validated = False

    def estimate(self, frame: np.ndarray, zone_masks: list | None = None) -> DensityResult:
        raise RuntimeError("no density model configured")

    def info(self) -> dict:
        return {"name": self.name, "version": self.version, "validated": False,
                "available": False,
                "note": "Set CROWD_DENSITY_BACKEND=dense with models/dmcount_shab.pth"}

def _density_zone_counts(density_map: np.ndarray | None,
                         zone_masks: list | None) -> dict:
    """Integrate density-map cells inside normalised camera zones.

    A density map's sum represents people, so zone values are calculated from
    the map directly.  This avoids the common but incorrect image-area scaling
    that makes a count change when the input is resized.
    """
    if density_map is None or density_map.ndim != 2 or not zone_masks:
        return {}
    import cv2

    h, w = density_map.shape
    counts: dict = {}
    for zone in zone_masks:
        key = zone.get("id", zone.get("name", "zone"))
        polygon = np.asarray(zone.get("polygon", []), dtype=np.float32)
        if len(polygon) < 3:
            counts[key] = 0.0
            continue
        polygon *= np.array([w, h], dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        counts[key] = float(density_map[mask.astype(bool)].sum())
    return counts

class TorchscriptDensityEstimator(DensityEstimator):
    """DM-Count-style density adapter for an operator-supplied TorchScript model.

    The checkpoint is deliberately never auto-downloaded: provenance and terms
    must be cleared, and `validated` stays False until a camera-specific
    validation gate is signed off. Until then the router keeps the output
    advisory and `selected_count` null.
    """

    def __init__(self, path: str, input_size: int = 768, validated: bool = False):
        import torch

        self.path = path
        self.input_size = input_size
        self.validated = validated
        self.name = f"density:{path.split('/')[-1]}"
        self.version = "torchscript"
        self._torch = torch
        self._module = torch.jit.load(path, map_location="cpu").eval()
        self._lock = threading.Lock()

    def estimate(self, frame: np.ndarray, zone_masks: list | None = None) -> DensityResult:
        import cv2

        h, w = frame.shape[:2]
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))
        tensor = (self._torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0)[None]
        with self._lock, self._torch.no_grad():
            density = self._module(tensor)[0, 0].cpu().numpy()
        # Density-map units are people per output cell.  Resizing its input
        # changes the grid resolution but must not multiply the total people.
        total = float(density.sum())
        zone_counts = _density_zone_counts(density, zone_masks)
        return DensityResult(total, zone_counts,
                             "advisory" if not self.validated else "validated",
                             (time.perf_counter() - started) * 1000.0,
                             self.name, self.version, density)

    def info(self) -> dict:
        return {"name": self.name, "version": self.version,
                "validated": self.validated, "available": True,
                "input_size": self.input_size,
                "note": "Advisory until the per-camera promotion gate is signed off"}
