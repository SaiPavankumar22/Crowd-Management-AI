#!/usr/bin/env python3
"""Offline model/tiling benchmark on real clips.

    python backend/tools/bench_models.py --clips samples/a.mp4 \
        --models yolo26n.pt --tiling off,2x2,3x3 --frames 60
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters import TiledDetector, UltralyticsDetector  # noqa: E402
from app.tiling import TilingPolicy  # noqa: E402


def build_model(name: str, imgsz: int, conf: float, tiling: str):
    detector = UltralyticsDetector(name, conf=conf, imgsz=imgsz,
                                   device=0 if _cuda() else "cpu").load()
    if tiling == "off":
        return detector, None
    rows, cols = (int(v) for v in tiling.split("x"))
    policy = TilingPolicy(enabled=True, rows=rows, cols=cols, overlap=0.15,
                          # force the gate so the tiling path is measured even
                          # on scenes that would not normally trigger it
                          min_box_height_px=10_000.0, min_boxes=1)
    return TiledDetector(detector, policy), policy


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def vram_mb() -> float:
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 0)
    except Exception:
        return 0.0


def read_frames(path: str, count: int) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def run(model, frames: list) -> dict:
    model.detect(frames[0])  # warm-up, excluded from timings
    counts, times = [], []
    for frame in frames:
        started = time.perf_counter()
        result = model.detect(frame)
        times.append((time.perf_counter() - started) * 1000.0)
        counts.append(result.count)
    return {
        "frames": len(frames),
        "mean_count": round(statistics.fmean(counts), 1) if counts else 0,
        "median_count": sorted(counts)[len(counts) // 2] if counts else 0,
        "mean_ms": round(statistics.fmean(times), 1) if times else 0,
        "p95_ms": round(sorted(times)[int(0.95 * (len(times) - 1))], 1) if times else 0,
        "fps": round(1000.0 / statistics.fmean(times), 1) if times else 0,
        "vram_mb": vram_mb(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["yolo26n.pt"])
    ap.add_argument("--tiling", default="off,3x3",
                    help="comma list of off|2x2|3x3")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--manual", default="", help="clip=count,clip=count")
    ap.add_argument("--json", dest="json_path", default="")
    args = ap.parse_args()

    manual = {}
    for item in filter(None, args.manual.split(",")):
        key, value = item.split("=")
        manual[os.path.basename(key)] = int(value)

    rows = []
    for clip in args.clips:
        frames = read_frames(clip, args.frames)
        if not frames:
            print(f"skip (unreadable): {clip}")
            continue
        for name in args.models:
            for tiling in args.tiling.split(","):
                model, _ = build_model(name, args.imgsz, args.conf, tiling)
                stats = run(model, frames)
                truth = manual.get(os.path.basename(clip))
                row = {
                    "clip": os.path.basename(clip),
                    "model": name,
                    "tiling": tiling,
                    **stats,
                    "manual_count": truth,
                    "abs_error": (abs(stats["mean_count"] - truth)
                                  if truth is not None else None),
                }
                rows.append(row)
                print(f"{row['clip'][:34]:34} {name:12} tiling={tiling:5} "
                      f"count={row['mean_count']:7} {row['mean_ms']:7}ms "
                      f"{row['fps']:6}fps vram={row['vram_mb']:.0f}MB"
                      + (f" err={row['abs_error']}" if row["abs_error"] is not None else ""),
                      flush=True)
                del model
                try:
                    import torch
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
