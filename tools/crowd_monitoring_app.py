#!/usr/bin/env python3
"""Headless YOLOv8 person counter with one Supervision polygon zone.

Example:
  python crowd_monitoring_app.py --source clip.mp4 --max-frames 300
  python crowd_monitoring_app.py --source rtsp://user:pass@host/stream --realtime

The default polygon covers the central 80% of the image. Configure a real
camera's polygon and capacity before using the alert as an operational signal.
"""
from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="video file, webcam index, or RTSP URL")
    parser.add_argument("--weights", default="yolov8x.pt")
    parser.add_argument("--capacity", type=int, default=100)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames; 0 runs until the source ends")
    parser.add_argument("--realtime", action="store_true",
                        help="pace local video to its source FPS")
    parser.add_argument("--log-every", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.capacity <= 0 or args.log_every <= 0:
        raise SystemExit("capacity and log-every must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = 0 if torch.cuda.is_available() else "cpu"
    logging.info("loading %s on %s", args.weights,
                 torch.cuda.get_device_name(0) if device == 0 else "CPU")
    model = YOLO(args.weights)
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source: {args.source}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / source_fps if source_fps and source_fps > 0 else 0.0
    zone: sv.PolygonZone | None = None
    started = time.perf_counter()
    frames = 0
    over_capacity_frames = 0
    last_count = 0

    try:
        while args.max_frames <= 0 or frames < args.max_frames:
            frame_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if zone is None:
                polygon = np.array([
                    [0.10 * width, 0.10 * height],
                    [0.90 * width, 0.10 * height],
                    [0.90 * width, 0.90 * height],
                    [0.10 * width, 0.90 * height],
                ], dtype=np.int32)
                zone = sv.PolygonZone(
                    polygon=polygon,
                    frame_resolution_wh=(width, height),
                    triggering_position=sv.Position.BOTTOM_CENTER,
                )

            result = model.predict(frame, classes=[0], conf=args.confidence,
                                   device=device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            in_zone = zone.trigger(detections)
            last_count = int(np.count_nonzero(in_zone))
            frames += 1
            over_capacity_frames += last_count >= args.capacity
            now = time.perf_counter()
            if frames % args.log_every == 0 or last_count >= args.capacity:
                fps = frames / max(now - started, 1e-9)
                level = logging.WARNING if last_count >= args.capacity else logging.INFO
                logging.log(level, "frames=%d people_in_zone=%d capacity=%d fps=%.2f%s",
                            frames, last_count, args.capacity, fps,
                            " CAPACITY ALERT" if last_count >= args.capacity else "")
            if args.realtime and frame_interval > 0:
                time.sleep(max(0.0, frame_interval - (time.perf_counter() - frame_started)))
    finally:
        cap.release()

    elapsed = max(time.perf_counter() - started, 1e-9)
    logging.info("finished frames=%d last_zone_count=%d elapsed_s=%.2f avg_fps=%.2f over_capacity_frames=%d",
                 frames, last_count, elapsed, frames / elapsed, over_capacity_frames)
    return 0 if frames else 2


if __name__ == "__main__":
    raise SystemExit(main())
