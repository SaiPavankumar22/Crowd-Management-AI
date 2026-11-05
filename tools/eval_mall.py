#!/usr/bin/env python3
"""Evaluate person-count detections on CUHK Mall head-count annotations.

Dataset: data/benchmarks/mall/dataset/mall_dataset
This is a research-only benchmark. Results are not representative of outdoor
ghat views and are not safety validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat
from ultralytics import YOLO


def evaluate(dataset: Path, weights: str, output: Path, limit: int = 0,
             batch_size: int = 16, imgsz: int = 640, conf: float = 0.30):
    frames_dir = dataset / "frames"
    gt_path = dataset / "mall_gt.mat"
    counts = np.asarray(loadmat(gt_path)["count"]).reshape(-1).astype(int)
    total = min(len(counts), limit) if limit else len(counts)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(weights)
    records = []
    for start in range(0, total, max(1, batch_size)):
        stop = min(total, start + max(1, batch_size))
        images = []
        frame_ids = []
        for index in range(start, stop):
            frame_id = index + 1
            frame = cv2.imread(str(frames_dir / f"seq_{frame_id:06d}.jpg"))
            if frame is None:
                raise FileNotFoundError(f"missing frame {frame_id}")
            images.append(frame)
            frame_ids.append(frame_id)
        started = time.perf_counter()
        result = model.predict(images, classes=[0], conf=conf, imgsz=imgsz,
                               device=device, verbose=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        for frame_id, gt, prediction in zip(frame_ids, counts[start:stop], result):
            pred_count = len(prediction.boxes) if prediction.boxes is not None else 0
            records.append({"frame": frame_id, "ground_truth": int(gt),
                            "prediction": int(pred_count),
                            "absolute_error": abs(int(pred_count) - int(gt)),
                            "batch_inference_ms": round(elapsed_ms / len(result), 3)})

    truth = np.asarray([r["ground_truth"] for r in records], dtype=np.float64)
    pred = np.asarray([r["prediction"] for r in records], dtype=np.float64)
    errors = pred - truth
    mae = float(np.mean(np.abs(errors))) if len(errors) else 0.0
    summary = {
        "dataset": "CUHK Mall crowd-counting dataset",
        "weights": weights,
        "device": torch.cuda.get_device_name(0) if device == 0 else "CPU",
        "torch": torch.__version__,
        "frames": len(records),
        "ground_truth_mean": round(float(np.mean(truth)), 3) if len(truth) else 0,
        "prediction_mean": round(float(np.mean(pred)), 3) if len(pred) else 0,
        "mae_people_per_frame": round(mae, 4),
        "rmse_people_per_frame": round(math.sqrt(float(np.mean(errors ** 2))), 4) if len(errors) else 0,
        "bias_prediction_minus_truth": round(float(np.mean(errors)), 4) if len(errors) else 0,
        "wape": round(float(np.sum(np.abs(errors)) / max(1, np.sum(truth))), 5),
        "mean_absolute_error_count": int(np.sum(np.abs(errors))),
        "inference_ms_per_frame_mean": round(float(np.mean([r["batch_inference_ms"] for r in records])), 3) if records else 0,
        "commercial_use": False,
        "limitations": ["research-only dataset", "indoor fixed webcam; not ghat-domain validation",
                        "frame count evaluates detector count only, not tracking, zones, or alert safety"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else
                                ["frame", "ground_truth", "prediction", "absolute_error", "batch_inference_ms"])
        writer.writeheader()
        writer.writerows(records)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/benchmarks/mall/dataset/mall_dataset"))
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/mall/results/yolo26n-summary.json"))
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all annotated frames")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()
    evaluate(args.dataset, args.weights, args.output, args.limit,
             args.batch_size, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
