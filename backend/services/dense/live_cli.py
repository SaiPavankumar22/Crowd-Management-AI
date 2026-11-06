"""Watch a camera from the command line, with no browser: count every few seconds and send alerts.

    python -m crowdshield.live_cli --source rtsp://192.168.1.20/stream --zones zones.json --interval 20 --notify

`zones.json` names each zone as a box (fractions of the picture: left, top, right, bottom) and how many
people it holds comfortably:

    {"Gate": {"box": [0.0, 0.4, 0.5, 1.0], "capacity": 60},
     "Stairs": {"box": [0.5, 0.2, 1.0, 1.0], "capacity": 40}}

With --notify, a new High Risk or Critical zone is sent to Telegram and email (whichever is set up
through TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID and GMAIL_USER / GMAIL_APP_PASSWORD). A camera on your own
network needs CROWDSHIELD_ALLOW_PRIVATE_URLS=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from . import notify
from .live import LiveMonitor, capacity_thresholds, check_url, grab_frame


def load_zones(path: Path, frame_shape) -> dict:
    h, w = frame_shape[:2]
    zones = {}
    for name, spec in json.loads(Path(path).read_text(encoding="utf-8")).items():
        x0, y0, x1, y1 = spec["box"]
        zones[name] = {"polygon": [(x0 * w, y0 * h), (x1 * w, y0 * h), (x1 * w, y1 * h), (x0 * w, y1 * h)],
                       "thresholds": capacity_thresholds(spec["capacity"]),
                       "area_m2": spec.get("area_m2")}
    return zones


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="rtsp:// or http(s):// stream, or a video file")
    ap.add_argument("--zones", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=20.0, help="seconds between counts")
    ap.add_argument("--notify", action="store_true", help="send alerts to the channels that are set up")
    ap.add_argument("--once", action="store_true", help="count one frame and stop")
    ap.add_argument("--log", type=Path, help="append every reading to this CSV file (time, zone, people, risk)")
    args = ap.parse_args(argv)

    if "://" in args.source:
        ok, reason = check_url(args.source)
        if not ok:
            print(reason, file=sys.stderr)
            return 2
    from .points import point_grid

    frame = None
    while frame is None:
        frame = grab_frame(args.source)
        if frame is None:
            print("Could not read the stream; retrying in 10 s.", file=sys.stderr)
            time.sleep(10)
    monitor = LiveMonitor(load_zones(args.zones, frame.shape), point_grid)
    log = _open_log(args.log)
    start = time.time()
    try:
        while True:
            began = time.time()
            try:
                result = monitor.update(began - start, frame)
            except Exception as exc:  # one bad frame or a model hiccup must not stop an all-day watch
                print(time.strftime("%H:%M:%S"), f"Could not count this frame ({type(exc).__name__}); "
                      "trying the next one.", file=sys.stderr, flush=True)
                if args.once:
                    return 1
                result = None
            if result:
                _report(result, args, log)
            if args.once:
                return 0
            # Keep the rhythm: the interval is from one count's start to the next, so a slow count
            # does not push every later count further out.
            time.sleep(max(1.0, args.interval - (time.time() - began)))
            fresh = grab_frame(args.source)
            if fresh is not None:   # keep the last picture if the stream hiccups
                frame = fresh
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        return 0
    finally:
        if log:
            log[0].close()


def _open_log(path):
    """(file, csv writer) for --log, with a header on a new file; None when not asked for."""
    if not path:
        return None
    fresh = not path.exists() or path.stat().st_size == 0
    handle = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if fresh:
        writer.writerow(["time", "zone", "people", "risk", "minutes_to_full"])
    return handle, writer


def _report(result, args, log) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(stamp, " | ".join(f"{z}: {r['count']:.0f} ({r['risk']})" for z, r in result["zones"].items()), flush=True)
    if log:
        for zone, r in result["zones"].items():
            log[1].writerow([time.strftime("%Y-%m-%d %H:%M:%S"), zone, f"{r['count']:.0f}", r["risk"],
                             "" if r["eta_s"] is None else f"{r['eta_s'] / 60:.1f}"])
        log[0].flush()
    for event in result["events"]:
        text = notify.alert_text(event["zone"], event["zone"], event["risk"], event["count"],
                                 event["per_m2"], None if event["eta_s"] is None else event["eta_s"] / 60)
        print("  ALERT:", text.splitlines()[0], flush=True)
        if args.notify:
            for line in notify.broadcast(text):
                print("  ", line, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
