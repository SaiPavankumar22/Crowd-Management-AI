"""Fetch public crowd / pilgrimage footage for testing the crowd monitor.

Downloads from Archive.org (no account needed), trims each source to a short
clip and writes a manifest with the source URL and licence so we never lose
track of what may be redistributed and what may not.

    python tools/fetch_footage.py            # fetch everything missing
    python tools/fetch_footage.py --list     # show what is already there

Output: .data/samples/pilgrim/<slug>.mp4 + manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / ".data" / "samples" / "pilgrim"
RAW_DIR = BASE_DIR / ".tmp" / "footage_raw"

# identifier, slug, licence, why we want it
SOURCES = [
    ("SapnaShahaniArdhKumbhMela", "kumbh_ardh_2007_street",
     "CC BY 2.0", "Ardh Kumbh Mela pilgrims on the ghats and approach roads"),
    ("SapnaShahaniArdhKumbhMela2007_0", "kumbh_ardh_2007_ghat",
     "CC BY 2.0", "Ardh Kumbh Mela river-bank crowd, dense standing group"),
    ("SapnaShahaniArdhKumbhMela2007_1", "kumbh_ardh_2007_procession",
     "CC BY 2.0", "Ardh Kumbh Mela procession, moving dense crowd"),
    ("Swajana-ArdhKumbhMela2007549", "kumbh_ardh_2007_camp",
     "CC BY-NC 2.0 (non-commercial)", "Kumbh camp / queue area, slow dense movement"),
    ("NagaSadhusInKumbhMela2013MahaShivaratri", "kumbh_2013_maha_shivratri",
     "not stated (test use only)", "Maha Shivratri bathing crowd at Kumbh 2013"),
    ("india-railways-crowd-bharat-civic-sense", "railway_crowd_india",
     "not stated (test use only)", "Indian railway platform crowd, queue-like density"),
    ("youtube-gN7u50CTCCQ", "tawaf_kaaba_a",
     "not stated (test use only)", "Tawaf around the Kaaba, circular dense crowd"),
    ("youtube-mlWf0H-_gC4", "tawaf_kaaba_b",
     "not stated (test use only)", "Tawaf crowd, oblique angle"),
    ("youtube-51-LDbF7OFc", "tawaf_kaaba_c",
     "not stated (test use only)", "Tawaf crowd, wider shot"),
]

CLIP_SECONDS = 45
MAX_WIDTH = 1280

# High-resolution footage. YouTube copies are for local evaluation only: they are
# NOT redistributable, unlike the CC-BY Archive.org items above.
YT_LICENCE = "YouTube (all rights reserved) - local evaluation only"

# video id, slug, scene note, [start, end] seconds of the source to keep
YT_SOURCES = [
    ("7PL06-02P-o", "kumbh_2025_walk_4k",
     "Maha Kumbh Mela 2025 walking tour, dense one-way pilgrim flow", [900, 945]),
    ("CtmghCnDEhU", "kumbh_2025_river_walk",
     "Maha Kumbh 2025 walk to the river on Mauni Amavasya", [10, 55]),
    ("sIXvtuVBQ-4", "kumbh_2025_crowd_walk",
     "Maha Kumbh 2025 crowd walking, close-packed pedestrians", [5, 50]),
    ("5Z0OMaVL1T4", "tawaf_4k_walk",
     "Kaaba tawaf, 4K60 continuous circular mass of pilgrims", [120, 165]),
    ("DTCQU72cHa4", "station_rush_mumbai",
     "Mumbai local train platform at 8pm rush, queue-like crushing density", [30, 75]),
    ("9gJRP_vQjO4", "hajj_stock_nocopyright",
     "Hajj 4K stock clip, uploader claims no-copyright (unverified: treat as "
     "evaluation-only until the licence is confirmed)", [0, 14]),
]


def log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        # Windows consoles default to cp1252; archive filenames are not always ASCII.
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "godavari-crowd/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def pick_video_file(meta: dict):
    """Best mp4 under 250 MB, else the largest mp4 we can find."""
    best = None
    for f in meta.get("files", []):
        name = f.get("name", "")
        if not name.lower().endswith((".mp4", ".m4v")):
            continue
        try:
            size = int(f.get("size", 0))
        except (TypeError, ValueError):
            size = 0
        if size == 0 or size > 250 * 1024 * 1024:
            continue
        if best is None or size > best[1]:
            best = (name, size)
    return best


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "godavari-crowd/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r    {pct:5.1f}%  {done/1e6:6.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def probe(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration,size",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        data = json.loads(out.stdout or "{}")
        st = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
        fps = round(float(num) / float(den or 1), 2) if float(den or 1) else None
        return {
            "width": st.get("width"),
            "height": st.get("height"),
            "fps": fps,
            "duration": round(float(fmt.get("duration") or 0), 1),
            "bytes": int(fmt.get("size") or 0),
        }
    except Exception:
        return {}


def transcode(src: Path, dst: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log("  !! ffmpeg not on PATH, keeping the raw file")
        return False
    cmd = [
        ffmpeg, "-y", "-v", "error", "-ss", "0", "-t", str(CLIP_SECONDS), "-i", str(src),
        "-vf", f"scale='min({MAX_WIDTH},iw)':-2,fps=25",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        log(f"  !! ffmpeg failed: {res.stderr.strip()[:200]}")
        return False
    return True


def fetch_youtube(vid: str, slug: str, note: str, span: list[int]) -> dict | None:
    """Grab one section of a YouTube video at <=720p via yt-dlp."""
    log(f"\n== {slug}  (youtube:{vid})")
    out_tpl = str(RAW_DIR / f"yt_{vid}.%(ext)s")
    start, end = span
    cmd = [
        sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-playlist",
        "-f", "bv*[height<=720][ext=mp4]/bv*[height<=720]/b",
        "--download-sections", f"*{start}-{end}", "--force-keyframes-at-cuts",
        "-o", out_tpl, f"https://www.youtube.com/watch?v={vid}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        log(f"  !! yt-dlp failed: {res.stderr.strip()[-300:]}")
        return None
    raw = next((p for p in RAW_DIR.glob(f"yt_{vid}.*")), None)
    if raw is None:
        log("  !! yt-dlp reported success but wrote nothing")
        return None
    out = OUT_DIR / f"{slug}.mp4"
    if not transcode(raw, out):
        shutil.copy2(raw, out)
    raw.unlink(missing_ok=True)
    info = probe(out)
    log(f"  -> {out.name} {info}")
    return {
        "slug": slug,
        "file": str(out.relative_to(BASE_DIR)).replace("\\", "/"),
        "source_page": f"https://www.youtube.com/watch?v={vid}",
        "source_span_s": span,
        "licence": YT_LICENCE,
        "scene": note,
        "fetched": time.strftime("%Y-%m-%d"),
        **info,
    }


def fetch_one(identifier: str, slug: str, licence: str, note: str) -> dict | None:
    log(f"\n== {slug}  ({identifier})")
    try:
        meta = http_json(f"https://archive.org/metadata/{identifier}")
    except urllib.error.URLError as exc:
        log(f"  !! metadata failed: {exc}")
        return None
    picked = pick_video_file(meta)
    if not picked:
        log("  !! no usable mp4 in this item")
        return None
    fname, size = picked
    url = f"https://archive.org/download/{identifier}/{urllib.parse.quote(fname)}"
    log(f"  file {fname} ({size/1e6:.1f} MB)")
    raw = RAW_DIR / f"{slug}{Path(fname).suffix}"
    if not raw.exists():
        download(url, raw)
    out = OUT_DIR / f"{slug}.mp4"
    if not transcode(raw, out):
        shutil.copy2(raw, out)
    raw.unlink(missing_ok=True)
    info = probe(out)
    log(f"  -> {out.name} {info}")
    return {
        "slug": slug,
        "file": str(out.relative_to(BASE_DIR)).replace("\\", "/"),
        "source_page": f"https://archive.org/details/{identifier}",
        "source_file": url,
        "licence": licence,
        "scene": note,
        "clip_seconds": CLIP_SECONDS,
        "fetched": time.strftime("%Y-%m-%d"),
        **info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*", help="slugs to fetch")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    have = {m["slug"] for m in manifest}

    if args.list:
        for m in manifest:
            print(f"{m['slug']:28s} {m.get('width')}x{m.get('height')} "
                  f"{m.get('duration')}s  {m['licence']}")
        return 0

    def record(entry: dict | None, slug: str) -> None:
        nonlocal manifest
        if entry:
            manifest = [m for m in manifest if m["slug"] != slug] + [entry]
            manifest.sort(key=lambda m: m["slug"])
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for identifier, slug, licence, note in SOURCES:
        if args.only and slug not in args.only:
            continue
        if slug in have:
            log(f"== {slug}: already have it")
            continue
        record(fetch_one(identifier, slug, licence, note), slug)

    for vid, slug, note, span in YT_SOURCES:
        if args.only and slug not in args.only:
            continue
        if slug in have:
            log(f"== {slug}: already have it")
            continue
        record(fetch_youtube(vid, slug, note, span), slug)

    total = sum(m.get("bytes", 0) for m in manifest)
    log(f"\n{len(manifest)} clips, {total/1e6:.1f} MB total in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
