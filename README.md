# Godavari Crowd Monitor

**Real-time crowd counting, zone congestion and capacity alerting for ghats, temples,
stations and mass gatherings — from any camera stream you already have.**

Point it at a CCTV RTSP URL or a recorded clip. It detects every person with YOLO, tracks
them across frames, counts them inside the zones you draw, turns those counts into
occupancy and density, and raises an alert before a zone stops being safe — on a dashboard
built to be watched in a control room.

![Live dashboard: two sources, per-zone levels, alert stream, trend](docs/dashboard.png)

[Quick start](#quick-start) · [Docker](#run-it-in-docker) · [Deployment](docs/DEPLOYMENT.md) ·
[Scaling architecture](docs/PRODUCTION_ARCHITECTURE.md) · [Improvement plan](docs/improvement.md) ·
[Test footage](docs/FOOTAGE.md)

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Run it in Docker](#run-it-in-docker)
- [How it works](#how-it-works)
- [The dashboard](#the-dashboard)
- [Zones, occupancy and alerts](#zones-occupancy-and-alerts)
- [Performance and multi-camera limits](#performance-and-multi-camera-limits)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [Data and storage](#data-and-storage)
- [Test footage](#test-footage)
- [Deployment](#deployment)
- [Development tools](#development-tools)
- [Project layout](#project-layout)
- [Known limits and roadmap](#known-limits-and-roadmap)
- [Licensing](#licensing)

---

## What it does

| | |
|---|---|
| **Ingest** | Uploaded video, bundled sample clips, `rtsp://` CCTV streams, or a local webcam index. Files loop; network streams reconnect on drop. |
| **Detect** | YOLO26-nano (NMS-free, trained for small/dense objects) per frame, with YOLO11n/YOLOv8n fallback. CUDA if present, CPU otherwise. |
| **Track** | ByteTrack identities, so a person is not counted twice and each zone can show how many *distinct* people are inside it. |
| **Zones** | Polygon zones drawn on the frame, or an auto-grid (3×2, 4×3, …), each with its own name and capacity. |
| **Measure** | Per-zone people count (EMA-smoothed), occupancy % of capacity, relative density, flow direction and speed. |
| **Show** | Live annotated MJPEG feed, heatmap of where the crowd compresses, KPIs with sparklines, occupancy trend, control-room wall view. |
| **Alert** | Warning on rising edge, capacity alert after sustained overload, audio chime, SQLite history, CSV export. |

Everything is a single Python service with no external database, no message broker and no
cloud dependency — it runs on a laptop, a small VM, or a GPU box.

---

## Quick start

Windows, from the project folder (everything — venv, uploads, clips, alert DB — stays on
`D:`, nothing is written to the system drive):

```bash
bash tools/install.sh       # creates ./.venv and installs pinned deps
bash tools/start_dev.sh     # serves http://127.0.0.1:8000
```

`tools\start_dev.bat` does the same from Explorer. Then open **http://127.0.0.1:8000** and:

1. **+ Add** → choose *Sample* and pick a clip (the pilgrim library is under
   *Pilgrim*), or *Upload* your own `.mp4`, or *Source* and paste
   `rtsp://user:pass@camera-ip:554/stream`.
2. Click **Auto-grid** for instant zones, or tick **Draw zone** and click polygon points
   on the frame; set a name and capacity, then **Finish**.
3. Watch the zone rail and the feed. A zone turns yellow → orange → red as it fills;
   sustained overload appears in the alert stream and fires a chime.

Linux/macOS: the same scripts work (`tools/install.sh` creates `.venv/bin/python`).

---

## Run it in Docker

```bash
docker build -t godavari:cpu .                                  # CPU image, 0.67 GB
docker run -d --name godavari -p 127.0.0.1:8000:8000 \
  -v godavari-data:/app/.data --restart unless-stopped godavari:cpu
```

GPU host (NVIDIA driver + container toolkit; verified end to end on an RTX 3050 — the
container reports `"device":"gpu"`, `"cuda": true`):

```bash
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
             -t godavari:gpu .                        # 4.4 GB
docker run -d --name godavari --gpus all -p 127.0.0.1:8000:8000 \
  -v godavari-data:/app/.data --restart unless-stopped godavari:gpu
```

Or with compose: `docker compose up -d --build`, and for GPU
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`.

`cameras.json`, uploads, sample clips and the alert database live in the `godavari-data`
volume, so containers are disposable. Full instructions — Nebius GPU VM walkthrough,
TLS + password proxy, backups, upgrades, troubleshooting — are in
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

> The dashboard has **no authentication**. Publish it on loopback and put a proxy with a
> password in front before anyone else can reach it.

---

## How it works

```
                       ┌──────────────────────── FastAPI process ───────────────────────┐
 uploaded .mp4  ─┐     │  CameraWorker × N  (one thread per camera)                     │
 sample .mp4    ─┼───► │    capture ──► resize ──► YOLO detect ──► ByteTrack IDs        │
 rtsp://…       ─┤     │       │                     (one shared model, locked)         │
 webcam index   ─┘     │       ├──► people-in-frame count                               │
                       │       ├──► foot points ──► zone polygons ──► per-zone counts   │
                       │       │                        └──► occupancy %, density, flow │
                       │       ├──► heatmap grid (48×27, decaying)                      │
                       │       └──► annotated JPEG @ 12 fps ──► /api/cameras/{id}/stream│
                       │                                                                 │
                       │  AlertEngine: sustain + cooldown ──► SQLite history (crowd.db)  │
                       │  Broadcaster: full snapshot @ 0.6 s ──► WebSocket /ws           │
                       └─────────────────────────────────────────────────────────────────┘
                                              │
                                    browser dashboard (static/)
```

The server sends **one annotated JPEG stream** and **thin JSON snapshots**; zones, chips,
heatmap, flow arrows, gauges and trends are all drawn client-side on a canvas over the
video. That keeps per-camera CPU cost on the server down to detection plus one JPEG encode,
and lets overlays stay full-resolution regardless of stream quality.

| Spec requirement | Where it lives |
|---|---|
| 1. Video input (CCTV / recorded) | `app/pipeline.py` (file, RTSP, webcam + reconnect), `app/main.py` upload endpoint |
| 2. YOLO person detection | `app/detector.py` — YOLO26n, fallbacks, GPU auto-detect |
| 3. People count per frame | `app/pipeline.py` `_process()` → `metrics.count` |
| 4. Zones | `app/zones.py`, polygons from the dashboard or auto-grid |
| 5. Density and occupancy | `app/zones.py` — occupancy % of capacity, persons/10k px² |
| 6. Live dashboard | `static/` + WebSocket snapshots + MJPEG stream |
| 7. Capacity alerts | `app/alerting.py` — sustain window, cooldown, SQLite history |

---

## The dashboard

| Panel | What it shows |
|---|---|
| **Sources** | Every camera with live count and worst zone level; add/remove; per-source status (`LIVE`, `RECONNECTING`, stopped) |
| **Stage** | Annotated feed, HUD with smoothed people-in-frame, FPS and source, per-zone chips (`count/capacity · occupancy % · level`) |
| **Heatmap** | 48×27 decaying density grid built from foot points — shows *where* the crowd compresses, not just how many |
| **Flow** | Per-zone average track velocity as a direction arrow with speed; eased over ~1 s so it does not blink |
| **Zone rail** | Ring gauge, level pill, editable capacity, flow readout and sparkline per zone |
| **KPIs** | People across all sources, tracked identities, inference rate, zones at risk — each with a sparkline |
| **Trend** | Occupancy history with alert markers, so you can see the run-up to a breach |
| **Alert stream** | Live alerts plus persisted history, CSV export, audio chime on capacity breach |
| **Wall view** | All cameras as tiles with count and level, for a control-room display |

Keyboard: `H` heatmap · `Z` zones · `F` flow · `Esc` cancel zone drawing.

---

## Zones, occupancy and alerts

Per zone: `occupancy % = smoothed people in zone / capacity`, where the smoothed count is an
EMA over frames (60/40) to remove detection jitter.

| Level | Rule | What happens |
|---|---|---|
| **OK** | < 2/3 of the warning line | — |
| **BUSY** | ≥ 2/3 of the warning line | colour change only |
| **WARNING** | ≥ `CROWD_WARN_PCT` (default 75 %) | one alert on the rising edge — hysteresis stops repeat spam |
| **OVERCROWDED** | ≥ 100 % for `CROWD_ALERT_SUSTAIN` (default 5 s) | capacity alert + chime, repeating at most every `CROWD_ALERT_COOLDOWN` (60 s) |

A person counts as *inside* a zone when their bounding-box foot point falls inside the
polygon, which is what keeps an overlapping queue from double-counting. Alerts are written to
SQLite and survive restarts; `GET /api/alerts/export` gives you the CSV.

---

## Performance and multi-camera limits

All cameras share **one** model instance behind a lock, so memory stays flat as you add
sources. Each camera has its own capture thread and its own inference budget
(`CROWD_TARGET_INFER_FPS`, default 5).

Measured, not estimated:

| Setup | Throughput |
|---|---|
| This dev machine, CPU torch, `imgsz=1280` | 3.6 inferences/s (one camera); 1.8/s each with two cameras |
| `godavari:cpu` container, `imgsz=640` | 3.2 FPS for one camera |
| `godavari:gpu` on an RTX 3050, `imgsz=640` | **32 inferences/s** — four cameras at 8.0 FPS each, 1.8 GB VRAM |

| Hardware | Practical cameras with this app | Notes |
|---|---|---|
| 8 vCPU, no GPU | 2–4 | Lower `CROWD_IMGSZ` / `CROWD_MAX_FRAME_WIDTH` to gain headroom |
| One L40S (or similar) | ~10–20 at 5 FPS | Estimate (6–10× a 3050); model time stops being the bottleneck, JPEG encoding and Python threads take over |
| 100+ cameras | not this process — see [the scaling doc](docs/PRODUCTION_ARCHITECTURE.md) | MediaMTX fan-in, NVDEC decode, Triton batching, TimescaleDB, server-side mosaic |

A note that matters more than FPS: box detectors **under-count dense crowds**. On our own
test clips, YOLO26n finds 4 people in an aerial shot of tens of thousands at the Haram, and
1–3 in an overhead railway-platform crush. That is why the roadmap below puts a head-point
density model next to the detector rather than trusting counts alone.

---

## Configuration

Every setting is an environment variable; nothing needs editing in code.

| Variable | Default | Purpose |
|---|---|---|
| `CROWD_MODEL` | `yolo26n.pt` | `yolo26s.pt` for better far-field recall (more compute), `yolo11n.pt` / `yolov8n.pt` alternatives |
| `CROWD_CONF` | `0.30` | Detection confidence threshold |
| `CROWD_IMGSZ` | `640` | Inference resolution — 960–1280 for wide/aerial views with tiny people |
| `CROWD_TARGET_INFER_FPS` | `5` | Inference rate per camera |
| `CROWD_MAX_FRAME_WIDTH` | `1280` | Frames are downscaled to this width before detection |
| `CROWD_JPEG_QUALITY` | `65` | Quality of the annotated stream |
| `CROWD_STREAM_FPS` | `12` | MJPEG stream frame rate |
| `CROWD_WS_PERIOD` | `0.6` | Seconds between dashboard snapshots |
| `CROWD_FRAME_STALE_S` | `8` | Seconds before a camera is considered stale |
| `CROWD_FROZEN_FRAME_S` | `10` | Seconds of unchanged pixels before frozen status |
| `CROWD_DECODER` | `auto` | Decoder backend: auto, opencv, ffmpeg, gstreamer or nvdec |
| `CROWD_BATCH_SIZE` | `1` | Maximum inference micro-batch size |
| `CROWD_MAX_BATCH_WAIT_MS` | `8` | Maximum scheduler wait before dispatching a batch |
| `CROWD_FRAME_QUEUE_SIZE` | `2` | Bounded inference queue size |
| `CROWD_LOOP` | `1` | Loop file sources |
| `CROWD_WARN_PCT` | `75` | Zone warning level (% of capacity) |
| `CROWD_ZONE_CAPACITY` | `100` | Capacity given to auto-created zones |
| `CROWD_ALERT_SUSTAIN` | `5` | Seconds at ≥100 % before a capacity alert |
| `CROWD_ALERT_COOLDOWN` | `60` | Minimum seconds between repeat alerts for one zone |
| `ALERT_WEBHOOK_URL` | empty | Optional JSON webhook for alert delivery |
| `ALERT_WEBHOOK_SECRET` | empty | Optional HMAC-SHA256 signing secret |
| `ALERT_WEBHOOK_RETRIES` | `3` | Webhook delivery attempts |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address (`0.0.0.0` inside Docker) |
| `CROWD_DATA_DIR` | `./data` | `cameras.json`, `crowd.db` (dev scripts and Docker point this at `.data/data`) |
| `CROWD_UPLOADS_DIR` | `./uploads` | Uploaded clips |
| `CROWD_SAMPLES_DIR` | `./samples` | Sample library (subfolders become picker groups) |

---

## HTTP API

Everything the dashboard does is available over HTTP, so an operations script or an existing
NOC dashboard can drive it.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/cameras` | All cameras with live metrics |
| `POST` | `/api/cameras` | Add one: multipart `file=`, or form `sample=` / `source=`, plus `name=`, `loop=` |
| `GET` | `/api/cameras/{id}` | One camera |
| `GET` | `/api/cameras/{id}/diagnostics` | Camera health, decoder and timing diagnostics |
| `DELETE` | `/api/cameras/{id}` | Stop and remove |
| `PUT` | `/api/cameras/{id}/zones` | Replace zones: `{"zones":[{name, polygon, capacity, warn_pct}]}` |
| `GET` | `/api/cameras/{id}/stream` | Annotated MJPEG stream |
| `GET` | `/api/cameras/{id}/snapshot` | Single annotated JPEG |
| `GET` | `/api/samples` | Available sample clips (subfolder clips as `dir/name`) |
| `GET` | `/api/alerts?limit=100` | Persisted alert history |
| `GET` | `/api/alerts/export` | Alert history as CSV |
| `GET` | `/api/system` | Model, device, torch/CUDA, uptime, counters |
| `GET` | `/api/health` | Service and camera health summary |
| `GET` | `/metrics` | Prometheus-compatible process metrics |
| `GET` | `/api/models` | Available and active models |
| `PUT` | `/api/models` | Switch between configured models |
| `PUT` | `/api/cameras/{id}/calibration` | Save ground-plane calibration points |
| `WS` | `/ws` | Full snapshot (cameras + alerts + system) at `CROWD_WS_PERIOD` |

```bash
# add a sample camera and give it two zones
curl -X POST localhost:8000/api/cameras -F sample=pilgrim/tawaf_4k_walk.mp4 -F name=Tawaf
curl -X PUT localhost:8000/api/cameras/<id>/zones -H 'content-type: application/json' \
  -d '{"zones":[{"name":"Gate A","capacity":3,"polygon":[[0.02,0.52],[0.48,0.52],[0.48,0.99],[0.02,0.99]]}]}'
```

---

## Data and storage

```
.data/
├── data/
│   ├── cameras.json      # cameras + zone polygons (back this up)
│   └── crowd.db          # alert history, SQLite (back this up)
├── uploads/              # uploaded clips
├── samples/              # sample clips; subfolders become picker groups
│   ├── crowd_street.mp4
│   └── pilgrim/          # 15 real crowd clips + manifest.json
├── ultralytics/          # Ultralytics settings cache
├── torch/                # model download cache
└── matplotlib/
```

The dev scripts (`tools/start_dev.sh`) relocate `data/`, `uploads/` and `samples/` into
`.data/` and put the Ultralytics/torch/matplotlib caches there too, so nothing lands outside
the project. `.venv/`, `.data/` and `.tmp/` are gitignored; in Docker the whole `.data` tree
is one volume. Media paths are stored absolute, but if a file moves the app re-resolves it by
filename inside the samples/uploads folders, so relocating the data folder does not kill
saved cameras.

---

## Test footage

Fifteen real mass-gathering clips (Kumbh 2025, Ardh Kumbh 2007, Maha Shivratri 2013, Tawaf
around the Kaaba, Mumbai platform rush, aerial Haram) ship as a selectable library, with
provenance, licences, and what each one is good for in
**[docs/FOOTAGE.md](docs/FOOTAGE.md)**. Fetch or extend it with:

```bash
./.venv/Scripts/python.exe tools/fetch_footage.py --list
./.venv/Scripts/python.exe tools/fetch_footage.py                # fetch what's missing
```

Four of the clips carry Creative Commons licences (three CC BY 2.0, one non-commercial); the
YouTube-sourced ones are **evaluation only** — do not redistribute them or ship them to a
customer.

---

## Deployment

Short version:

```bash
docker compose up -d --build                       # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # GPU
```

Long version, including the Nebius GPU VM walkthrough, HTTPS + password proxy, backups,
upgrade/rollback and a troubleshooting table:
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

Design context for anything beyond a handful of cameras — why MediaMTX, NVDEC, Triton and
TimescaleDB, and how to size a single L40S for ~100 cameras — is in
**[docs/PRODUCTION_ARCHITECTURE.md](docs/PRODUCTION_ARCHITECTURE.md)**.

---

## Development tools

| Tool | What it does |
|---|---|
| `tools/install.sh` / `tools/start_dev.sh` / `tools/start_dev.bat` | Project-local venv, pinned deps, server start |
| `tools/ui_check.py` | Renders the dashboard in headless Chromium over CDP, prints DOM facts, fails on console errors, saves a screenshot — no GUI needed |
| `tools/fetch_footage.py` | Downloads/trims the test clip library and writes `manifest.json` |
| `tools/download_sample.py`, `tools/extract_pexels.py` | Older single-clip helpers |

```bash
./.venv/Scripts/python.exe tools/ui_check.py http://127.0.0.1:8000/ shot.png
```

---

## Project layout

```
app/
  main.py        FastAPI app: REST, MJPEG, WebSocket, upload
  pipeline.py    CameraWorker (capture → detect → track → zones) + CameraRegistry
  detector.py    ModelManager: YOLO26n with fallbacks, device selection
  zones.py       Zone model, occupancy/density, level thresholds
  alerting.py    Sustained-threshold alert engine with hysteresis and cooldown
  heatmap.py     Decaying density grid from foot points
  store.py       SQLite alert history
  ingest.py      Upload/sample/RTSP source handling
  config.py      Env-var configuration
static/          Dashboard (index.html, app.js, style.css)
tools/           Install, start, headless UI check, footage fetcher
docs/            Architecture, deployment, footage, screenshots
Dockerfile, docker-compose.yml, docker-compose.gpu.yml
```

---

## Known limits and roadmap

Honest state of the project — it is a working prototype with production-shaped pieces, not a
certified life-safety system.

- **Density is relative, not metric.** `persons/10k px²` depends on camera geometry. A
  per-camera homography to metres would give real people/m², which is what the literature's
  safety bands (≈4–5 ppl/m² safe, > 6 ppl/m² flow collapse) are expressed in.
- **Detector recall collapses in packed crowds** (see the measurements in
  [docs/FOOTAGE.md](docs/FOOTAGE.md)). A head-point density model (P2PNet / DM-Count) used as
  a cross-check, with routing "trust detection when sparse, density when dense", is the
  intended next model addition.
- **No frozen-frame detection yet.** A stalled camera reads as "no congestion", which is the
  most dangerous failure mode for an alarm system; it is called out in the architecture doc.
- **No authentication, no multi-user, no alert delivery** beyond the dashboard and CSV. Alert
  webhooks/SMS and SSO are deployment work, not done here.
- **One process, one machine.** The sharded design (many decode workers, one Triton server,
  TimescaleDB, a control plane) is specified but not implemented.
- Zone counts are per-camera with no cross-camera de-duplication; overlapping views of the
  same gate will double-count.

---

## Licensing

- Application code in this repository: as you choose for your project.
- **Ultralytics YOLO is AGPL-3.0.** Running it as a network service for third parties can
  oblige you to publish the corresponding source. For internal use this is normally fine;
  commercially, buy an Ultralytics licence or swap the detector.
- Test footage: Creative Commons for four Archive.org clips (three CC BY 2.0, one CC BY-NC),
  evaluation-only for the YouTube clips. Per-clip licences in
  [docs/FOOTAGE.md](docs/FOOTAGE.md).
