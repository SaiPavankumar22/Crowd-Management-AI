# Godavari Crowd Monitor

## Problem statement

**Real-Time Crowd Monitoring and Congestion Management** — develop a technology-enabled
solution to monitor crowd density and occupancy at key bathing ghats, access routes and
public areas, identify congestion or overcrowding risks, and provide timely alerts and
actionable information to the designated control teams.

Godavari does that from CCTV / RTSP streams or recorded clips: person detection, zone
occupancy, dense-crowd estimates, and control-room alerts on one dashboard.

![Live dashboard: collapsible sidebar, five camera views, areas and gate lines, KPIs](docs/dashboard.png)

[Quick start](#quick-start) · [Docker](#run-it-in-docker) · [Deployment](docs/DEPLOYMENT.md) ·
[Operator guide](docs/GUIDE.md) · [Scaling architecture](docs/PRODUCTION_ARCHITECTURE.md) ·
[Detectors](docs/DETECTORS.md) · [Test footage](docs/FOOTAGE.md) ·
[Results report (PDF)](docs/RESULTS_REPORT.pdf)

---

## Contents

- [Problem statement](#problem-statement)
- [What it does](#what-it-does)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [Fresh clone: what runs and what stays off](#fresh-clone-what-runs-and-what-stays-off)
- [Run it in Docker](#run-it-in-docker)
- [How it works](#how-it-works)
- [Dense-crowd estimators](#dense-crowd-estimators-pet--dm-count)
- [Choosing a detector model](#choosing-a-detector-model)
- [The dashboard](#the-dashboard)
- [Zones, occupancy and alerts](#zones-occupancy-and-alerts)
- [Performance and multi-camera limits](#performance-and-multi-camera-limits)
- [Engine settings that pay off](#engine-settings-that-pay-off)
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

## Tech stack

| Layer | Choice |
|---|---|
| **Language / runtime** | Python 3.11 |
| **API & server** | FastAPI, Uvicorn |
| **Person detection** | Ultralytics YOLO26 (default), RT-DETR optional; OpenCV for decode/render |
| **Tracking** | Supervision ByteTrack |
| **Dense counting** | PET (head points) and DM-Count (density maps) under `backend/services/dense` |
| **Deep learning** | PyTorch — CUDA on GPU hosts, CPU otherwise |
| **Dashboard** | Static HTML / CSS / JS (MJPEG streams + WebSocket snapshots) |
| **Local state** | SQLite alerts DB + `cameras.json` |
| **Optional archive** | MongoDB Atlas mirror |
| **Packaging** | Docker Compose (CPU or NVIDIA GPU images) |

Environment variables, detector switches and deployment knobs live in
[`.env.example`](.env.example) and [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Quick start

Model weights (detector, PET and DM-Count) are stored with Git LFS, so a plain
clone leaves ~130-byte pointer files where the checkpoints should be. Pull them
once after cloning, or the app stops at startup with a `git lfs pull` hint:

```bash
git lfs install && git lfs pull
```

Windows, from the project folder (everything — venv, uploads, clips, alert DB — stays on
`D:`, nothing is written to the system drive):

```bash
bash backend/tools/install.sh       # creates ./.venv and installs pinned deps
bash backend/tools/start_dev.sh     # serves http://127.0.0.1:8000
```

`backend\tools\start_dev.bat` does the same from Explorer. Then open **http://127.0.0.1:8000** and:

1. **+ Add** → choose *Sample* and pick a clip (the pilgrim library is under
   *Pilgrim*), or *Upload* your own `.mp4`, or *Source* and paste
   `rtsp://user:pass@camera-ip:554/stream`.
2. Click **Auto-grid** for instant zones, or tick **Draw zone** and click polygon points
   on the frame; set a name and capacity, then **Finish**.
3. Watch the zone rail and the feed. A zone turns yellow → orange → red as it fills;
   sustained overload appears in the alert stream and fires a chime.

Sample clips are not in the repository (footage licensing); `python
backend/tools/fetch_footage.py` fetches a few, or upload your own `.mp4`.

Linux/macOS: the same scripts work (`backend/tools/install.sh` creates `.venv/bin/python`).

### Fresh clone: what runs and what stays off

A clone that completed `install.sh` (or a Docker image built after `git lfs pull`)
runs detector-only immediately — no database, no broker, no cloud account:

| Included by default | Needs one decision from you |
|---|---|
| Detector, ByteTrack tracking, zones, gate lines, heatmap, alerts, dashboard, API | **PET + DM-Count:** installed but off — set `CROWD_POINT_BACKEND=dense` / `CROWD_DENSITY_BACKEND=dense` |
| PyTorch: `install.sh` tries CUDA wheels and falls back to CPU; Docker builds CPU by default (`TORCH_INDEX_URL` for GPU) | **Model:** `yolo26n.pt` unless `CROWD_MODEL` says otherwise |
| Weight preflight at startup and in the Docker build | **Areas, gate lines, calibration:** empty per camera until you draw them |
| SQLite alert history in `data/crowd.db`, cameras in `data/cameras.json` | **Auth:** off; set `CROWD_AUTH_OPERATOR_TOKEN` / `CROWD_AUTH_VIEWER_TOKEN` to require tokens |
| Loopback bind `127.0.0.1:8000` | **Sample clips:** not committed; fetch or upload your own |

`backend/tools/preflight.py` is the guard behind that: it refuses to start when
`models/` still holds Git LFS pointers or an enabled estimator's checkpoint is
missing, and it prints the command that fixes it. A missing detector only warns,
because Ultralytics can download it.

Optional host packages: `ffmpeg` for codecs OpenCV cannot decode, and `git-lfs`
for the weights. Everything else the app needs is pinned in `requirements.txt`.

The full day-to-day manual — sources, engine tuning, drawing areas correctly,
KPIs, alerts, validation, troubleshooting — is
**[docs/GUIDE.md](docs/GUIDE.md)**.

---

## Run it in Docker

Run `git lfs pull` before building. The image copies `models/` verbatim and the
build stops with a clear message if those files are still LFS pointers.

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
volume, so containers are disposable. Full instructions — GPU VM walkthrough,
TLS + password proxy, backups, upgrades, troubleshooting — are in
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

When `MONGODB_URL` and `MONGODB_DB_NAME` are set in the root `.env`, Docker Compose and
`python run.py` load them automatically. SQLite remains the local source of truth; a
background worker mirrors cameras (without source paths/credentials), alerts, audit
events, count samples and heatmaps to MongoDB collections. Uploaded video and evidence
remain on the local volume. MongoDB outages do not stop inference or local writes; check
`mongodb` in `/api/system` for archive connection and queue status.
An existing SQLite history can be migrated to Atlas at any time. Migrate history with
`python backend/tools/migrate_sqlite_to_mongo.py --apply` (run
inside the container with `--sqlite /app/.data/data/crowd.db --cameras
/app/.data/data/cameras.json`). The migration uses idempotent upserts and can be rerun
safely.

Removing a source stops its worker and removes its saved camera entry. An uploaded video
is deleted from `uploads/` when no other source uses it; bundled sample clips are kept.

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

## Dense-crowd estimators (PET + DM-Count)

A body detector counts bodies. In a packed crowd the people behind them are
heads only, so the box count understates the crowd no matter how good the
detector is. Two independent counting models therefore run beside it, from the
Dense checkpoints in `models/` (not committed — large binaries):

| Model | Checkpoint | What it adds |
|---|---|---|
| PET | `models/pet_shab.pth` | One point per visible head: count **and** locations, per area |
| DM-Count | `models/dmcount_shab.pth` | A density map; the map sums to a count, integrated per area |

Place the files under `models/` (or set `CROWD_MODELS_DIR`), then enable:

```bash
CROWD_POINT_BACKEND=dense
CROWD_DENSITY_BACKEND=dense
```

Network definitions live in `backend/services/dense` (academic-use licence). Each load
verifies SHA-256. Without checkpoints the service runs detector-only and estimators
report `available: false`.
Both are **advisory**: the API reports `detector_count`, `head_point_count` and
`density_count` side by side and `selected_count` stays `null` until a
per-camera validation is signed off in the dashboard. They are never summed.

When PET and DM-Count agree within 25% of each other, the dashboard stops
showing the detector floor and shows their mean as `≈N`
(`MODEL CONSENSUS · ESTIMATE`); the advisory alert then names both estimates
and the spread. When they disagree, the badge falls back to `≥N` from the
detector instead of inventing a middle. Neither path sets `selected_count`, so
a consensus is information for the operator, not a promotion.

Measured on the sample footage (RTX PRO 6000, five cameras):

| Clip | Detector (detail pass) | PET | DM-Count | Headline | Spread |
|---|---:|---:|---:|---:|---:|
| 12269404 1080p crowd | ≥156 | 389 | 383 | **≈386** | 1.6% |
| Getty 683716020 | ≥74 | 472 | 419 | **≈446** | 11.9% |
| Getty 685608730 | 19 | 16 | 15 | ≈15 | 9.2% |
| Getty 2289512996 | ≥10 | 27 | 49 | ≥10 | 57% |
| VIRAT 010204 | ≥11 | 16 | 7 | ≥11 | 56% |

PET runs every 5 s and DM-Count every 6 s per camera; both together add ~3.7 GB
of GPU memory (2.6 → 6.3 GB) and no dropped frames. PET's ShanghaiTech count correction
(`1.375 × raw^0.942`) is implemented but off by default: it was fitted on
foreign footage, so a camera has to earn it locally.

```bash
CROWD_POINT_BACKEND=dense     # enable PET head points
CROWD_DENSITY_BACKEND=dense   # enable DM-Count density
CROWD_POINT_PERIOD_S=5              # advisory cadence per camera
CROWD_DENSITY_PERIOD_S=6
CROWD_MODELS_DIR=./models           # where the .pth files live
CROWD_PET_CORRECTION=1              # opt in to the count correction
```

If a checkpoint is missing or altered the estimators report `available: false`
with the reason, an audit row is written, and the detector path keeps running.

---

## Choosing a detector model

Four families are registered. The **Engine** panel has a picker, `.env` can pin a
default (`CROWD_MODEL`), and `PUT /api/models` switches at runtime: the
replacement is loaded and warmed before the old one is retired, and a failed
load leaves the previous model running.

| Model | Family | Licence | When to use it |
|---|---|---|---|
| `yolo26n.pt` | YOLO26 | AGPL-3.0 | **Default.** Best recovery on packed frames |
| `rtdetr-l.pt` | RT-DETR | Apache-2.0 | Better at medium density, misses packed frames |
| `yolo11n.pt`, `yolov8n.pt` | YOLO | AGPL-3.0 | Fallbacks and cheap comparisons |
| `rf-detr-base.pth` | RF-DETR | Apache-2.0 | needs `pip install rfdetr` |
| `dfine_m_coco.pt` | D-FINE | Apache-2.0 | needs the `dfine` package |

RT-DETR is one switch away, but it is not the default: on the 1080p crowd clip it
finds **36** people where YOLO26n's detail pass finds **156**, because escalation
triggers on a near-empty full-frame count. On the 60 labelled mall frames the
ranking reverses (RT-DETR MAE 9.23, YOLO26n 15.33), so RT-DETR is the better pick
for medium density and the worse one for a packed frame. RF-DETR and D-FINE are
wired and listed but not benchmarked here; the picker shows them disabled until
their package is installed. Switching details: [docs/DETECTORS.md](docs/DETECTORS.md).

## API authentication

Off by default, because the documented deployment binds to loopback behind a
password proxy. Set `CROWD_AUTH_OPERATOR_TOKEN` and/or `CROWD_AUTH_VIEWER_TOKEN`
to require a credential on every API route:

```bash
curl -H "X-API-Key: $CROWD_AUTH_VIEWER_TOKEN" localhost:8000/api/cameras
curl -X PUT -H "X-API-Key: $CROWD_AUTH_OPERATOR_TOKEN" \
     -d '{"name":"rtdetr-l.pt"}' localhost:8000/api/models
```

Reads accept either token; every mutating request needs the operator token.
`/api/health`, the dashboard shell and `/static/` stay public so health checks
and the page keep working; a missing or wrong token is `401`, a viewer attempting
a write is `403`. Tokens are compared in constant time and never logged, and
`?token=` is accepted for streams that cannot send headers.

## The dashboard
The console has three columns: a **collapsible sidebar** (sources, engine, insights), one
**workspace** centred on the live video, and an **inspector** with three tabs — *Alerts*,
*Areas & gates* and *Trust*. The sidebar remembers its state, and choosing a section while
collapsed expands it; secondary things (detection profile, analytics, reports, heatmap
timeline) live there instead of competing with the camera.

| Area | What it shows |
|---|---|
| **Sidebar · Cameras** | Every camera with live count, trust tag and worst-area level; add upload / sample / RTSP sources; remove a source |
| **Sidebar · Engine** | Model, device, CUDA, queue depth, frame age, dropped frames, alert outbox, and the detection profile switch |
| **Sidebar · Insights** | Event intelligence (forecasts, growth, queues, area movement), report downloads, archived heatmap timeline |
| **Stage** | The camera in the selected view, HUD with smoothed people-in-frame, trust badge, FPS and source |
| **KPIs** | People across all sources, tracked identities, inference rate, areas at risk — each with a sparkline |
| **Trend** | Occupancy history with alert markers, so you can see the run-up to a breach |
| **Inspector · Alerts** | Persisted alert history with acknowledge and close lifecycle, plus an unacknowledged counter |
| **Inspector · Areas & gates** | Steps to draw an area, the per-area ring gauge / level / capacity / flow / sparkline, and gate-line entry-exit counts |
| **Inspector · Trust** | Per-camera analytics health: measured, degraded or stale, and why |
| **Wall view** | All cameras as tiles with count and level, for a control-room display |

Areas are called *areas* in the interface; the HTTP API and the database still say `zones`
(`/api/cameras/{id}/zones`, `metrics.zones`), so scripts written against the old names keep
working.

### Camera views

One camera can be inspected five ways, without losing the measurement — the HUD count and
trust badge stay visible in every view:

| View | What you see |
|---|---|
| **All views** | The five views below as tiles of the same camera at once — click a tile to open it full size |
| **Boxes** (default) | Live feed with detection boxes, area outlines and gate lines |
| **Actual** | The clean picture, no boxes and no overlays |
| **Heatmap** | Density heatmap over the scene with area boundaries and an opacity slider |
| **Areas** | The scene dimmed so each area stands out, tinted by level with `name · count/capacity · %` |
| **Flow** | Movement direction and speed per area |

Detection boxes are burnt into the stream by the worker, so choosing a box-free view also
tells that camera's worker to stop drawing them; the choice follows the camera you select.
Everything else is drawn on the overlay canvas and costs the worker nothing.

The **All views** mosaic needs the boxed and the clean picture at the same time, so the stream
cache keeps the last pre-annotation frame and serves it at `?raw=1` (encoded lazily, with a
one-frame cache — a camera nobody watches this way pays nothing). Both images are the same
instant of the same camera, and switching cameras inside the mosaic follows immediately.

Keyboard: `0` all views · `1`–`5` switch camera view · `Enter` save the area you are drawing ·
`Esc` cancel drawing.

### Drawing areas and gate lines

1. Open the **Areas & gates** tab (or press **Edit areas** under the video).
2. Press **New area**. A banner appears above the video and the cursor becomes a crosshair.
3. Click each corner of the area. **Undo point** removes the last one.
4. Give it a name and a capacity, then press **Save area** (or `Enter`).

Areas are stored per camera with a version number in `data/cameras.json`, so the boundary is
the same after a restart, and every alert records which version it fired under. **Auto-grid**
splits the frame into equal areas in one step. A **gate line** is two clicks: it then counts
people crossing it in each direction (`↑` entries, `↓` exits), and both areas and gate lines
can be removed from their rows in the same tab.

The tab also has a ground-plane calibration wizard. Click four visible points
around the measured ground area, then enter each point's surveyed X/Y position
in metres. Use measured coordinates; the wizard cannot validate a camera's
survey accuracy for you.

Each camera also has a profile selector in **This source**. `balanced`, `dense`
and `custom` are applied by the shared inference worker per camera; `global`
inherits the engine profile. Dense mode uses larger inference images and raises
compute cost, so compare it on the target GPU before assigning it broadly.

Clicks are mapped to the video frame itself: the stage adopts the source's real aspect ratio,
so a 2.15:1 camera is not letterboxed inside a 16:9 box and a corner you click is the corner
the detector measures.

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

### Engine settings that pay off

Everything below is an env var; the Engine panel only exposes the detection profile and the
model picker, deliberately — the rest are deployment decisions, not control-room toggles.

| Setting | Cheap / CPU | Balanced (small GPU) | Maximum detail (large GPU) |
|---|---|---|---|
| `CROWD_IMGSZ` | `640` | `640` | `640` global, `dense` profile on selected cameras (1280) |
| `CROWD_TARGET_INFER_FPS` | `2`–`3` | `5`–`8` | `8`–`10` |
| `CROWD_MAX_FRAME_WIDTH` | `960` | `1280` | `1280`–`1920` |
| `CROWD_MAX_FRAME_AGE_MS` | `300` | `500` | `500` |
| `CROWD_BATCH_SIZE` / `CROWD_MAX_BATCH_WAIT_MS` | `1` / `8` | `2` / `8` | `3`–`4` / `8`–`12` |
| `CROWD_STREAM_FPS` / `CROWD_JPEG_QUALITY` | `8` / `55` | `12` / `65` | `12` / `65` |
| PET + DM-Count | off | one, 5–6 s period | both, 5–6 s period |
| Practical cameras | 2–4 | 4–6 | 10–20 |

Two things to leave alone: `CROWD_TILING=auto` (tiles only when median box height drops under
40 px) and `CROWD_DETAIL_PASS=1` (the escalation that turns a wrong `1` into `≥135`). Both
spend compute only where people are small, and disabling them saves frame rate by producing
the wrong number. If **Dropped frames** climbs in the Engine panel, cut
`CROWD_MAX_FRAME_WIDTH` first.

What each knob costs, the per-camera `balanced` / `dense` profiles, and how to tell whether
the engine is keeping up: **[docs/GUIDE.md § Engine](docs/GUIDE.md#the-engine-panel-and-efficiency-presets)**.

---

## Production architecture (implemented)

These pieces exist in the code, are covered by tests (`tests/test_architecture.py`),
and run on the GPU deployment.

| Capability | Where | What it actually does |
|---|---|---|
| **Detector adapter boundary** | `backend/app/adapters.py` | Every backend returns one `DetectionResult`; the pipeline never touches Ultralytics objects. `UltralyticsDetector` also serves Ultralytics-exported `.onnx` weights. |
| **Licence-aware model registry** | `backend/app/detector.py`, `GET /api/models` | Name, backend, version, **licence**, expected VRAM, warm state per model. Hot swap loads and warms the replacement first, keeps the old model on failure, and writes an audit entry. |
| **Tiled inference** | `backend/app/tiling.py` | 3×3 overlapping tiles with cross-tile NMS, gated on *median box height* so compute is only spent where people are small. Measured on the packed-gate clip: **6.8 → 31.0 (2×2) → 61.8 (3×3) people**, 7.3 → 20.6 → 26.5 ms. |
| **Mode router** | `backend/app/router.py` | `sparse`/`moderate` → detector count is operational; `dense`/`unknown` → `ANALYTICS_DEGRADED`, `selected_count: null`, density shown separately as advisory. Hysteresis on recovery, immediate on loss of trust. |
| **Density companion** | `backend/app/adapters.py`, `backend/app/dense.py` | PET / DM-Count dense estimators. Default is an explicit unavailable state — never an invented number. Advisory until a per-camera validation gate is signed off. |
| **Durable alert outbox** | `backend/app/outbox.py`, `backend/app/store.py` | Alerts are persisted before delivery, retried with exponential backoff + jitter, dead-lettered, deduplicated on the alert id, and can be acknowledged/escalated. Survives restarts. |
| **Fair scheduler** | `backend/app/scheduler.py` | One pending request per camera (drop-oldest), round-robin dispatch, micro-batching, and a **frame-age budget** (`CROWD_MAX_FRAME_AGE_MS`) that drops stale frames instead of analysing them. No inline inference on camera threads. |
| **Analytics health** | `backend/app/analytics.py` | Dead-man switch per camera: no inference, count collapse on a healthy stream, and count drift against the previous window all become `degraded` rather than a comfortable zero. |
| **Geometry versioning & audit** | `backend/app/pipeline.py`, `backend/app/store.py` | Zone edits bump `zones_version`; calibration is versioned and attributed. Every alert records the zone version, calibration version, model version and estimation mode it fired under. |
| **Throughput** | `app/pipeline.py` | People-per-minute in/out per virtual line — the number that predicts a surge, not just instantaneous head-count. |

### When the detector cannot see the crowd

A 768×432 clip of a few hundred people yields **0.3 detections** at the default
settings, because each person is ~12×25 px. Nothing in that result says the
detector failed — an empty road returns the same thing — so the dashboard used
to show a trusted "1". `app/detail.py` fixes this without inventing numbers:

| pass | people found | cost |
|---|---|---|
| full frame @ 640 (default) | 0.3 | 10 ms |
| 2×2 probe @ 640 (triggered only when the default found ≤ 6) | 4.4 | 15 ms |
| 3×3 detail @ 1280 (only if the probe disagrees) | **135.4** | 58 ms |

Two cheaper triggers were measured and rejected: edge density scored 0.42–0.89
on *every* sample clip (it measures texture, not people), and frame-to-frame
motion was ~0.000 on the packed clips (standing crowds in near-static shots).
Escalation is therefore detection-against-detection and needs no per-site
calibration. Mode is sticky per camera, so a packed scene cannot alternate
between a trusted "7" and a lower bound of 146 on consecutive frames.

What the operator sees is a floor, not a count: the HUD reads `≥ 135`, the badge
says `DETECTOR · LOWER BOUND — full frame 6 → detail 135 · undercount likely`,
and capacity alerts are suppressed in favour of one advisory per cooldown
("Dense crowd: 135 people detected is a lower bound, not a count … manual
observation required"). A 3×3 pass on a dense camera costs about half its frame
rate (≈8 → 4 FPS per camera), which is why it is gated so hard.

```bash
curl -s localhost:8000/metrics | grep detail_
curl -s localhost:8000/api/cameras | jq '.[] | {name, tier: .metrics.tier,
  count: .metrics.count, primary: .metrics.primary_count, mode: .metrics.mode}'
```

```bash
curl -s localhost:8000/api/models | jq '.models[] | {name, license, status}'
curl -s localhost:8000/api/analytics/health | jq '.cameras'
curl -s -X POST localhost:8000/api/alerts/<id>/ack -d '{"by":"shift-supervisor"}'
curl -s localhost:8000/api/audit | jq
```

Benchmark any model/tiling combination on the same clips:

```bash
python backend/tools/bench_models.py --clips samples/a.mp4 samples/b.mp4 \
    --models yolo26n.pt --tiling off,2x2,3x3 --frames 30
```

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

The dev scripts (`backend/tools/start_dev.sh`) relocate `data/`, `uploads/` and `samples/` into
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
./.venv/Scripts/python.exe backend/tools/fetch_footage.py --list
./.venv/Scripts/python.exe backend/tools/fetch_footage.py                # fetch what's missing
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

Long version, including the GPU VM walkthrough, HTTPS + password proxy, backups,
upgrade/rollback and a troubleshooting table:
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

Design context for anything beyond a handful of cameras — why MediaMTX, NVDEC, Triton and
TimescaleDB, and how to size a single L40S for ~100 cameras — is in
**[docs/PRODUCTION_ARCHITECTURE.md](docs/PRODUCTION_ARCHITECTURE.md)**.

---

## Development tools

| Tool | What it does |
|---|---|
| `backend/tools/install.sh` / `start_dev.sh` / `start_dev.bat` | Project-local venv, pinned deps, server start |
| `backend/tools/ui_check.py` | Renders the dashboard in headless Chromium over CDP, prints DOM facts, fails on console errors, saves a screenshot — no GUI needed |
| `backend/tools/fetch_footage.py` | Downloads/trims the test clip library and writes `manifest.json` |

```bash
./.venv/Scripts/python.exe backend/tools/ui_check.py http://127.0.0.1:8000/ shot.png
```

---

## Project layout

```
app/
  main.py        Composition root: middleware, startup, router include
  state.py       Model manager + camera registry shared by the routers
  api/           One router module per concern
    cameras.py   CRUD, zones, gate lines, calibration, dense-count validation
    alerts.py    Alert stream, evidence, ack/close/escalate, outbox, audit
    engine.py    Detector registry, health, Prometheus metrics, analytics health
    insights.py  Event analytics, heatmap history, printable PDF reports
    stream.py    MJPEG streams, snapshots, dashboard websocket
  pipeline.py    CameraWorker (capture → detect → track → zones) + CameraRegistry
  detector.py    ModelManager: model families, warm-before-swap, tiling policy
  adapters.py    Detector/estimator adapters (YOLO, RT-DETR, RF-DETR, D-FINE, PET, DM-Count)
  detail.py      Detail-pass escalation for small people
  router.py      Counting router: detector floor vs dense estimates
  dense.py       PET / DM-Count wrappers with checksum-verified checkpoints
  zones.py       Zone model, occupancy/density, level thresholds
  alerting.py    Sustained-threshold alert engine with hysteresis and cooldown
  outbox.py      Durable alert outbox with HMAC webhook transport
  district.py    Edge → district publisher and district aggregation (plan §8)
  heatmap.py     Decaying density grid from foot points
  store.py       SQLite history, alerts, audit, outbox tables
  ingest.py      Upload/sample/RTSP source handling
  auth.py        Optional operator/viewer API tokens
  config.py      Env-var configuration
static/          Dashboard (index.html, app.js, style.css)
backend/         Application code, dense services, and operator tools
  app/           FastAPI app, pipeline, adapters
  services/dense PET + DM-Count network definitions
  tools/         Install, start, UI check, footage fetcher, benchmarks
docs/            Architecture, deployment, footage, screenshots, results report
Dockerfile, docker-compose.yml, docker-compose.gpu.yml
```

---

## Known limits and roadmap

Honest state of the project — it is a working prototype with production-shaped pieces, not a
certified life-safety system.

- **Density is relative, not metric.** `persons/10k px²` depends on camera geometry. A
  per-camera homography to metres would give real people/m², which is what the literature's
  safety bands (≈4–5 ppl/m² safe, > 6 ppl/m² flow collapse) are expressed in.
- **Detector recall collapses in packed crowds** (see [docs/FOOTAGE.md](docs/FOOTAGE.md)).
  PET + DM-Count already run as advisory dense estimators; per-camera validation is still
  required before treating them as operational counts.
- **Frozen / stale cameras are marked degraded**, but operators must still treat a dark or
  stalled feed as “unknown”, not “empty and safe”.
- **No accounts or SSO.** Optional operator/viewer API tokens exist and HMAC-signed webhook
  delivery is proven end-to-end, but there are no users, sessions or SMS gateway.
- **One machine per node.** Camera nodes publish signed summaries to a district node
  (`app/district.py`); the central tier (Triton, TimescaleDB, a control plane) is still
  design only, and there is no cross-camera de-duplication between nodes.
- Zone counts are per-camera with no cross-camera de-duplication; overlapping views of the
  same gate will double-count.

---

## Licensing

- Application code in this repository: as you choose for your project.
- **Ultralytics YOLO is AGPL-3.0.** Serving it as a network service for third parties can
  oblige you to publish corresponding source. For internal demos this is normally fine;
  commercially, buy an Ultralytics licence or swap to an Apache detector such as RT-DETR.
- **PET / DM-Count service code** under `backend/services/dense` is academic-use oriented;
  keep checkpoints local and validate claims before operational safety use.
- Secrets (`MONGODB_URL`, API tokens, webhook secrets) belong only in `.env` (gitignored).
  Rotate any credential that was ever shared outside the team.
- Test footage: Creative Commons for four Archive.org clips (three CC BY 2.0, one CC BY-NC),
  evaluation-only for the YouTube clips. Per-clip licences in
  [docs/FOOTAGE.md](docs/FOOTAGE.md).
