# Operator guide

How to read the dashboard, draw areas, and choose engine settings that keep the
counts useful without wasting GPU. Setup and deployment live in the
[README](../README.md) and [docs/DEPLOYMENT.md](DEPLOYMENT.md); this file is the
day-to-day manual.

- [Read this first: estimates vs counts](#read-this-first-estimates-vs-counts)
- [Adding sources](#adding-sources)
- [The Engine panel and efficiency presets](#the-engine-panel-and-efficiency-presets)
- [Areas (zones): creating and applying them](#areas-zones-creating-and-applying-them)
- [Gate lines](#gate-lines)
- [Ground-plane calibration](#ground-plane-calibration)
- [Camera views and keyboard](#camera-views-and-keyboard)
- [KPIs: what each number is](#kpis-what-each-number-is)
- [Alerts: levels, lifecycle, delivery](#alerts-levels-lifecycle-delivery)
- [Trust tab and analytics health](#trust-tab-and-analytics-health)
- [Dense estimators and the validation gate](#dense-estimators-and-the-validation-gate)
- [Insights, reports, heatmap archive](#insights-reports-heatmap-archive)
- [Auth, metrics, archival](#auth-metrics-archival)
- [Where data lives](#where-data-lives)
- [Troubleshooting](#troubleshooting)

---

## Read this first: estimates vs counts

Every camera reports several numbers, and they are deliberately not merged:

| What you see | Meaning |
|---|---|
| `386` (plain) | Detector count in a sparse/moderate scene; the router trusts it |
| `≥135` | **Lower bound.** The detector found at least this many; the detail pass says the real crowd is larger and cannot be resolved |
| `≈516` | **Consensus estimate.** PET and DM-Count agree within 25%; the badge names both and their spread |
| `DENSE · ANALYTICS DEGRADED` | The frame is too packed for the detector to be operational; advisory numbers only |

`selected_count` in the API stays `null` until an operator promotes an estimator
for that camera (see [the validation gate](#dense-estimators-and-the-validation-gate)).
A `≈` number is information for a human, never a silent promotion.

---

## Adding sources

**+ Add** in the sidebar, then pick one:

| Type | What happens |
|---|---|
| **Sample** | Bundled clip library; subfolders (`Pilgrim`) become groups. Clips loop |
| **Upload** | Any `.mp4`/`.avi`/`.mov`/`.mkv`/`.webm`/`.m4v`/`.mpg`/`.wmv` up to `CROWD_MAX_UPLOAD_BYTES` (1 GiB default); loops |
| **Source** | `rtsp://user:pass@host:554/stream`, an HTTP video URL, or a webcam index (`0`). Network drops reconnect |

One worker thread per source; **✕** on a source removes it and stops its worker.
An uploaded file is deleted with the last source that used it. Credentials in a
stream URL are redacted before any remote archive or log line.

The camera **status** is `running`, `connecting`, or a health failure — see
[Trust](#trust-tab-and-analytics-health).

---

## The Engine panel and efficiency presets

The **Engine** sidebar shows the shared model and the queue, not a per-camera
setting:

| Row | Read it as |
|---|---|
| **Model** | Active detector and whether it loaded |
| **Device / CUDA** | `CUDA GPU` or `CPU`; CUDA availability is reported separately from use |
| **Sources** | Running / total workers |
| **Dashboards** | WebSocket clients currently receiving snapshots |
| **Inference queue** | Pending frames and the p95 *frame age* — how stale the frame being analysed was |
| **Dropped frames** | Frames discarded because they aged past `CROWD_MAX_FRAME_AGE_MS` (good: stale frames are dropped, not analysed) |
| **Alert outbox** | Pending / failed webhook deliveries, or "no webhook configured" |

Two switches live here:

- **Detection profile** — `Balanced` (confidence 0.30, `imgsz` 640), `Dense crowd`
  (confidence 0.15, `imgsz` 1280), or `Configured defaults` (whatever
  `CROWD_CONF` / `CROWD_IMGSZ` say). Global, applies to every camera without its
  own profile.
- **Detector model** — hot swap; the replacement is loaded and warmed before the
  old one is retired, and a failed load keeps the previous model running.

Per camera, **This source → Detection profile** offers `Use global profile`,
`balanced`, `dense`, `custom`. This is the most useful efficiency lever: put
`dense` only on the two cameras that actually need it (packed ghat steps), keep
the rest on `balanced`/global.

### Presets

Start from the hardware, then measure. Numbers below are the ones measured for
this project, not projections.

| Setting | CPU only (8 vCPU) | Small GPU (8–12 GB, e.g. 3050) | Large GPU (24 GB+, L40S/6000) |
|---|---|---|---|
| `CROWD_MODEL` | `yolo26n.pt` | `yolo26n.pt` | `yolo26n.pt` |
| `CROWD_IMGSZ` | `640` | `640` | `640` (dense profile raises to 1280 for selected cameras) |
| `CROWD_MAX_FRAME_WIDTH` | `960` | `1280` | `1280`–`1920` if people are tiny |
| `CROWD_TARGET_INFER_FPS` | `2`–`3` | `5`–`8` | `8`–`10` |
| `CROWD_STREAM_FPS` | `8` | `12` | `12` |
| `CROWD_JPEG_QUALITY` | `55`–`60` | `65` | `65` |
| `CROWD_WS_PERIOD` | `1.0` | `0.6` | `0.6` |
| `CROWD_BATCH_SIZE` / `CROWD_MAX_BATCH_WAIT_MS` | `1` / `8` | `2` / `8` | `3`–`4` / `8`–`12` |
| `CROWD_TILING` | `auto` | `auto` | `auto` |
| `CROWD_DETAIL_PASS` | `1` | `1` | `1` |
| PET / DM-Count | leave **off** | one of them, period 5–6 s | both, period 5–6 s |
| Practical cameras | 2–4 | 4–6 | 10–20 |

Measured anchors: 3.6 inferences/s on CPU torch at `imgsz=1280` (1.8/s with two
cameras); 3.2 FPS in the CPU container at 640; 32 inferences/s on an RTX 3050 at
640 — four cameras at 8 FPS, 1.8 GB VRAM. PET + DM-Count together add ~3.7 GB and
no dropped frames at a 5/6 s cadence.

### What each knob actually does

| Env var | Default | Effect and when to change it |
|---|---|---|
| `CROWD_IMGSZ` | `640` | Model input size. The single biggest cost. Raise to 960/1280 only with GPU headroom; 1280 roughly triples model time |
| `CROWD_CONF` | `0.30` | Lower finds more small people and more false boxes. Dense profile uses 0.15 |
| `CROWD_TARGET_INFER_FPS` | `5` | Per-camera budget. Above ~8 you buy latency, not accuracy, for crowd counting |
| `CROWD_MAX_FRAME_WIDTH` | `1280` | Downscale before inference. Lower = cheaper decode/resize, fewer pixels per head. Raise only if a distant ghat needs it |
| `CROWD_MAX_FRAME_AGE_MS` | `500` | Frames older than this are dropped. Lower (300) when overloaded, so counts describe *now* |
| `CROWD_FRAME_QUEUE_SIZE` | `2` | Pending frames per camera, drop-oldest |
| `CROWD_BATCH_SIZE` | `1` | Micro-batch across cameras. 2–4 helps a GPU with many sources; 1 is best on CPU |
| `CROWD_MAX_BATCH_WAIT_MS` | `8` | How long a batch waits to fill. Larger = better GPU utilisation, staler frames |
| `CROWD_TILING` | `auto` | `auto` tiles only when median box height < `CROWD_TILE_MIN_BOX_HEIGHT_PX` (40 px). Leave it: 3×3 recovered 6.8 → 61.8 people on the packed gate clip at 20.6 ms |
| `CROWD_DETAIL_PASS` | `1` | Escalation when the full frame finds ≤ `CROWD_DETAIL_PROBE_BELOW` (6) people. This is what turns a wrong `1` into `≥135`; disabling it saves ~half the frame rate on dense cameras and loses the crowd |
| `CROWD_POINT_PERIOD_S` / `CROWD_DENSITY_PERIOD_S` | `5` / `6` in `.env.example` (1.5 if unset) | Estimator cadence per camera. 5–6 s is plenty for a trend; 1.5 s costs GPU for numbers nobody reads |
| `CROWD_PET_CORRECTION` | `0` | ShanghaiTech-fitted count correction. Off until a camera is validated locally |
| `CROWD_LOOP` | `1` | Loop file sources (tests, demos) |
| `CROWD_DECODER` | `auto` | Tries GStreamer (NVDEC when available), then FFmpeg, then OpenCV's own backend |

Two levers are deliberately **not** exposed as toggles: the tiling gate and the
detail escalation. Both were measured, both prevent silently wrong counts at the
cost of compute, and both are already adaptive per camera.

How to tell whether the engine is keeping up: open **Engine**. If
**Dropped frames** climbs while p95 frame age sits near the budget, lower
`CROWD_MAX_FRAME_WIDTH` (or distribute cameras across processes) rather than
turning off the detail pass. `curl -s localhost:8000/api/system | jq .scheduler`
gives the same counters.

---

## Areas (zones): creating and applying them

An **area** is a named polygon on one camera with a capacity. It is the unit of
occupancy, level colour, alerting, gate flow and the Areas/Flow views. There is
no separate "apply" step: saving an area hands it to that camera's worker on the
next frame, and it survives restarts.

### Draw an area with the mouse

1. Select the camera you want to measure (areas belong to one camera, not to the
   whole site).
2. Open **Areas & gates** — the inspector tab, or the **Areas & gates** button
   above the video.
3. Press **＋ New area**. The cursor becomes a crosshair and a banner appears
   over the video.
4. Click each corner of the polygon. Corners follow the real frame, so the point
   you click is the point measured. **Undo point** removes the last one; at least
   three are required.
5. Type a name (defaults to `Area N`) and a **capacity** in people for that
   polygon, then press **Save area** (or `Enter`). `Esc` cancels.

The area appears in the inspector row with a ring gauge, level, capacity, flow
and sparkline, and in the **Areas** view tinted by level.

Good areas are drawn at *pinch points*: the top of the steps, the queue mouth, the
narrow walkway — small polygons where a capacity is meaningful. A polygon that
covers half the frame gives you a number you cannot act on.

### Auto-grid

**Auto-grid** asks three questions — columns (1–6), rows (1–6), capacity per
area — and replaces the camera's areas with that grid. It is the fast way to get
every part of a wide ghat covered, and it confirms before overwriting existing
areas. Names come out as `Zone 1…N`; rename by redrawing if you need site names.

### Change capacity, delete, verify

- **Capacity** is editable inline in each area row (the number field next to the
  name); changing it re-saves the geometry with the new capacity and takes effect
  immediately.
- **✕** on the row deletes that area.
- Redrawing or auto-gridding bumps the camera's `zones_version`; every alert
  records the version it fired under, so history stays interpretable.

Verify from a shell:

```bash
curl -s localhost:8000/api/cameras/<id> | jq '.zones[] | {name, capacity, warn_pct, metrics}'
```

### How a person ends up inside an area

- The detector's box is tracked by ByteTrack, so each person has one identity.
- A person is inside if the **foot point** (bottom-centre of the box) is inside
  the polygon — not the box centre, which is what stops two overlapping queues
  from double-counting.
- The raw count is smoothed with an EMA (60/40) to remove jitter.
- `occupancy % = smoothed count / capacity`; `density` is persons per 10,000 px²
  of polygon area, and persons/m² once calibration exists.

| Level | Rule | Dashboard |
|---|---|---|
| **OK** | below 2/3 of the warning line | green |
| **BUSY** | ≥ 2/3 of the warning line | yellow |
| **WARNING** | ≥ `CROWD_WARN_PCT` (75 %) | orange, one alert on the rising edge |
| **OVERCROWDED** | ≥ 100 % for `CROWD_ALERT_SUSTAIN` (5 s) | red, capacity alert + chime, repeats at most every 60 s |

The warning line is stored per area as `warn_pct` (75 by default); the UI does
not expose it, the API does.

### If an area shows 0

| Symptom | Cause |
|---|---|
| Area exists, count stays 0 | Nobody's foot point is inside it — usually drawn on the wrong camera, or around a distant crowd smaller than the detector can resolve at `imgsz=640` |
| Area row missing entirely | Areas are per camera; you are looking at a different source |
| Count jumps but level stays OK | Capacity is set far above the real crowd, so occupancy never reaches 75 % |
| **Areas at risk** is 0 | No area is at WARNING/OVERCROWDED on a running camera — by definition, not a bug |
| Areas vanish after a redraw | Auto-grid replaces all areas on that camera; that is what the confirm dialog warns about |

---

## Gate lines

A gate line counts crossings, which is what predicts a surge; occupancy alone
does not. Press **＋ Add gate line**, click the two ends on the video, name it,
and it saves on the second click. Each line reports **↑ entries** and **↓ exits**
by direction, plus people-per-minute throughput. Lines can be removed from their
rows. Keep lines perpendicular to the walking direction and clear of the frame
edge, where tracks are born and die.

---

## Ground-plane calibration

Calibration turns pixel density into people per square metre:

1. In **Areas & gates**, start the calibration wizard.
2. Click four points on the ground plane around a measurable area.
3. Enter each point's surveyed X/Y in metres.

The result is a homography per camera, versioned and attributed. A new
calibration recomputes every area's m² and **revokes any previous estimator
promotion** on that camera, because the validation was signed against the old
geometry. Without calibration, density stays relative (`persons/10k px²`).

---

## Camera views and keyboard

| View | Shows |
|---|---|
| **All views** | The five views as tiles of the same instant; click a tile to open it |
| **Boxes** | Live feed with detection boxes, area outlines, gate lines |
| **Actual** | Clean picture, no boxes |
| **Heatmap** | Decaying density grid over the scene, opacity slider |
| **Areas** | Scene dimmed, areas tinted by level with `name · count/capacity · %` |
| **Flow** | Direction and speed per area |

Keys: `0` all views · `1`–`5` switch view · `Enter` save the area being drawn ·
`Esc` cancel drawing. Boxes are burnt into the stream by the worker, so a
box-free view also stops that camera from drawing them; everything else is drawn
in the browser and costs the server nothing.

The **Wall** view tiles every camera with count and level for a control-room
screen; **Fullscreen** expands it.

---

## KPIs: what each number is

The KPI row is a site-wide roll-up, so it can legitimately differ from one
camera's card:

| KPI | Definition | Note |
|---|---|---|
| **People, all sources** | Sum of each running camera's headline number — consensus `≈` when PET and DM-Count agree within 25 %, otherwise the detector floor/`≥` | Same number the camera cards show |
| **Tracked identities** | Sum of active ByteTrack IDs | A person seen by two cameras counts twice; this is a workload number, not a crowd size |
| **Inference rate** | Mean per-camera inference FPS | Drops when detail escalation or dense profile engages |
| **Areas at risk** | Count of areas at WARNING or OVERCROWDED on running cameras | Stays 0 until areas exist and fill |

Sparklines are the last 90 samples (300 s for the total). The trend chart shows
the total with alert markers.

---

## Alerts: levels, lifecycle, delivery

- A **warning** fires once on the rising edge past `warn_pct`; hysteresis stops
  repeat spam while the area hovers near the line.
- A **capacity** alert needs occupancy ≥ 100 % sustained for
  `CROWD_ALERT_SUSTAIN` seconds, then repeats no more often than
  `CROWD_ALERT_COOLDOWN` (60 s).
- In dense mode, capacity alerts are suppressed in favour of one **advisory** per
  cooldown ("Dense crowd: 135 people detected is a lower bound… manual
  observation required"). A floor cannot prove a capacity breach.
- Alerts are written to SQLite **before** delivery, then acknowledged and
  **closed** from the Alerts tab (close requires acknowledge). The API also
  supports escalate, evidence snapshots and audit.
- Optional webhook delivery (`ALERT_WEBHOOK_URL`, `ALERT_WEBHOOK_SECRET`) is
  HMAC-signed, retried with backoff and jitter, deduplicated and dead-lettered;
  the outbox survives restarts. The Engine panel shows its depth.
- History survives restarts; `GET /api/alerts/export` returns CSV.

---

## Trust tab and analytics health

**Trust** answers "can I believe this camera right now?":

| Status | Meaning |
|---|---|
| measured | Fresh frames, plausible counts |
| degraded | No inference, a count collapse on a healthy stream, count drift, or the router deciding the crowd is too dense for the detector |
| stale / frozen / dark / blurry / low_contrast | Frame quality or freshness checks failed (`CROWD_FRAME_STALE_S`, `CROWD_FROZEN_FRAME_S`, brightness/contrast/blur thresholds) |

A dead camera reads `degraded`, never a comfortable `0`. `/api/health` returns
`degraded` while any camera is unhealthy or analytics-degraded;
`/api/analytics/health` lists the reasons per camera.

---

## Dense estimators and the validation gate

PET (head points) and DM-Count (density map) run beside the detector and are
advisory by default:

1. Enable them (`CROWD_POINT_BACKEND=dense`, `CROWD_DENSITY_BACKEND=dense`) and
   keep `models/pet_shab.pth`, `models/dmcount_shab.pth` present; every load
   verifies SHA-256.
2. **Calibrate the camera.** Promotion is refused outright without a ground plane,
   because validation is signed against a geometry.
3. Declare the checkpoint validated on this node: `CROWD_POINT_MODEL_VALIDATED=1`
   for PET, `CROWD_DENSITY_VALIDATED=1` for DM-Count. Without it the API rejects
   promotion with “checkpoint is not enabled as validated on this node”.
4. Judge them per camera: compare `head_point_count`, `density_count` and the
   detector floor against manual counts from the same instant, then record
   checked pairs with `POST /api/cameras/<id>/validation-samples`
   (`source` is `head_point` or `density`, plus `manual_count`, `model_count`).
   Adding a sample invalidates any previous sign-off until you review again.
5. Promote with `PUT /api/cameras/<id>/estimation-policy`
   (`{"selected_source": "head_point"}` or `{"selected_source": "density"}`).
   It requires `CROWD_VALIDATION_MIN_SAMPLES` (20) samples for that source and a
   MAPE within `CROWD_VALIDATION_MAX_MAPE` (20 %); the approval is valid for
   `CROWD_VALIDATION_TTL_DAYS` (30) days and records the calibration version.
6. Revoke with `{"selected_source": "none"}`. Recalibrating the camera revokes the
   promotion automatically; that is intentional.

Only then does `selected_count` carry a number. Until a camera is promoted, the
badge and the API keep showing the floor/consensus instead.

---

## Insights, reports, heatmap archive

- **Event intelligence** — five-minute growth forecasts per camera (linear trend
  on recent samples), queue length and estimated wait where exit history exists,
  and area movement. Forecasts are extrapolations; the panel says so.
- **Reports** — `Daily PDF` and `Event PDF (6h)` are generated from stored
  samples and alerts for shift handover.
- **Heatmap timeline** — archived heatmap grids with a scrubber and Play, for
  reviewing where a crowd built up before an alert.

---

## Auth, metrics, archival

- **Tokens** — set `CROWD_AUTH_OPERATOR_TOKEN` (read + write) and/or
  `CROWD_AUTH_VIEWER_TOKEN` (read only). Send `X-API-Key`; streams also accept
  `?token=`. Missing token → 401, viewer write → 403. `/api/health`, the page and
  `/static/` stay public.
- **Metrics** — `GET /metrics` is Prometheus text: per-camera counts, estimates,
  scheduler and estimator counters. Useful alerts: `godavari_density_estimate`,
  `godavari_head_point_estimate`, plus `detail_*` and queue series.
- **MongoDB archive** — optional (`MONGODB_URL`, `MONGODB_DB_NAME`): cameras
  without source paths, alerts, audit, count samples and heatmaps are mirrored in
  the background. SQLite stays the source of truth; an outage only queues.
- **District** — `CROWD_DISTRICT_PUSH_URL` publishes signed summaries from this
  node to a district node (`CROWD_DISTRICT_SECRET` on both sides).

---

## Where data lives

| Path | Contents | Back up |
|---|---|---|
| `data/cameras.json` | Cameras, areas, gate lines, calibration, profiles | **yes** |
| `data/crowd.db` | Alerts, audit, outbox, validation samples, histories | **yes** |
| `data/evidence/` | Alert evidence snapshots | optional |
| `uploads/`, `samples/` | Media | optional |
| `models/` | Checkpoints | keep with the deployment |

`start_dev.sh` relocates these under `.data/`; Docker keeps the whole tree in the
`godavari-data` volume. Deleting `cameras.json` loses every area and gate line —
there is no undo beyond your backup.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| App exits 2 at startup | Weight preflight: LFS pointers or an enabled estimator's checkpoint missing. It prints the exact command |
| `PET 0`, `head_point_count: null` | Estimator not enabled, checkpoint missing/corrupt (SHA-256), or the dense service failed — `/api/models` shows `available` and the reason |
| Counts look far too low on a packed clip | Expected for box detectors; the detail pass gives `≥N`. Enable PET/DM-Count for a near-true number |
| Inference FPS halved on one camera | That camera is in dense/detail mode (3×3 at 1280). Correct behaviour; move it off `dense` profile if you need frame rate |
| Dropped frames climbing | Lower `CROWD_MAX_FRAME_WIDTH`, reduce `CROWD_TARGET_INFER_FPS`, or split cameras across processes |
| Dashboard shows no data | `curl -s localhost:8000/api/health`; a `degraded` status names the camera |
| Alerts not arriving by webhook | Engine → Alert outbox; `/api/audit`; check the HMAC secret both sides |
| Zone numbers do not match the HUD | The HUD is people-in-frame; areas only count foot points inside their polygon |
