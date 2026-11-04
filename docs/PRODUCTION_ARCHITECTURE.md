# Godavari — production architecture for large-scale crowd monitoring

This document is the result of a research pass over three reference repositories, the
crowd-analytics literature, and current (2025–2026) NVIDIA/deployment documentation. It
answers three questions:

1. What can we actually borrow from the three repos we were pointed at?
2. Which regional / density techniques are worth their cost, ranked?
3. What should this platform become in order to run 100+ cameras at a mass gathering?

Everything below is scoped to **the code that exists in this repository today**: a FastAPI
service with one decode thread per camera, a shared YOLO detector, ByteTrack tracking,
polygon zones, EMA smoothing, a SQLite alert history and a canvas overlay dashboard.

---

## 1. The three reference repositories — what they are, what they are not

We read the full file trees and source of all three. Their names promise more than the code
delivers, and that matters for planning: **none of them contains a usable region/perspective
layer**, which is exactly the piece a production crowd system needs most.

### `Steve-Dusty/nvidia-secure` — fall / fight / distress detection, not crowd monitoring

- Cloud NVIDIA NIM calls (Grounding-DINO, Nemotron-VL, BodyPose, Parakeet ASR) plus a
  parallel DeepStream config set (PeopleNet INT8 → NvDCF tracker → bodypose SGIE).
- **No polygon zones, no ROI, no homography, no density map.** The only spatial data is a
  hard-coded camera lat/lon list for a Mapbox view.
- Counting is `len(detections)`; "density" is `people / megapixels`, which then drives
  `if density > 0.5: CROWD_INCIDENT` — at 2 MP that fires on a *single* person. Do not copy.
- Real-time is impossible on the NIM path: three serial HTTP calls per sampled frame.

**Worth stealing (design patterns only):**

| Pattern | Why it transfers |
|---|---|
| Multi-modal severity fusion with a corroboration bonus (`max(visual, audio)`, `+1` when both are elevated) | Cheap, explainable way to combine a vision count surge with an audio "chaos" scalar into one severity |
| Pairwise interaction events with a stable de-dup key (`min(id1,id2)_max(id1,id2)`) | Exactly the shape needed for "two streams converging" / group-formation events |
| Incident lifecycle: open on detection, **auto-close on recovery**, 3 s cooldown | Our alerting currently only opens alerts; explicit close-on-recovery is a real improvement |
| Deterministic JSONL audit trail with daily rotation, ISO timestamps | Trivially cheap, plays well with Loki/ClickHouse later |
| Severity → concrete action ladder (dispatch?, priority 1–5, specific wording) | Operators act on playbooks, not on numbers |
| Fail-closed defaults when a model/LLM call fails (escalate, never silently de-escalate) | Correct safety posture for a crowd system |

### `sararaa/Pigeon_AgentHacks` — traffic simulation, no vision at all

- FastAPI + Redis pub/sub + WebSocket + React/Google Maps. `requirements.txt` has no
  `torch`, `opencv` or `ultralytics`: it simulates *vehicles* as agents on an OSM graph.
- Zones are street-name substring matches inside a 500 m geocode buffer — not geometry.
- Ships committed API keys and a `.env` with merge-conflict markers (their problem, not ours,
  but a reminder to keep our secrets in env files).

**Worth stealing:**

| Pattern | Why it transfers |
|---|---|
| **Redis pub/sub → WebSocket fan-out** decoupling workers from the API tier | The right shape for many cameras: workers publish per-zone state, API fans out, producers and consumers scale independently |
| Fixed-rate, drift-corrected scheduler loop (`sleep(interval - elapsed)`) | This is the correct skeleton for a per-camera inference cadence that must hold target FPS under variable latency |
| Per-entity neighbour query as a density primitive (count within R, converted to m²) | A cheap local-density signal that needs no density model |
| Per-zone model registry with lag features (`lag_1..3`, `lag_24`, hour, weekday) | Directly reusable as per-zone occupancy forecasting for arrival surges |
| LLM narrative layer over structured metrics (JSON in → prose briefing out) | "Gate 4 is packed, expect a surge in 20 minutes" is a genuinely differentiating operator feature |

### `zaki1003/Crowd-Counting-Platform` — the only vision repo, a research demo

- Flask + CSRNet (VGG-16 + dilated backend), FIDTM (HRNet), P2PNet (VGG16-BN, point-based),
  and a crowd-specific YOLOv5 (`C3RFEM` + `MultiSEAM` heads, RetinaFace-style anchors).
- Count = `sum(density_map)` for CSRNet; NMS-style peak extraction for FIDTM; thresholded
  point scores for P2PNet. **No area normalisation, no homography, no zones, no tracking.**
- Ships broken: no Dockerfile though `cloudbuild.yaml` builds one, hard-coded
  `/home/zaki/...` weight paths, `method == 'crnet'` vs UI value `csrnet` (CSRNet unreachable),
  models loaded at import time, shared `static/org_img.jpg` across requests.

**Worth stealing:**

| Pattern | Why it transfers |
|---|---|
| One polymorphic inference contract: every backend returns `(count, visualisation)` | Formalise as `infer(frame) -> CountResult(count, points, density_map, model_id, latency_ms)` so detector and density model are interchangeable |
| Emit **both** density-integral and point-count as a cross-check | A free sanity alarm when one estimator silently fails |
| FIDTM's k-NN per-point scale estimate (`sigma ≈ 0.1 × sum of 3 nearest distances`, capped at 4 % of min dimension) | A per-person apparent-size proxy — the cheap version of a perspective map |
| Crowd-specific detection heads (RetinaFace anchors + attention blocks) | The single most transferable *modelling* idea for dense oblique scenes |
| TensorRT/ONNX engine fallback in the detector backend | The difference between 1 and 20 cameras per GPU |
| MJPEG multipart generator as a fallback delivery tier | Works through every proxy, zero JS — keep it for low-value tiles |

---

## 2. Regions: what "identifying a region" should mean here

Ranked by value per unit of effort for a system that already has YOLO + ByteTrack foot-points.

### Tier 1 — days of work, unlocks everything else

1. **Metric homography per camera.** Four point pairs on a known ground plane →
   `cv2.findHomography(..., cv2.RANSAC)`. Foot-points become metres; polygon areas become m²;
   every density number becomes people/m². **Validate against a known distance** (a kerb, a
   lane marking). Without validation, every downstream number is fiction.
2. **Metric density grid + safety bands.** 0.5–1 m cells, `count / area`, recomputed at 1–4 Hz.
   The internationally used danger thresholds are well established: **< 4–5 ppl/m² is safe
   standing density; above ~4 compression-asphyxia risk rises sharply; above ~6 flow collapses
   and laminar→turbulent transition becomes likely.** Those are the numbers our alerts should
   speak in, not "occupancy % of a guessed capacity".
3. **Walkable mask from accumulated foot-point occupancy.** Cells that see foot-points over
   7–30 days are walkable; never-visited cells are walls/planters. This is free, self-calibrating,
   and more trustworthy than a segmentation model asked to see ground under a crowd.
4. **Track-based metric velocity field.** Per-ID velocity from finite differences on
   homography-mapped positions → per-cell vector field. Gives congestion/stop detection
   (mean speed < ~0.2–0.3 m/s while density > ~2 ppl/m²), counter-flow via circular statistics
   of direction (`R` low + high density = opposing streams, a genuine crush risk factor), and
   queue detection via direction coherence + near-constant metric spacing. No new model, ~50 lines.

### Tier 2 — weeks, high value

5. **HDBSCAN hotspots with hysteresis.** Cluster foot-points (`eps ≈ 1.2–1.5 m`, `min_samples=5`);
   HDBSCAN beats DBSCAN here because density varies across a site (dense queue vs sparse concourse).
   Track clusters frame to frame (`linear_sum_assignment`), store `d(count)/dt` and `d(area)/dt`,
   alert on *relative* growth (+20 % in 60 s), require ≥ 3 s persistence.
6. **Density-aware speed classification.** Bucket velocities into halt/slow/normal/fast and score
   percentiles **conditioned on local density** (the VelocityNet formulation). "Fast" is anomalous
   at 4 ppl/m² in a queue and normal on an empty concourse — fixed thresholds both cry wolf and miss events.
7. **Pedestrian-scale auto-calibration** for cameras nobody will click points on: regress
   `bbox_height_px` against `foot_y_px` over a day of tracks, extrapolate the horizon, anchor
   absolute scale with an adult-height prior (~1.70 m ± 0.09). Expect **±10–20 % systematic error**
   — label it, and prefer true homography wherever a safety decision depends on the threshold.
8. **Density-model cross-check** — P2PNet (official SHHA weights) or DM-Count as an independent
   head-point estimate. Use it to raise a *recall alarm* where detector counts drop abnormally,
   not as the primary source. PET has the best MAE but is **academic-licence only**.
9. **Fallen-person detection via bbox geometry.** A standing person's height after calibration is
   ~1.7 m; a long horizontal box far below that, held ≥ 5 s, is a fall. Nearly free, and it uses
   the homography we already have.

### Tier 3 — contingent, expensive

10. **Shared site map + geometric dedup** for overlapping cameras (same metric ground position
    ± 0.5 m within a drift tolerance) *before* any appearance model. This is what actually
    prevents double counting.
11. **OSNet ReID** (`osnet_ain_x1_0`, Torchreid) for gap filling between non-overlapping views.
    Cross-domain numbers are the argument: MSMT17→Market1501 gives 70.1 Rank-1 vs ResNet50's 46.3.
    Still a low-precision signal on 30-pixel crops — geometry first, appearance second.
12. **Dense flow (RAFT/RAPIDFlow) on downscaled, ROI-cropped frames** only for zones where tracks
    are missing (severe occlusion).
13. **SAM2-assisted zone editing** (interactive, human-corrected) — not autonomous zoning:
    segmentation is least reliable exactly under dense occlusion, where the ground is invisible.

### Tier 4 — research projects, not features

- Learned multi-view fusion (CVCS, view-contribution weighting) — needs a synthetic training pipeline.
- Hydrodynamic "crowd pressure" metrics — conceptually attractive, but the mapping from flow
  statistics to physical danger is **not quantitatively established**. Dashboard indicator at most,
  never an alarm trigger. Physical pressure needs force sensing; vision cannot give it.
- Deep video-anomaly models — standard benchmarks (Avenue, Ped2) are moderate density with obvious
  anomalies. They are not evidence of crush-precursor detection. Would need our own labelled surges.

---

## 3. Target architecture

Three hard truths drive the design:

1. **`cv2.VideoCapture` per camera is the ceiling.** Software H.264 decode costs roughly
   ~0.4–1 CPU core per 1080p30 stream plus ~3 MB per frame of memory traffic. 100 cameras =
   ~14 cores decoding and unbounded latency growth. Hardware decode is the fix, not tuning.
2. **Decode and infer must be decoupled.** NVDEC on an L40S does ~900 H.264 FPS per engine at
   1080p — i.e. 100 cameras × 10 FPS is barely one engine's worth *if you only decode what you analyse*
   (`drop-frame-interval=2` on a 30 FPS feed). Inference is the budget, not decode.
3. **Camera failures, not model accuracy, will dominate your incidents.** A frozen camera in a
   crowded zone reads as "no congestion" and suppresses the alert.

### Component layout

```
 cameras (RTSP / ONVIF)
        │  one pull per camera
        ▼
 ┌──────────────────┐   RTSP / HLS / WebRTC      ┌─────────────────────────────┐
 │ MediaMTX         │───────────────────────────▶│ browsers: tile wall (HLS),  │
 │ fan-in + proxy   │                            │ focus camera (LL-HLS/WHEP)  │
 └────────┬─────────┘                            └─────────────────────────────┘
          │ NVDEC decode (drop-frame-interval)
          ▼
 ┌──────────────────────────────┐   batched tensors   ┌──────────────────────┐
 │ decode/preprocess workers    │────────────────────▶│ Triton Inference     │
 │ 2–4 per GPU, 25–50 cams each │◀────────────────────│ detector + optional  │
 │ sharded by zone              │   detections        │ density/pose models  │
 └──────────────┬───────────────┘                     └──────────────────────┘
                │ zone aggregates, 1 Hz per camera
                ▼
 ┌──────────────────────────────┐        ┌────────────────────────────────┐
 │ aggregator / alert state     │───────▶│ TimescaleDB (1 s hypertable +  │
 │ machine (hysteresis, dedup)  │        │ continuous aggregates)          │
 └──────┬───────────────┬───────┘        └────────────────────────────────┘
        │ alerts        │ state
        ▼               ▼
 ┌─────────────┐  ┌───────────────┐      ┌──────────────────────────────┐
 │ SSE fan-out │  │ MinIO / S3    │      │ Prometheus + Grafana + DCGM  │
 │ (dashboards)│  │ alert clips   │      │ per-camera FPS, lag, XID     │
 └─────────────┘  └───────────────┘      └──────────────────────────────┘
```

### Why each choice

- **MediaMTX** as the single fan-in: every camera is pulled once, so three consumers don't triple
  the camera's uplink. It gives RTSP/RTMP/SRT/HLS/WebRTC in, a REST control API, per-path
  `runOnDemand` (don't pull the 80 cameras nobody is watching), and Prometheus metrics on `:9998`.
- **NVDEC decode, not FFmpeg.** Decode only the frames you analyse. At 1080p H.264 on Ada,
  ~900 FPS per engine; 100 cams × 8 FPS ≈ 1 engine.
- **Triton** for inference because of `dynamic_batching`, `priority_levels` (operator-focus cameras
  ahead of background tiles) and **`max_queue_size` + `timeout_action`** — the primitive that lets you
  *shed* stale frames instead of letting latency grow. Batch 8–16 at 640 px FP16 is the sweet spot.
- **Redpanda or NATS JetStream** for events — but note the volume is tiny: aggregate to **one event
  per camera per second** (~100 msg/s, well under 1 MB/s). This is an ops-burden decision, not a
  throughput one. Redis Streams is a cache here, not a system of record.
- **TimescaleDB** (Postgres) as the store: joins to the camera registry and alert rules in one place.
  Sizing: 100 zones × 1 Hz ≈ 0.9 GB/day raw, ~150 MB/day compressed → 30 d raw + 2 y of 1-min
  aggregates fits in ~40 GB. ClickHouse only when multi-venue.
- **SSE for feeds, WebSocket only for commands.** 50 operators × 100 zones = 5 000 small messages/s
  worst case, fanned out from *one* consumer. Never 5 000 video streams: render the 10×10 wall
  **server-side into one mosaic** (`nvmultistreamtiler` → NVENC → HLS at 1–2 Mbps, 1–2 FPS), and
  open LL-HLS/WebRTC only for the camera an operator focuses. 50 × 2 Mbps = 100 Mbps egress
  instead of 10 Gbps.

### Sizing for the Nebius VM

| VM | Spec | Runs |
|---|---|---|
| ingest | 8 vCPU / 16 GB / 1 Gbps | MediaMTX, ~150 Mbps inbound at 100 cams × 1.5 Mbps substream |
| gpu-1 | **L40S 48 GB**, `1gpu-8vcpu-32gb` | 3 decode workers (35 cams each) + Triton (2 models, batch 16) |
| gpu-2 | L40S, optional | N+1 failover — this is what makes it an operational tool |
| app | 8 vCPU / 32 GB | FastAPI, SSE fan-out, alert state machine, mosaic encoder, nginx |
| data | 8 vCPU / 32 GB / 500 GB NVMe | Redpanda, TimescaleDB, MinIO (90 d clips) |

Capacity planning: L40S runs YOLO-class detection end-to-end at roughly 700–1100 FPS FP16
(NVIDIA's published L40S figures: RT-DETR 643 FPS, PeopleNet+tracker 1114 FPS).
**100 cameras at 8 FPS = 800 inf/s → one L40S with 30–50 % headroom**, two for N+1.
Do not put 100 cameras on one A10: throughput drops exactly when a zone fills (more boxes → slower NMS).

---

## 4. What changes in *this* codebase, in order

| # | Change | Effort | Why first |
|---|---|---|---|
| 1 | **Homography + m² per zone** (4-point calibration in the zone editor) | 1–2 d | Every metric becomes physical; alerts can quote ppl/m² against 4/5/6 thresholds |
| 2 | **Alert lifecycle**: open → sustained → *cleared on recovery*, with cooldown | ½ d | Today alerts only open; operators need an "all clear" |
| 3 | **Camera health**: `last_frame_at`, frame-age watchdog, forced reconnect with exponential backoff, frozen-frame detection (scene variance) | 1–2 d | The #1 trust killer; a frozen camera suppresses congestion alerts |
| 4 | **Metric grid density + track velocity field + stop/counter-flow/queue flags** | 2–3 d | Highest-value analytics that need no new model |
| 5 | **Per-camera process isolation** (one worker process per camera or per shard) | 3–5 d | GIL and one slow `read()` currently stalls a whole shard; also enables restart-on-crash |
| 6 | **TimescaleDB instead of SQLite**; one row/camera/s | 2–3 d | Retention, continuous aggregates, concurrent writers |
| 7 | **SSE fan-out from a shared state cache** (Redis) | 2 d | Decouples dashboards from the detection loop, enables restart without dropping clients |
| 8 | **HDBSCAN hotspots + growth rate** | 3–5 d | Turns the dashboard into early warning |
| 9 | **MediaMTX + NVDEC decode workers + Triton** | 2–3 wk | The scale step; keep the current thread-per-camera path as the dev/CPU path |
| 10 | **Server-side mosaic for the wall view** | 1 wk | 50 operators without 5 000 video sessions |
| 11 | **OSNet ReID + shared site map** | 3–4 wk | Only if cameras overlap; geometry dedup first |

Items 1–4 are weeks, not months, and they change the product from "counting demo" to
"safety instrument". Items 9–11 are the hardware-era rewrite and should be gated on real camera count.

---

## 5. Failure modes to design for explicitly

| Failure | Symptom | Mitigation |
|---|---|---|
| Camera offline / RTSP blip | frame age climbs; some SDKs get stuck in an infinite reconnect loop | Own supervisor on `last_frame_at` that force-restarts the source; backoff 1→30 s; OFFLINE after 5 tries; grey tile, never a stale image |
| **Frozen frame** | frame age 0, count constant — looks healthy | scene-hash/variance over 60 s; frozen cameras must never *clear* an alert |
| GPU OOM | Triton / worker dies | cap batch and instance count, 10 % VRAM reserve, restart at half batch |
| Clock drift | cross-camera "simultaneous" rules misfire | chrony with verified offset < 50 ms; stamp both camera PTS and ingest time; always compare with a tolerance window |
| Alert storm | one surge → hundreds of events → operator blindness | hysteresis (alert ≥10 s, clear only below 90 % for 30 s), per-zone cooldown, dedup in the state machine |
| Model drift after swap | counts jump 20 % | version every event with `model_version`; shadow-run 1 h on 5 % of cameras before promoting |
| Stale frame inferred as live | system reports "normal" during the incident | refuse to fire or clear alerts on events older than 2× the analytics interval |

---

## 6. Test footage

Real crowds are in `.data/samples/pilgrim/` (see `docs/FOOTAGE.md` for provenance and licence).
Two things to know before trusting a number from them:

- **Resolution is the binding constraint.** The Archive.org films are 320–640 px wide, so a person
  is 10–25 px tall. YOLO26n at 1280 px finds 2–16 people per frame on them; that is a detector
  limit, not a crowd measurement. They are useful as *legacy-CCTV stress tests* (what happens when
  the source is bad), not as density ground truth.
- **High-density proves the case for a density model.** On the 4K Tawaf and Kumbh clips, detector
  recall collapses where pilgrims interlock — which is precisely the regime where a head-point model
  (P2PNet/DM-Count) is the right second opinion, and where foot-point tracking breaks down.

Re-fetch or extend the set with `python tools/fetch_footage.py` (Archive.org items are CC-BY where
noted; YouTube items are marked evaluation-only and must not be redistributed).

---

## 7. Sources

- NVIDIA NVDEC application note (per-engine decode FPS) — https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html
- DeepStream performance on L40S — https://docs.nvidia.com/deepstream/text/DS_Performance
- `nvurisrcbin` (drop-frame-interval, rtsp reconnect) — https://docs.nvidia.com/deepstream/text/DS_plugin_gst-nvurisrcbin
- Triton dynamic batcher (queue size, timeout_action, priorities) — https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
- TensorRT explicit quantization — https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-explicit-quantization.html
- MediaMTX configuration reference — https://mediamtx.org/docs/references/configuration-file
- TimescaleDB hypertable sizing — https://www.tigerdata.com/docs/build/performance-optimization/improve-hypertable-performance
- ClickHouse insert strategy — https://clickhouse.com/docs/concepts/best-practices/selecting-an-insert-strategy
- Awesome-Crowd-Counting dataset/model index — https://github.com/gjy3035/Awesome-Crowd-Counting
- P2PNet (official SHHA weights) — https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet
- DM-Count (official weights) — https://github.com/cvlab-stonybrook/DM-Count
- PET (ICCV'23, best MAE, academic licence) — https://github.com/cxliu0/PET
- Torchreid model zoo (OSNet cross-domain numbers) — https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
- VelocityNet — density-aware velocity anomaly for crowds — https://arxiv.org/html/2510.18187v1
- Zero-shot SAM crowd counting (segmentation limits in dense crowds) — https://arxiv.org/html/2402.17514v2
- Crowd disaster density thresholds — https://www.sciencedirect.com/science/article/pii/S2212420924004151
