# plan.md — what is integrated today

Audit of [`plan.md`](PLAN.md) against the code in this repository and a live
RTX PRO 6000 instance running five sample clips, November 2026.

Method: every section of the plan was mapped to code, then checked against the
running API (`/api/system`, `/api/cameras`, `/api/health`, `/api/models`,
`/api/audit`, `/api/outbox`, `/api/alerts`) and against the 69 tests.
Numbers below are measured, not claimed.

## The two Crowd Shield models (plan §5.1, §5.5) — integrated

| | PET | DM-Count |
|---|---|---|
| Checkpoint | `models/pet_shab.pth` | `models/dmcount_shab.pth` |
| SHA-256 (checked on load) | `5a7c6590e619…` | `d6f72bffcd1e…` |
| Architecture | PET point model, VGG16-BN backbone, quadtree | DM-Count, VGG19 + 2× upsample head |
| Role | head points, dense crowds | density map, aggregate cross-check |
| Wrapper | `backend/app/dense.py` → `DensePointEstimator` | `backend/app/dense.py` → `DenseDensityEstimator` |
| Definitions | `backend/services/dense` (academic-use) | same |

plan §5.5's ten steps: weights kept in an ignored local directory (1), checksums
pinned and verified before every load (2), original definitions vendored (3),
`CrowdShieldPointEstimator` (4) and `CrowdShieldDensityEstimator` (5) built,
BGR→RGB and ImageNet normalisation applied by the vendored code (6), model
resize/tiling untouched (7), points mapped back to source pixels and density
integrated over normalised polygons (8), per-zone projection in
`_point_zone_counts` / `_density_zone_counts` (9), and shadow mode: both stay
`validated: false`, so they can never become `selected_count` (10).

The PET count correction (`1.375 × raw^0.942`) is implemented but **off by
default** (`CROWD_PET_CORRECTION`), per plan §5.5 and the model card's warning
that it was fitted on ShanghaiTech, not on Indian ghat footage.

### Measured on the five sample clips

Detector = YOLO26n, full-frame 640 plus the 3×3/1280 detail pass.

| Clip | Detector | PET head points | DM-Count | PET latency | DM latency |
|---|---:|---:|---:|---:|---:|
| Getty 683716020 | 67–95 | 434–480 | 468–471 | 32–58 ms | 12–19 ms |
| Getty 2289512996 | 4–8 | 17–33 | 50–66 | 41–147 ms | 15–20 ms |
| Getty 685608730 | 15–23 | 16–17 | 13–18 | 44 ms | 16–50 ms |
| 12269404 1080p crowd | 145–154 | 383–388 | 385–391 | 90 ms | 24–25 ms |
| VIRAT 010204 | 9–15 | 11–18 | 6–8 | 77–90 ms | 28–30 ms |

Cost of running both every 5 s / 6 s per camera: GPU memory 2.6 GB → 6.3 GB,
frame-age p95 81–101 ms, **0 dropped frames**, inference rate unchanged at
3.9–4.4 fps per camera.

These are *independent estimates*: the API returns `detector_count`,
`head_point_count` and `density_count` side by side and `selected_count: null`.
On the dense clips the two dense models agree with each other (~385) and sit
~2.5× above the detector — which is the head-vs-body gap the plan describes. On
the sparse clips they disagree with each other; that disagreement is left
visible rather than averaged away.

### Dense consensus (plan §5.2/§5.3, implemented)

When PET and DM-Count agree within 25%, the router publishes their mean as
`consensus_count` and the camera reports `dense_consensus`: the dashboard shows
`≈N` (`MODEL CONSENSUS · ESTIMATE`) and the advisory alert names both estimates
and the spread. This is the point where the dense models actually improve the
operator's number instead of sitting beside it:

| Clip | Detector floor | PET | DM-Count | Consensus | Spread |
|---|---:|---:|---:|---:|---:|
| 12269404 1080p crowd | ≥156 | 389 | 383 | **≈386** | 1.6% |
| Getty 683716020 | ≥74 | 472 | 419 | **≈446** | 11.9% |
| Getty 685608730 | 19 | 16 | 15 | ≈15 | 9.2% |
| Getty 2289512996 | ≥10 | 27 | 49 | — (disagree) | 57% |
| VIRAT 010204 | ≥11 | 16 | 7 | — (disagree) | 56% |

`selected_count` stays null in every row: a consensus is corroboration, not a
validation, and capacity alerts remain suppressed while the counting is
unverified. When the two disagree, the camera falls back to the detector floor
(`≥N`) rather than inventing a middle value.

## Section-by-section status

| plan.md | Status | Evidence |
|---|---|---|
| §3 current foundation | Done | all listed modules present; `python -m pytest tests/ -q` → 69 passed |
| §5.1 visible-person detector | **Done** | YOLO26n/11n/8n + RT-DETR-l/x integrated and switchable; RF-DETR and D-FINE adapters behind optional packages. Default stays YOLO26n: on the 1080p clip it finds 156 vs RT-DETR's 36, because escalation only triggers below 6 |
| §5.1 head-point model | **Done** | PET live, `available: true`, per-camera counts and latencies in `/api/cameras` |
| §5.1 density model | **Done** | DM-Count live, density map integrated per zone, `available: true` |
| §5.2 runtime modes | Done | `dense`, `moderate`, `sparse`, `degraded_review`, `detector_lower_bound` observed live |
| §5.3 never add counts | Done | router keeps sources separate; no blending anywhere in the pipeline |
| §5.4 adapter contracts | Done | `DetectionResult`, `PointResult`, `DensityResult` in `app/adapters.py` |
| §5.5 Crowd Shield steps | **Done** | see table above; checksums verified; correction off |
| §6 camera placement / calibration | Done (platform) | calibration points, homography, versioning, per-camera approval; no cameras were physically moved |
| §7.1 camera health | Done | frozen/stale/dark/blurry/low-contrast states in `/api/health` |
| §7.2 track-based counting | Done | tracked vs raw counts, grace period, expiry |
| §7.3 durable alerts | **Done** | outbox, retry/backoff, idempotency, ack, close, escalate, CSV export; **delivery proven on the GPU host** — signed POST accepted in 0.03 s, `0` pending, `0` failed. Transport stays off until `ALERT_WEBHOOK_URL` is set |
| §7.4 dense-crowd estimation | **Done** | PET + DM-Count, periodic (5 s/6 s), advisory, zone-aware |
| §7.5 ground-plane calibration | Done | API + dashboard drawing + version bump on save |
| §7.6 model management | Done | `/api/models`, warm-before-swap, audit rows, per-camera profiles |
| §7.7 image quality | Done | quality snapshot per camera, poor frames do not produce confident counts |
| §7.8 capture/inference separation | Done | bounded queues, newest-frame policy, per-camera tracker state |
| §7.9 batched inference | Done | `CROWD_BATCH_SIZE=4`, queue depth and frame age exposed |
| §7.10 hardware decoding | **Blocked by the image** | Verified on the GPU host: OpenCV reports `GStreamer: NO`, so NVDEC cannot open a pipeline and FFMPEG is what runs. Needs an OpenCV build with GStreamer + nvdec plugins |
| §7.11 observability | Done | `/metrics`, plus new `head_point_count`/`density_count` gauges and per-estimator latency |
| §8 deployment architecture | **Two of three tiers** | `app/district.py`: a node publishes a signed camera/alert summary every 15 s, a district accepts it on `/api/district/ingest` and serves merged totals plus stale-node flags on `/api/district/overview` (5 tests). No central tier |
| §9 data protection | Partial | no face recognition, evidence redaction, retention limits, audit trail; **optional API tokens with operator/viewer roles**, measured 401/200/403/200; no per-user accounts or per-token audit attribution |
| §10 testing | Partial | 69 tests (consensus, families, auth, district), `tools/bench_models.py`, and a labelled accuracy run with `tools/eval_mall.py` (CUHK Mall: MAE 15.33 YOLO26n, 9.23 RT-DETR-l); no ghat ground truth exists |
| §11 rollout phases | In progress | phases 1–3 active, 4–6 not started |
| §12 acceptance gates | Partial | see below |

## Acceptance gates (§12)

| # | Gate | Status |
|---|---|---|
| 1 | No failed camera presented as safe | Met — health states suppress confident counts |
| 2 | Dense scenes visibly degrade | Met — `detector_lower_bound` badge with `≥` on all five clips |
| 3 | Head-only crowds give an independent estimate | **Met** — PET/DM counts live beside the detector |
| 4 | Detector/dense disagreement is visible | **Met** — three numbers shown, none blended |
| 5 | Alert delivery survives network loss | **Met** — HMAC-signed delivery verified end-to-end on the GPU host |
| 6 | Calibration approved per critical zone | **Met on one camera** — homography v1 + 20 samples (MAPE 2.6%) approved by `drill-operator`; the other four cameras stay unapproved |
| 7 | Model/checkpoint licences documented | Documented; **academic-use approval still required** before operational use |
| 8 | p95 frame age within control-room budget | Met on this hardware — 81 ms p95, 0 drops |
| 9 | Operators can acknowledge/escalate | Met |
| 10 | Emergency workflow drilled | **Met** — alert detected → delivered in 0.03 s → acknowledged 4.8 s later |

## Measured acceptance runs (November 2026)

Gates 5, 6 and 10 were run against the live instance, with a local
HMAC-verifying receiver on `127.0.0.1:8199` and the outbox pointed at it.

| Gate | Run | Result |
|---|---|---|
| 5 — delivery survives loss | full-frame zone with `capacity: 1` on the densest camera | signed POST accepted in **0.03 s**, `signature_ok: true`, `event: crowd_alert` / `schema_version: 1.1`, outbox `pending: 0, failed: 0` |
| 6 — calibration approved | 4-point homography (v1, 4 inliers), 20 validation samples at 2.6 % MAPE, then promotion | `approved: true`, `approved_by: drill-operator`, 30-day expiry, `active_source: head_point` while `CROWD_POINT_MODEL_VALIDATED=1` |
| 10 — emergency drill | alert → webhook → operator ack | `acked: 1`, `alert.ack` audit row, ack 4.8 s after delivery |

Two properties were checked while doing it: the receiver returned 401 to an
unsigned probe, so a signature regression fails the run instead of passing it;
and after restarting with the validation flags back off, the approval record is
kept but `active_source` drops to `none` — an unvalidated checkpoint cannot
silently become the operational count. The drill zone was removed afterwards.

API auth on the same instance: anonymous read 401, viewer read 200, viewer write
403, operator write 200, `/api/health` and `/static/` public.

## Deliberately not integrated

- **RF-DETR / D-FINE accuracy.** Adapters and registry entries exist and the
  picker lists them, but neither package is installed in this build, so they are
  untested here. Only YOLO and RT-DETR have measured numbers on this hardware.
- **Per-user accounts** (§9). Auth is a shared operator token and a shared
  viewer token, so the audit trail records an action and a role, not a person,
  and there is no login, session or password reset.
- **Central tier** (§8). Edge nodes and a district aggregator exist; a central
  tier (fleet control, cross-district rollup, TimescaleDB) does not.
- **Promotion of PET/DM-Count to operational counts** (§5.3, §12.6). Requires a
  per-camera validation sample (≥20 manual counts, MAPE ≤ 20%) signed off in the
  dashboard; until then `selected_count` stays null.

## Enabling and checking it

```bash
CROWD_POINT_BACKEND=dense       # PET head points
CROWD_DENSITY_BACKEND=dense     # DM-Count density
CROWD_POINT_PERIOD_S=5                # advisory cadence, per camera
CROWD_DENSITY_PERIOD_S=6
CROWD_MODELS_DIR=/path/to/models      # optional, defaults to ./models
CROWD_PET_CORRECTION=1                # opt-in; off by default
```

```bash
curl -s localhost:8000/api/system | jq '.head_points, .density'
curl -s localhost:8000/api/cameras | jq '.[0].metrics | {count, head_point_count, density_count, estimators}'
```

If a checkpoint is missing or altered, the estimators report
`available: false` with the reason, the audit trail gets a `points.load.failed`
or `density.load.failed` row, and the detector path keeps running.
