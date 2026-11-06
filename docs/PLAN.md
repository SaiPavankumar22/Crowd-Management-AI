# Godavari Crowd Intelligence Platform
# Production Upgrade and Model Integration Plan

## 1. Purpose

This document is the consolidated implementation plan for making Godavari
suitable for a controlled crowd-monitoring deployment during **Godavari
Pushkaralu 2027**.

The plan assumes that model fine-tuning is not immediately available. The first
production path therefore uses pretrained models, camera calibration, model
routing, shadow evaluation and conservative operator review.

Godavari Pushkaralu 2027 is currently reported as a 12-day event from **June 26
to July 7, 2027**, with planning estimates of **8–10 crore pilgrims**, peak-day
footfall approaching one crore, and approximately **440 ghats across six
districts**.

The system must not attempt to count the entire event from one central server.
It should monitor many operational zones locally and send reliable metrics and
alerts to district and central control rooms.

## 2. Core safety principle

The current observation is fundamental:

> In a dense crowd, people at the front and sides may show full bodies, while
> people in the middle may show only heads. A full-body bounding-box detector
> counts the visible bodies and skips many head-only people.

Therefore:

```text
Full-body detector:
    visible people, tracking, flow and line crossing

Head-point model:
    people whose heads are visible but bodies are occluded

Density model:
    aggregate count and density when individual localization is unreliable
```

The system must expose uncertainty. A low detector count in a dense-looking
scene must never be interpreted as proof that the zone is safe.

## 3. Current application foundation

Important current modules:

- [`app/main.py`](app/main.py) — REST, WebSocket, MJPEG, health and model APIs.
- [`app/pipeline.py`](app/pipeline.py) — camera workers, tracking and zone flow.
- [`app/detector.py`](app/detector.py) — current YOLO model manager.
- [`app/adapters.py`](app/adapters.py) — detector and density result contracts.
- [`app/router.py`](app/router.py) — sparse, moderate, dense and degraded modes.
- [`app/detail.py`](app/detail.py) — gated tiled/detail inference.
- [`app/tiling.py`](app/tiling.py) — overlapping tile generation and NMS.
- [`app/scheduler.py`](app/scheduler.py) — bounded queues and batching.
- [`app/analytics.py`](app/analytics.py) — analytics dead-man switch.
- [`app/health.py`](app/health.py) — frozen, stale and image-quality checks.
- [`app/calibration.py`](app/calibration.py) — ground-plane homography support.
- [`app/outbox.py`](app/outbox.py) — durable alert delivery.
- [`app/store.py`](app/store.py) — SQLite history, audit and outbox state.
- [`app/mongo_sync.py`](app/mongo_sync.py) — optional MongoDB archive.
- [`static/`](static/) — operator dashboard.

Already implemented:

- Camera health monitoring.
- Frozen, stale, dark, blurry and low-contrast states.
- Track-based zone membership.
- Tiled inference for small visible people.
- Dense-scene mode routing.
- Detector/density result contracts.
- Durable alert outbox and acknowledgement.
- Camera calibration primitives.
- Runtime model/profile management.
- Bounded inference queues and optional batching.
- Decoder selection and software fallback.
- Prometheus-style metrics.
- MongoDB archive support that preserves local SQLite as canonical.
- Input-size limits and source-credential redaction.
- Architecture and camera-management tests.

The remaining work is primarily production model integration, operational
validation, camera deployment and safety governance.

## 4. Target architecture

```text
CCTV / VMS / approved drone feeds
              |
              v
      Edge analytics node
       |       |       |
       |       |       +--> camera health and quality
       |       +----------> bounded capture and decoder
       +------------------> inference scheduler
                              |
             +----------------+----------------+
             |                |                |
       body detector     head-point model   density model
             |                |                |
          tracker       point zones       density zones
             |                |                |
             +----------------+----------------+
                              |
                   calibration and mode router
                              |
                    occupancy, flow and alerts
                              |
                   district control room
                              |
                 central Pushkaralu command room
```

The central system should receive counts, health, alerts, thumbnails and
selected evidence clips. It should not require every raw stream to be
continuously decoded at one central location.

## 5. Model integration plan

### 5.1 Model responsibilities

#### Visible-person detector

Use a box detector for:

- Visible person boxes.
- ByteTrack or BoT-SORT identities.
- Entry and exit line crossings.
- Flow direction.
- Walking speed.
- Queue movement.
- Sparse and moderate crowd zones.

Candidate models:

1. RT-DETRv2-S/M.
2. D-FINE-S/M.
3. YOLO26-S/M as the easiest current fallback.
4. RF-DETR-M/L after license and deployment review.

The candidate must be selected using Godavari footage and hardware benchmarks,
not COCO mAP alone.

#### Head-point model

Use a point model when heads are visible but bodies are occluded.

First candidate:

- PET from the Crowd Shield model repository.

It can provide:

- Head points.
- Approximate person locations.
- Dense-crowd heatmaps.
- Zone-level point counts.

It cannot replace normal identity tracking.

The `Khushi-7/crowd-shield-counting` weights are marked academic/research use.
They require licensing approval before commercial or operational Pushkaralu use.

#### Density-map model

Use DM-Count for:

- Aggregate zone counts.
- Density maps.
- Crowd-growth detection.
- Detector under-count warnings.

It should not be used for:

- Person identity.
- Line crossing.
- Persistent tracking.
- Unreviewed critical safety decisions.

### 5.2 Runtime modes

#### Sparse

```text
Body detector + tracker
```

The detector is the operational count source.

#### Moderate

```text
Body detector + tracker
Density or head-point model runs periodically as a cross-check
```

The detector remains primary, but disagreement is visible.

#### Dense

```text
Head-point or density model for aggregate occupancy
Body detector remains diagnostic
Identity tracking is marked unreliable
```

#### Unknown/degraded

```text
Camera or model quality is insufficient
No automatic safety count is published
Operator review is required
```

### 5.3 Never add model counts

Do not calculate:

```text
detector_count + PET_count + DM_Count_count
```

This will double-count people.

Expose independent estimates:

```json
{
  "detector_count": 42,
  "head_point_count": 88,
  "density_count": 83,
  "selected_count": null,
  "count_source": "operator_review",
  "confidence": "low",
  "mode": "dense",
  "reason": "body_detector_under_count"
}
```

Only a validated camera policy may promote a density or point count to
`selected_count`.

### 5.4 Adapter interfaces

Every detector backend should return a common result:

```python
DetectionResult(
    boxes,
    confidences,
    class_ids,
    inference_ms,
    model_name,
    model_version,
)
```

Point models should return:

```python
PointResult(
    points,
    raw_count,
    corrected_count,
    confidence,
    inference_ms,
    model_name,
    model_version,
)
```

Density models should return:

```python
DensityResult(
    density_map,
    total_count,
    zone_counts,
    confidence,
    inference_ms,
    model_name,
    model_version,
)
```

The pipeline must depend on these contracts rather than Ultralytics-specific
objects.

### 5.5 Hugging Face Crowd Shield integration

Repository:

<https://huggingface.co/Khushi-7/crowd-shield-counting>

Files:

- `pet_shab.pth`
- `dmcount_shab.pth`

Integration steps:

1. Download weights only into a local ignored model directory.
2. Verify checksums and record model version.
3. Import the original network definitions from the approved source.
4. Build `PETPointAdapter`.
5. Build `DMCountAdapter`.
6. Convert BGR frames to the expected color format.
7. Match the original resize and normalization procedure.
8. Restore coordinates from model output to original-frame coordinates.
9. Project points and density maps into calibrated zones.
10. Run in shadow mode before using any result for alerts.

The existing TorchScript density adapter cannot directly load these `.pth`
files. A dedicated adapter is required.

The PET correction:

```text
corrected_count = 1.375 × raw_count^0.942
```

must remain disabled for operational decisions until it is validated on local
camera views. It was fitted on ShanghaiTech data and may not transfer to Indian
ghat footage.

## 6. Camera placement and calibration

Camera placement is likely to improve operational accuracy more than changing
between two similar detector families.

Deploy separate cameras for:

- Ghat entry.
- Ghat exit.
- Queue corridors.
- Bathing steps.
- River-edge danger zones.
- Bridge approaches.
- Barricade funnels.
- Emergency lanes.
- Vehicle/pedestrian crossings.
- Medical camp approaches.

Avoid using one wide camera to infer precise occupancy across an entire ghat.

For each critical camera:

1. Select four or more ground-plane reference points.
2. Enter real-world coordinates or measured distances.
3. Calculate a homography.
4. Project detector feet or head points into world coordinates.
5. Calculate usable zone area in m².
6. Define a capacity with the responsible safety authority.
7. Save calibration version and operator approval.

Benefits:

- Comparable density between cameras.
- More meaningful people/m².
- Better capacity thresholds.
- Better zone-level density maps.
- Easier post-event audit.

## 7. Full improvement backlog

### 7.1 Camera health and dead-camera detection

**Description**

Detect stale streams, frozen frames, black frames, darkness, blur, low contrast
and repeated decode failures.

**Integration**

- Update health on every decoded frame.
- Expose health in REST and WebSocket payloads.
- Show camera health separately from crowd level.
- Add recovery hysteresis.
- Prevent a failed camera from displaying `OK`.

**Benefit**

A dead camera showing zero people is a dangerous false negative.

### 7.2 Track-based zone counting

**Description**

Count unique active track IDs instead of counting raw boxes repeatedly.

**Integration**

- Maintain per-camera track membership.
- Add a short grace period for temporary occlusion.
- Expire lost tracks after a configured timeout.
- Expose raw and tracked counts separately.

**Benefit**

Prevents repeated detections from inflating occupancy and improves alert
stability.

### 7.3 Durable external alerts

**Description**

Persist alerts before delivery and retry them after network failure.

**Integration**

- SQLite outbox remains local authority.
- Webhook, email, SMS, Teams and Slack transports can consume the outbox.
- Add idempotency keys.
- Add acknowledgement and escalation.
- Add evidence thumbnails and confidence.

**Benefit**

An operator does not lose a critical alert because a webhook or network was
temporarily unavailable.

### 7.4 Dense-crowd estimation

**Description**

Use PET and DM-Count as separate dense-scene estimators.

**Integration**

- Run them periodically, not necessarily on every video frame.
- Keep their output advisory until camera validation.
- Use calibrated zone masks.
- Mark identity tracking unreliable in dense mode.

**Benefit**

Head-only and heavily occluded people become visible to the analytics system.

### 7.5 Ground-plane calibration

**Description**

Convert image-space measurements into world-space density.

**Integration**

- Add calibration UI.
- Version every calibration.
- Require operator approval.
- Reject invalid or degenerate homographies.

**Benefit**

A density threshold becomes meaningful across different camera angles.

### 7.6 Runtime model management

**Description**

Allow model backends, profiles and versions to be listed and switched safely.

**Integration**

- Add a model registry.
- Load and warm a model before replacing the active one.
- Record model changes.
- Report model version in every result.
- Add per-camera model profiles.

**Benefit**

Different camera types can use different accuracy and latency profiles.

### 7.7 Image quality and low-light handling

**Description**

Measure brightness, contrast, blur, saturation and frame age.

**Integration**

- Add per-camera thresholds.
- Show quality history.
- Distinguish bad visibility from a genuinely empty zone.
- Prevent poor-quality frames from generating confident safety counts.

**Benefit**

Operators can fix camera problems instead of misinterpreting model failure as
low occupancy.

### 7.8 Capture and inference separation

**Description**

Separate decoding, frame buffering, inference, tracking and publishing.

**Integration**

- Use bounded per-camera queues.
- Keep the newest frame.
- Drop old frames rather than allowing latency to grow.
- Preserve camera-local tracker state.

**Benefit**

The control room receives current information instead of delayed analysis.

### 7.9 Batched inference

**Description**

Batch frames with compatible model, device and image-size settings.

**Integration**

- Use configurable batch size.
- Use maximum batch wait time.
- Measure queue depth and frame age.
- Keep batch size 1 as a CPU-safe fallback.

**Benefit**

GPU utilization improves when many cameras are active.

### 7.10 Hardware video decoding

**Description**

Use GStreamer, FFmpeg, NVDEC or DeepStream where the edge hardware supports
them.

**Integration**

- Detect available backend.
- Fall back to OpenCV software decoding.
- Expose active decoder per camera.
- Benchmark CPU and latency impact.

**Benefit**

Video decoding does not consume the CPU budget needed for inference and
orchestration.

### 7.11 Observability

**Description**

Make every camera, model and alert explainable.

**Integration**

Export:

```text
camera_frames_total
camera_decode_failures_total
camera_frame_age_seconds
camera_quality_score
inference_batches_total
inference_seconds_total
inference_queue_depth
inference_queue_drops_total
detector_count
head_point_count
density_count
zone_occupancy_ratio
alert_delivery_failures_total
```

Add structured logs, Prometheus collection, Grafana dashboards and audit
events.

**Benefit**

Operators can distinguish camera failure, model failure, queue latency and a
real crowd event.

## 8. Deployment architecture

### Edge analytics node

Each edge node should process a measured cluster of cameras, for example 8–16
depending on resolution, GPU and model mix.

Each node should have:

- Hardware decoder.
- Local model cache.
- Inference scheduler.
- Local SQLite history.
- Durable alert outbox.
- Short evidence buffer.
- Health endpoint.
- Network reconnect logic.

### District control room

The district layer should display:

- Zone occupancy.
- Crowd growth rate.
- Camera health.
- Model confidence.
- Open alerts.
- Acknowledgements.
- Available emergency resources.
- Selected live evidence.

### Central command center

The central layer should aggregate metrics and coordinate resources. It should
not be the only place where alerts can be generated.

If the central network fails, the edge and district layers must continue
operating.

## 9. Data protection and governance

The system is for counting and crowd safety, not identification.

Required controls:

- No face recognition.
- No identity database.
- Blur faces in retained evidence where practical.
- Purpose-limited analytics.
- Clear CCTV and AI-analytics notices.
- Role-based dashboard access.
- Encryption in transit and at rest.
- Raw-video retention limits.
- Audit logs for access and exports.
- Documented breach procedures.
- Human confirmation for critical action.

The real `.env` must never be committed. Commit only `.env.example` with
placeholders.

## 10. Testing and validation

### Unit tests

Test:

- Calibration validation.
- Homography projection.
- Zone membership.
- Track expiry.
- Dense-mode routing.
- Detector/density disagreement.
- Alert sustain and cooldown.
- Outbox retry and idempotency.
- Source credential redaction.
- Upload size limits.

### Integration tests

Test:

- Camera add/remove.
- Camera persistence.
- Camera profile changes.
- WebSocket health updates.
- Model switching.
- Calibration API.
- Alert acknowledgement and close.
- MongoDB outage behavior.
- Upload cleanup on failure.

### Model benchmark

For every model and camera, record:

```text
manual_count
detector_count
head_point_count
density_count
absolute_error
relative_error
line_crossing_error
alert_latency
inference_latency
end_to_end_fps
frame_age
GPU_memory
```

Evaluate:

- Sparse scenes.
- Moderate scenes.
- Dense scenes.
- Head-only crowds.
- Overhead views.
- Night views.
- Backlit views.
- Rain and haze.
- Camera freezes.
- Network interruptions.

Do not approve a model based only on ShanghaiTech or COCO numbers.

## 11. Rollout phases

### Phase 1: Platform hardening

- Complete health and diagnostics.
- Complete adapter interfaces.
- Complete alert outbox.
- Add calibration wizard.
- Add audit logging.
- Add operator permissions.

### Phase 2: Offline model evaluation

- Integrate PET adapter.
- Integrate DM-Count adapter.
- Benchmark against the current detector.
- Verify model preprocessing and output scaling.
- Record licenses and checksums.

### Phase 3: Shadow operation

- Run detector and dense models together.
- Keep dense models advisory.
- Compare outputs with manual counts.
- Tune camera-specific thresholds.
- Review model disagreement.

### Phase 4: Controlled pilot

- Choose representative ghats.
- Include sparse, moderate, dense, night and difficult cameras.
- Test operator response.
- Test network and camera failure.
- Run emergency drills.

### Phase 5: Edge scale-out

- Add edge GPU nodes.
- Add hardware decoding.
- Connect district control rooms.
- Add local failover.
- Freeze validated model versions.

### Phase 6: Event operations

- Daily camera-health review.
- No unreviewed model updates.
- No unreviewed calibration changes.
- Alert ownership by zone.
- Incident evidence preservation.
- Continuous support and replacement procedures.

## 12. Acceptance gates

Do not promote the system to operational use until these gates are satisfied:

1. No failed camera is presented as safe.
2. Dense scenes visibly enter degraded or dense mode.
3. Head-only crowds produce an independent advisory estimate.
4. Detector and dense-model disagreement is visible.
5. Alert delivery survives network interruption.
6. Camera calibration is approved per critical zone.
7. Model and checkpoint licenses are documented.
8. P95 frame age is within the control-room requirement.
9. Operators can acknowledge and escalate alerts.
10. Emergency teams have tested the response workflow.

Accuracy thresholds should be agreed with the responsible safety authority
after manual counting. They should not be invented from generic benchmark
scores.

## 13. Recommended first implementation sequence

1. Add PET and DM-Count adapters without changing current detector alerts.
2. Add head-point and density values to the dashboard.
3. Add dense-scene mode labels and model disagreement warnings.
4. Complete camera calibration UI.
5. Run shadow evaluation on representative local footage.
6. Approve only the models and cameras that pass local validation.
7. Add hardware decoding and edge deployment.
8. Add district-level operations and escalation.
9. Run failure and emergency drills.
10. Freeze the validated production configuration before the event.

## 14. Final recommendation

The best no-fine-tuning production strategy is:

```text
Visible people:
    RT-DETRv2-S/M or D-FINE-S/M
    with YOLO26 fallback

Visible heads in dense crowds:
    PET, only after license approval and local validation

Aggregate density:
    DM-Count as an independent cross-check

Tracking:
    ByteTrack/BoT-SORT only where body detections are reliable

Measurement:
    Per-camera calibration and people/m²

Operations:
    Edge processing, district control rooms and central aggregation

Safety:
    Explicit degraded states and human-confirmed critical actions
```

The most important improvement is not simply replacing the current YOLO model.
It is making Godavari recognize when full-body detection is no longer
trustworthy and switch to a calibrated head-point or density analysis path
without pretending that the result has individual identity accuracy.

## 15. References

- [Godavari Pushkaralu 2027 — Andhra Pradesh Government](https://godavaripushkaram2027.ap.gov.in/)
- [East Godavari District — Pushkaralu 2027](https://eastgodavari.ap.gov.in/godavari-pushkaralu-2027/)
- [The Hindu — AI crowd management plans](https://www.thehindu.com/news/national/andhra-pradesh/ai-to-help-manage-crowd-during-2027-godavari-pushkaralu-in-andhra-pradesh/article71388987.ece)
- [NDMA crowd-management guide](https://ndma.gov.in/sites/default/files/PDF/Reports/managingcrowdsguide.pdf)
- [Crowd Shield model](https://huggingface.co/Khushi-7/crowd-shield-counting)
- [Crowd Shield source repository](https://github.com/khushi-git-7/Crowd_shield)
- [RT-DETR repository](https://github.com/lyuwenyu/RT-DETR)
- [D-FINE repository](https://github.com/Peterande/D-FINE)
- [RF-DETR repository](https://github.com/roboflow/RF-detr)
- [DM-Count repository](https://github.com/cvlab-stonybrook/DM-Count)
- [P2PNet repository](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)
- [Crowd-counting article](https://medium.com/@kaamyakpant_67666/building-a-crowd-counting-model-using-deep-learning-e8b8e925674e)
